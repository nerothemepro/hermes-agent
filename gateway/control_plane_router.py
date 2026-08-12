"""Exact, fail-closed Telegram control-plane command router."""

from __future__ import annotations

import asyncio
import json
import os
import re
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Awaitable, Callable

from gateway.config import Platform

DEFAULT_PROJECT_PATH = Path("/workspace/hermes-agent-plugin")
DEFAULT_REGISTRY_DIR = Path("/opt/data/hermes/control-plane/runs")
PREPARE_BIN = "/workspace/hermes-agent-plugin/bin/hermes-control-plane-prepare"
HERSOCIAL_APPROVAL_BIN = "/workspace/hermes-agent-plugin/control-plane/hersocial-auto-post/start-hersocial-auto-post.sh"
RUN_ID_PATTERN = r"run_[a-z0-9]+_[a-z0-9]+"
GATE_ID_PATTERN = r"[a-z][a-z0-9_]*"
HERSOCIAL_POST_KEY_PATTERN = r"[a-z0-9][a-z0-9-]{2,80}"
SHA256_PATTERN = r"[a-f0-9]{64}"


@dataclass(frozen=True)
class RouterDecision:
    handled: bool
    response: str | None = None


async def _default_command_runner(argv: list[str], timeout_seconds: int) -> SimpleNamespace:
    def _run() -> SimpleNamespace:
        env = dict(os.environ)
        env.pop("HERMES_KANBAN_HOME", None)
        process = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            start_new_session=True,
        )
        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired as error:
            # The prepare helper can synchronously invoke sdtk-agent. Kill the
            # entire process group so a timeout cannot leave an orphaned child
            # mutating state after the owner received a fail-closed reply.
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
            raise TimeoutError from error
        return SimpleNamespace(returncode=process.returncode, stdout=stdout, stderr=stderr)

    return await asyncio.to_thread(_run)


class ControlPlaneRouter:
    """Owner-only exact command router, evaluated before gateway pairing/auth."""

    def __init__(
        self,
        config: dict | None,
        *,
        command_runner: Callable[[list[str], int], Awaitable[SimpleNamespace]] | None = None,
    ) -> None:
        config = config if isinstance(config, dict) else {}
        self.enabled = config.get("enabled") is True
        self.owner_id = str(config.get("owner_telegram_user_id") or "").strip()
        self.home_chat_id = self._normalize_chat_id(config.get("home_telegram_chat_id"))
        self.hersocial_approval_enabled = config.get("hersocial_approval_enabled") is True
        self.timeout_seconds = self._bounded_timeout(config.get("command_timeout_seconds"))
        self.project_path = Path(config.get("project_path") or DEFAULT_PROJECT_PATH).resolve()
        self.registry_dir = Path(config.get("registry_dir") or DEFAULT_REGISTRY_DIR).resolve()
        self.command_runner = command_runner or _default_command_runner

    @classmethod
    def from_gateway_config(cls, config: dict | None) -> "ControlPlaneRouter":
        raw = config.get("control_plane_router") if isinstance(config, dict) else None
        router_config = dict(raw) if isinstance(raw, dict) else {}
        owner_env = str(router_config.get("owner_telegram_user_env") or "HERMES_CONTROL_PLANE_OWNER_TELEGRAM_USER_ID")
        home_chat_env = str(router_config.get("home_telegram_chat_env") or "")
        router_config.setdefault("owner_telegram_user_id", os.environ.get(owner_env, ""))
        if home_chat_env:
            router_config.setdefault("home_telegram_chat_id", os.environ.get(home_chat_env, ""))
        return cls(router_config)

    @staticmethod
    def _normalize_chat_id(value) -> str:
        chat_id = str(value or "").strip()
        return chat_id.split(":", 1)[1] if chat_id.startswith("telegram:") else chat_id

    @staticmethod
    def _bounded_timeout(value) -> int:
        try:
            return min(30, max(1, int(value)))
        except (TypeError, ValueError):
            return 15

    async def handle(self, event) -> RouterDecision:
        if not self.enabled or getattr(event.source, "platform", None) != Platform.TELEGRAM:
            return RouterDecision(False)

        # First router decision: non-owners cannot reach parsing, registry, or CLI.
        sender_id = str(getattr(event.source, "user_id", "") or "")
        if not self.owner_id or sender_id != self.owner_id:
            return RouterDecision(True)

        text = (getattr(event, "text", "") or "").strip()
        # A configured home chat binds control commands to the owner group.
        # Normal owner conversation outside that group continues to the LLM unchanged.
        chat_id = self._normalize_chat_id(getattr(event.source, "chat_id", ""))
        if self.home_chat_id and chat_id != self.home_chat_id:
            if self._looks_like_control_attempt(text):
                return RouterDecision(True)
            return RouterDecision(False)
        if match := re.fullmatch(r"/site-audit\s+(docs|sdtk_public_web)", text):
            scope = "sdtk_public_web" if match.group(1) == "docs" else match.group(1)
            return await self._prepare("site_audit", {"scope": scope})
        if match := re.fullmatch(r"/research-brief\s+([^\r\n]{3,240})", text):
            return await self._prepare("research_brief", {"topic": match.group(1).strip()})
        if match := re.fullmatch(rf"(?:/status|STATUS)\s+({RUN_ID_PATTERN})", text):
            return self._status(match.group(1))
        if match := re.fullmatch(rf"APPROVE DISPATCH\s+({RUN_ID_PATTERN})", text):
            return await self._approve_dispatch(match.group(1))
        if match := re.fullmatch(rf"APPROVE GATE\s+({RUN_ID_PATTERN})\s+({GATE_ID_PATTERN})", text):
            return await self._approve_gate(match.group(1), match.group(2))
        if match := re.fullmatch(
            rf"APPROVE HERSOCIAL POST\s+({HERSOCIAL_POST_KEY_PATTERN})\s+({SHA256_PATTERN})", text
        ):
            if not self.hersocial_approval_enabled:
                return RouterDecision(True)
            return await self._approve_hersocial_post(match.group(1), match.group(2))
        if match := re.fullmatch(rf"CANCEL RUN\s+({RUN_ID_PATTERN})", text):
            return await self._cancel(match.group(1))

        if self._looks_like_control_attempt(text):
            return RouterDecision(True, self._syntax_refusal())
        return RouterDecision(False)

    @staticmethod
    def _looks_like_control_attempt(text: str) -> bool:
        return bool(re.match(r"(?i)^\s*(?:/|approve\b|cancel\b|status\b|run\b|audit\b|research\b)", text))

    @staticmethod
    def _syntax_refusal() -> str:
        return (
            "Exact syntax required; no action was taken.\n"
            "/site-audit docs\n/research-brief <topic>\n/status <run_id>\n"
            "APPROVE DISPATCH <run_id>\nAPPROVE GATE <run_id> <gate_id>\n"
            "APPROVE HERSOCIAL POST <post_key> <sha256>\nCANCEL RUN <run_id>"
        )

    async def _prepare(self, template: str, params: dict) -> RouterDecision:
        result = await self._command(["node", PREPARE_BIN, "--template", template, "--params", json.dumps(params)])
        if result is None:
            return RouterDecision(True, "Control-plane command is temporarily unavailable; no automatic retry was performed.")
        if result.returncode != 0:
            return RouterDecision(True, "Control-plane command failed closed; no dispatch occurred.")
        payload = self._json_output(result.stdout)
        if not isinstance(payload, dict) or payload.get("status") != "prepared_waiting_for_exact_dispatch_approval":
            return RouterDecision(True, "Control-plane command returned an invalid response; no dispatch occurred.")
        run_id = payload.get("run_id")
        preview = payload.get("preview") if isinstance(payload.get("preview"), dict) else {}
        if not isinstance(run_id, str) or not re.fullmatch(RUN_ID_PATTERN, run_id):
            return RouterDecision(True, "Control-plane command returned an invalid run id; no dispatch occurred.")
        return RouterDecision(
            True,
            "Preview prepared\n"
            f"run_id: {run_id}\n"
            f"tasks: {preview.get('task_count', 'unknown')}\n"
            f"gates: {preview.get('gate_count', 'unknown')}\n"
            f"profile: {preview.get('profile', 'unknown')}\n"
            f"deadline_minutes: {preview.get('deadline_minutes', 'unknown')}\n"
            f"cost_band: {preview.get('cost_band', 'unknown')}\n"
            f"APPROVE DISPATCH {run_id}",
        )

    async def _approve_dispatch(self, run_id: str) -> RouterDecision:
        if self._registry_record(run_id) is None:
            return RouterDecision(True, "Unknown run_id; no action was taken.")
        result = await self._command([
            "sdtk-agent", "run", "continue", "--project-path", str(self.project_path),
            "--run-id", run_id, "--confirm", "--json",
        ])
        return self._cli_result(result, "Dispatch accepted; monitor will report progress.")

    async def _approve_gate(self, run_id: str, gate_id: str) -> RouterDecision:
        state = self._state(run_id)
        if state is None or state.get("status") != "waiting_for_approval" or state.get("waiting_gate") != gate_id:
            return RouterDecision(True, "Unknown run_id or gate_id; no action was taken.")
        approved = await self._command([
            "sdtk-agent", "gate", "approve", "--project-path", str(self.project_path),
            "--run-id", run_id, "--gate", gate_id, "--approved-by", "owner", "--json",
        ])
        if approved is None or approved.returncode != 0:
            return self._cli_result(approved, "")
        continued = await self._command([
            "sdtk-agent", "run", "continue", "--project-path", str(self.project_path),
            "--run-id", run_id, "--json",
        ])
        return self._cli_result(continued, "Gate approved; run advanced through the audited CLI path.")

    async def _approve_hersocial_post(self, post_key: str, digest: str) -> RouterDecision:
        result = await self._command([
            HERSOCIAL_APPROVAL_BIN, "--record-approval", post_key, digest,
        ])
        if result is None:
            return RouterDecision(True, "HerSocial approval is temporarily unavailable; no automatic retry was performed.")
        if result.returncode != 0:
            return RouterDecision(True, "HerSocial approval failed closed; no post was published.")
        payload = self._json_output(result.stdout)
        if not isinstance(payload, dict) or payload.get("status") not in {"approved_pending_publish", "uploaded", "published"}:
            return RouterDecision(True, "HerSocial approval returned an invalid response; no post was published.")
        if payload.get("post_key") != post_key or payload.get("content_sha256") != digest:
            return RouterDecision(True, "HerSocial approval returned mismatched evidence; no post was published.")
        if payload.get("status") in {"uploaded", "published"}:
            video_url = payload.get("video_url")
            if not isinstance(video_url, str) or not re.fullmatch(r"https://[^\s]+", video_url):
                return RouterDecision(True, "HerSocial publish response lacked a valid permalink; no publish was confirmed.")
            if payload.get("status") == "uploaded":
                visibility_state = payload.get("visibility_state")
                if visibility_state not in {"unpublished", "unlisted", "private"}:
                    return RouterDecision(True, "HerSocial upload response lacked a valid non-public visibility state; no public post was confirmed.")
                return RouterDecision(True, f"HerSocial video uploaded for review ({visibility_state}): {video_url}")
            return RouterDecision(True, f"HerSocial post published: {video_url}")
        return RouterDecision(
            True, "HerSocial approval recorded; the attended publisher will report the final result.",
        )

    async def _cancel(self, run_id: str) -> RouterDecision:
        if self._registry_record(run_id) is None:
            return RouterDecision(True, "Unknown run_id; no action was taken.")
        result = await self._command([
            "sdtk-agent", "run", "cancel", "--project-path", str(self.project_path),
            "--run-id", run_id, "--json",
        ])
        return self._cli_result(result, "Run cancellation requested through the audited CLI path.")

    async def _command(self, argv: list[str]) -> SimpleNamespace | None:
        try:
            return await self.command_runner(argv, self.timeout_seconds)
        except (TimeoutError, subprocess.TimeoutExpired, OSError):
            return None

    def _cli_result(self, result: SimpleNamespace | None, success: str) -> RouterDecision:
        if result is None:
            return RouterDecision(True, "Control-plane command is temporarily unavailable; no automatic retry was performed.")
        if result.returncode != 0:
            return RouterDecision(True, "Control-plane command failed closed; no automatic retry was performed.")
        return RouterDecision(True, success)

    def _registry_record(self, run_id: str) -> dict | None:
        path = self.registry_dir / f"{run_id}.json"
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(record, dict) or record.get("run_id") != run_id:
            return None
        state_path = Path(str(record.get("state_path") or "")).resolve()
        expected = (self.project_path / ".sdtk" / "agent-runtime" / "runs" / run_id / "state.json").resolve()
        return record if state_path == expected else None

    def _state(self, run_id: str) -> dict | None:
        if self._registry_record(run_id) is None:
            return None
        path = self.project_path / ".sdtk" / "agent-runtime" / "runs" / run_id / "state.json"
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    def _status(self, run_id: str) -> RouterDecision:
        state = self._state(run_id)
        if state is None:
            return RouterDecision(True, "Unknown run_id; no action was taken.")
        root = self.project_path / ".sdtk" / "agent-runtime" / "runs" / run_id
        report = root / "reports" / "final_report.md"
        report_status = "available" if report.exists() else "not_generated"
        return RouterDecision(
            True,
            f"run_id: {run_id}\nstatus: {state.get('status', 'unknown')}\n"
            f"state_path: {root / 'state.json'}\nreport: {report}\nreport_status: {report_status}",
        )

    @staticmethod
    def _json_output(value: str) -> dict | None:
        try:
            parsed = json.loads(value)
        except (TypeError, json.JSONDecodeError):
            return None
        return parsed if isinstance(parsed, dict) else None
