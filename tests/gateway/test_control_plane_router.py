"""Contract tests for the fail-closed Phase C Telegram control-plane router."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import gateway.control_plane_router as router_module
from gateway.config import Platform
from gateway.control_plane_router import ControlPlaneRouter
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


OWNER_ID = "10001"


def _event(text: str, user_id: str = OWNER_ID, chat_id: str | None = None) -> MessageEvent:
    return MessageEvent(
        text=text,
        message_id="phase-c-test",
        source=SessionSource(
            platform=Platform.TELEGRAM,
            user_id=user_id,
            chat_id=chat_id or user_id,
            user_name="owner" if user_id == OWNER_ID else "other",
            chat_type="dm",
        ),
    )


def _router(*, enabled: bool = True, marketing_video_ep2_enabled: bool = False, runner=None) -> ControlPlaneRouter:
    return ControlPlaneRouter(
        {
            "enabled": enabled,
            "owner_telegram_user_id": OWNER_ID,
            "command_timeout_seconds": 7,
            "marketing_video_ep2_enabled": marketing_video_ep2_enabled,
        },
        command_runner=runner,
    )


@pytest.mark.asyncio
async def test_non_owner_drops_before_any_parse_or_cli() -> None:
    command_runner = AsyncMock()
    decision = await _router(runner=command_runner).handle(_event("APPROVE DISPATCH run_bad_bad", "20002"))

    assert decision.handled is True
    assert decision.response is None
    command_runner.assert_not_awaited()


@pytest.mark.asyncio
async def test_disabled_flag_preserves_normal_gateway_passthrough() -> None:
    command_runner = AsyncMock()
    decision = await _router(enabled=False, runner=command_runner).handle(_event("/site-audit sdtk_public_web"))

    assert decision.handled is False
    assert decision.response is None
    command_runner.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "approve it",
        "/site-audit",
        "/site-audit docs extra",
        "/site_audit sdtk_public_web",
        "APPROVE DISPATCH run_not-a-run",
        "/marketing-video ep2",
        "/marketing-video ep2-usage extra",
    ],
)
async def test_owner_invalid_or_partial_grammar_is_refused_without_cli(text: str) -> None:
    command_runner = AsyncMock()
    decision = await _router(runner=command_runner).handle(_event(text))

    assert decision.handled is True
    assert "Exact syntax" in (decision.response or "")
    command_runner.assert_not_awaited()


@pytest.mark.asyncio
async def test_unknown_gate_is_rejected_without_cli() -> None:
    command_runner = AsyncMock()
    decision = await _router(runner=command_runner).handle(_event("APPROVE GATE run_bad_bad wrong_gate"))

    assert decision.handled is True
    assert "Unknown run_id or gate_id" in (decision.response or "")
    command_runner.assert_not_awaited()


@pytest.mark.asyncio
async def test_normal_owner_message_passes_through_unchanged() -> None:
    decision = await _router().handle(_event("Please summarize the current health status."))

    assert decision.handled is False
    assert decision.response is None


def test_router_binds_exclusive_fence_to_home_chat_env(monkeypatch) -> None:
    monkeypatch.setenv("TELEGRAM_HOME_CHANNEL", "-100123")

    router = ControlPlaneRouter.from_gateway_config({
        "control_plane_router": {
            "enabled": True,
            "owner_telegram_user_id": OWNER_ID,
            "home_telegram_chat_env": "TELEGRAM_HOME_CHANNEL",
            "exclusive_control_plane_mode": True,
        }
    })

    assert router.home_chat_id == "-100123"
    assert router.exclusive_control_plane_mode is True


def test_command_runner_uses_windows_process_group_when_needed() -> None:
    assert router_module._command_start_kwargs(is_windows=True) == {
        "creationflags": getattr(router_module.subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
    }


def test_router_defaults_command_timeout_to_maximum_bounded_value() -> None:
    router = ControlPlaneRouter({"enabled": True, "owner_telegram_user_id": OWNER_ID})

    assert router.timeout_seconds == 30


def test_timeout_cleanup_uses_taskkill_tree_on_windows(monkeypatch) -> None:
    calls = []

    def _run(argv, **kwargs):
        calls.append((argv, kwargs))

    monkeypatch.setattr(router_module.subprocess, "run", _run)
    router_module._terminate_timed_out_process(SimpleNamespace(pid=12345), is_windows=True)

    assert calls == [(
        ["taskkill", "/F", "/T", "/PID", "12345"],
        {
            "stdout": router_module.subprocess.DEVNULL,
            "stderr": router_module.subprocess.DEVNULL,
            "check": False,
        },
    )]


@pytest.mark.asyncio
async def test_site_audit_uses_bounded_prepare_command_and_returns_preview() -> None:
    command_runner = AsyncMock(return_value=SimpleNamespace(
        returncode=0,
        stdout='{"status":"prepared_waiting_for_exact_dispatch_approval","run_id":"run_abc123_def456","preview":{"task_count":1,"gate_count":1,"profile":"herresearch","deadline_minutes":30,"cost_band":"low"}}',
        stderr="",
    ))

    decision = await _router(runner=command_runner).handle(_event("/site-audit sdtk_public_web"))

    assert decision.handled is True
    assert "run_abc123_def456" in (decision.response or "")
    assert "APPROVE DISPATCH run_abc123_def456" in (decision.response or "")
    argv = command_runner.await_args.args[0]
    assert argv[:3] == ["node", "/workspace/hermes-agent-plugin/bin/hermes-control-plane-prepare", "--template"]
    assert "site_audit" in argv


@pytest.mark.asyncio
async def test_hersocial_unpublished_upload_is_reported_as_reviewable_draft() -> None:
    post_key = "social-video-facebook-0123456789abcdef"
    digest = "a" * 64
    command_runner = AsyncMock(return_value=SimpleNamespace(
        returncode=0,
        stdout=(
            '{"status":"uploaded","post_key":"' + post_key + '",'
            '"content_sha256":"' + digest + '",'
            '"video_url":"https://www.facebook.com/reel/draft-example/",'
            '"visibility_state":"unpublished",'
            '"next_action":"manual_visibility_review_required"}'
        ),
        stderr="",
    ))
    router = ControlPlaneRouter(
        {
            "enabled": True,
            "owner_telegram_user_id": OWNER_ID,
            "hersocial_approval_enabled": True,
            "command_timeout_seconds": 7,
        },
        command_runner=command_runner,
    )

    decision = await router.handle(_event(f"APPROVE HERSOCIAL POST {post_key} {digest}"))

    assert decision.handled is True
    assert "uploaded for review" in (decision.response or "")
    assert "unpublished" in (decision.response or "")
    assert "https://www.facebook.com/reel/draft-example/" in (decision.response or "")
    assert "post published" not in (decision.response or "").lower()



@pytest.mark.asyncio
async def test_marketing_video_ep2_exact_command_prepares_fixed_template_only() -> None:
    command_runner = AsyncMock(return_value=SimpleNamespace(
        returncode=0,
        stdout='{"status":"prepared_waiting_for_exact_dispatch_approval","run_id":"run_abc123_def456","preview":{"task_count":7,"gate_count":5,"profile":"multi-profile","deadline_minutes":120,"cost_band":"high"}}',
        stderr="",
    ))

    decision = await _router(marketing_video_ep2_enabled=True, runner=command_runner).handle(_event("/marketing-video ep2-usage"))

    assert decision.handled is True
    assert "run_abc123_def456" in (decision.response or "")
    assert "APPROVE DISPATCH run_abc123_def456" in (decision.response or "")
    argv = command_runner.await_args.args[0]
    assert argv[:3] == ["node", "/workspace/hermes-agent-plugin/bin/hermes-control-plane-prepare", "--template"]
    assert "marketing_video_ep_usage" in argv
    assert "--params" in argv
    assert argv[argv.index("--params") + 1] == "{}"


@pytest.mark.asyncio
async def test_marketing_video_ep2_disabled_flag_creates_no_run() -> None:
    command_runner = AsyncMock()

    decision = await _router(marketing_video_ep2_enabled=False, runner=command_runner).handle(_event("/marketing-video ep2-usage"))

    assert decision.handled is True
    assert decision.response is None
    command_runner.assert_not_awaited()


@pytest.mark.asyncio
async def test_dispatch_timeout_reports_recorded_external_dispatch_without_retry(monkeypatch) -> None:
    run_id = "run_abc123_def456"

    async def _timeout(_argv, _timeout_seconds):
        raise TimeoutError

    router = _router(marketing_video_ep2_enabled=True, runner=_timeout)
    states = iter([
        {"status": "created", "tasks": {"research_evidence": {"status": "created"}}},
        {"status": "running", "tasks": {"research_evidence": {"status": "running_external", "external_ids": {"hermes_task_id": "t_new"}}}},
    ])
    monkeypatch.setattr(router, "_registry_record", lambda _run_id: {"run_id": run_id})
    monkeypatch.setattr(router, "_state", lambda _run_id: next(states))

    decision = await router.handle(_event(f"APPROVE DISPATCH {run_id}"))

    assert decision.handled is True
    assert "Dispatch submitted" in (decision.response or "")
    assert "unavailable" not in (decision.response or "")


@pytest.mark.asyncio
async def test_dispatch_nonzero_after_recorded_terminal_submission_reports_submitted(monkeypatch) -> None:
    run_id = "run_abc123_def456"
    command_runner = AsyncMock(return_value=SimpleNamespace(returncode=1, stdout="", stderr="worker blocked"))
    router = _router(marketing_video_ep2_enabled=True, runner=command_runner)
    states = iter([
        {"status": "running", "tasks": {"research_evidence": {"status": "ready"}}},
        {"status": "blocked", "tasks": {"research_evidence": {
            "status": "blocked",
            "external_ids": {"hermes_task_id": "t_terminal"},
        }}},
    ])
    monkeypatch.setattr(router, "_registry_record", lambda _run_id: {"run_id": run_id})
    monkeypatch.setattr(router, "_state", lambda _run_id: next(states))

    decision = await router.handle(_event(f"APPROVE DISPATCH {run_id}"))

    assert decision.handled is True
    assert "Dispatch submitted" in (decision.response or "")
    assert "failed closed" not in (decision.response or "")
    command_runner.assert_awaited_once()


@pytest.mark.asyncio
async def test_cli_timeout_returns_fail_closed_reply_without_retry() -> None:
    async def _timeout(_argv, _timeout_seconds):
        raise TimeoutError

    decision = await _router(runner=_timeout).handle(_event("/site-audit sdtk_public_web"))

    assert decision.handled is True
    assert "temporarily unavailable" in (decision.response or "")


@pytest.mark.asyncio
async def test_exclusive_home_group_refuses_natural_language_without_cli_or_llm_fallback() -> None:
    command_runner = AsyncMock()
    router = ControlPlaneRouter(
        {
            "enabled": True,
            "owner_telegram_user_id": OWNER_ID,
            "home_telegram_chat_id": "-100123",
            "exclusive_control_plane_mode": True,
        },
        command_runner=command_runner,
    )

    decision = await router.handle(_event("kiem tra giup tao run nay co bi ket khong", chat_id="-100123"))

    assert decision.handled is True
    assert "Exact syntax" in (decision.response or "")
    command_runner.assert_not_awaited()


@pytest.mark.asyncio
async def test_exclusive_mode_leaves_owner_messages_outside_home_group_on_normal_gateway_path() -> None:
    command_runner = AsyncMock()
    router = ControlPlaneRouter(
        {
            "enabled": True,
            "owner_telegram_user_id": OWNER_ID,
            "home_telegram_chat_id": "-100123",
            "exclusive_control_plane_mode": True,
        },
        command_runner=command_runner,
    )

    decision = await router.handle(_event("Please summarize the current health status.", chat_id="10001"))

    assert decision.handled is False
    assert decision.response is None
    command_runner.assert_not_awaited()

@pytest.mark.asyncio
async def test_video_self_service_prepare_is_exact_and_owner_gated() -> None:
    command_runner = AsyncMock(return_value=SimpleNamespace(
        returncode=0,
        stdout='{"status":"prepared_waiting_for_exact_dispatch_approval","run_id":"run_abc123_def456","exact_kickoff_approval":"APPROVE VIDEO KICKOFF run_abc123_def456 ' + ('a' * 64) + '"}',
        stderr="",
    ))
    router = ControlPlaneRouter(
        {"enabled": True, "owner_telegram_user_id": OWNER_ID, "marketing_video_self_service_enabled": True},
        command_runner=command_runner,
    )
    decision = await router.handle(_event("/marketing-video prepare EP3"))
    assert decision.handled is True
    assert "APPROVE VIDEO KICKOFF run_abc123_def456" in (decision.response or "")
    assert command_runner.await_args.args[0] == ["node", router_module.VIDEO_SELF_SERVICE_BIN, "prepare", "EP3"]


@pytest.mark.asyncio
async def test_video_self_service_kickoff_requires_exact_hash_and_returns_bounded_result() -> None:
    digest = 'b' * 64
    command_runner = AsyncMock(return_value=SimpleNamespace(returncode=0, stdout='{"status":"dispatched"}', stderr=""))
    router = ControlPlaneRouter(
        {"enabled": True, "owner_telegram_user_id": OWNER_ID, "marketing_video_self_service_enabled": True},
        command_runner=command_runner,
    )
    decision = await router.handle(_event(f"APPROVE VIDEO KICKOFF run_abc123_def456 {digest}"))
    assert decision.handled is True
    assert "kickoff submitted" in (decision.response or "").lower()
    assert command_runner.await_args.args[0] == ["node", router_module.VIDEO_SELF_SERVICE_BIN, "kickoff", "run_abc123_def456", digest]


@pytest.mark.asyncio
async def test_video_self_service_invalid_extra_argument_is_refused_without_cli() -> None:
    command_runner = AsyncMock()
    router = ControlPlaneRouter(
        {"enabled": True, "owner_telegram_user_id": OWNER_ID, "marketing_video_self_service_enabled": True},
        command_runner=command_runner,
    )
    decision = await router.handle(_event("/marketing-video prepare EP3 extra"))
    assert "Exact syntax" in (decision.response or "")
    command_runner.assert_not_awaited()
