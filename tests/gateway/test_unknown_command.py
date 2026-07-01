"""Tests for gateway warning when an unrecognized /command is dispatched.

Without this warning, unknown slash commands get forwarded to the LLM as plain
text, which often leads to silent failure (e.g. the model inventing a bogus
delegate_task call instead of telling the user the command doesn't exist).
"""

import json
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource, build_session_key


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


def _make_event(text: str) -> MessageEvent:
    return MessageEvent(text=text, source=_make_source(), message_id="m1")


def _make_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    adapter = MagicMock()
    adapter.send = AsyncMock()
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(
        emit=AsyncMock(),
        emit_collect=AsyncMock(return_value=[]),
        loaded_hooks=False,
    )

    session_entry = SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = session_entry
    runner.session_store.load_transcript.return_value = []
    runner.session_store.has_any_sessions.return_value = True
    runner.session_store.append_to_transcript = MagicMock()
    runner.session_store.rewrite_transcript = MagicMock()
    runner.session_store.update_session = MagicMock()
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._session_db = None
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._show_reasoning = False
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._should_send_voice_reply = lambda *_args, **_kwargs: False
    runner._send_voice_reply = AsyncMock()
    runner._capture_gateway_honcho_if_configured = lambda *args, **kwargs: None
    runner._emit_gateway_run_progress = AsyncMock()
    return runner


@pytest.mark.asyncio
async def test_unknown_slash_command_returns_guidance(monkeypatch):
    """A genuinely unknown /foobar should return user-facing guidance, not
    silently drop through to the LLM."""
    import gateway.run as gateway_run

    runner = _make_runner()
    # If the LLM were called, this would fail: the guard must short-circuit
    # before _run_agent is invoked.
    runner._run_agent = AsyncMock(
        side_effect=AssertionError(
            "unknown slash command leaked through to the agent"
        )
    )

    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )

    result = await runner._handle_message(_make_event("/definitely-not-a-command"))

    assert result is not None
    assert "Unknown command" in result
    assert "/definitely-not-a-command" in result
    assert "/commands" in result
    runner._run_agent.assert_not_called()


@pytest.mark.asyncio
async def test_unknown_slash_command_underscored_form_also_guarded(monkeypatch):
    """Telegram may send /foo_bar — same guard must trigger for underscored
    commands that normalize to unknown hyphenated names."""
    import gateway.run as gateway_run

    runner = _make_runner()
    runner._run_agent = AsyncMock(
        side_effect=AssertionError(
            "unknown slash command leaked through to the agent"
        )
    )

    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )

    result = await runner._handle_message(_make_event("/made_up_thing"))

    assert result is not None
    assert "Unknown command" in result
    assert "/made_up_thing" in result
    runner._run_agent.assert_not_called()


@pytest.mark.asyncio
async def test_known_slash_command_not_flagged_as_unknown(monkeypatch):
    """A real built-in like /status must NOT hit the unknown-command guard."""
    runner = _make_runner()
    # Make _handle_status_command exist via the normal path by running a real
    # dispatch. If the guard fires, the return string will mention "Unknown".
    runner._running_agents[build_session_key(_make_source())] = MagicMock()

    result = await runner._handle_message(_make_event("/status"))

    assert result is not None
    assert "Unknown command" not in result


@pytest.mark.asyncio
async def test_social_shortcut_rewrites_into_background_prompt(monkeypatch):
    runner = _make_runner()
    event = _make_event(
        "/social-video-preview topic: Camping weekend memory summary: Create a short Facebook post from a bounded research bundle. media_path: /opt/data/hermes/generated-videos/demo.mp4"
    )

    seen = {}

    async def _capture_background_command(captured_event):
        seen["text"] = captured_event.text
        return "bg started"

    runner._handle_background_command = _capture_background_command

    original_text = event.text
    result = await runner._handle_social_shortcut_command(event)

    assert result == "bg started"
    assert event.text == original_text
    assert seen["text"].startswith("/background /social-video-preview ")
    assert "topic: Camping weekend memory" in seen["text"]
    assert "summary: Create a short Facebook post" in seen["text"]



@pytest.mark.asyncio
async def test_herorches_shortcut_rewrites_into_background_prompt_without_args(monkeypatch):
    runner = _make_runner()
    event = _make_event("/health-all")

    seen = {}

    async def _capture_background_command(captured_event):
        seen["text"] = captured_event.text
        return "bg started"

    runner._handle_background_command = _capture_background_command

    original_text = event.text
    result = await runner._handle_herorches_shortcut_command(event)

    assert result == "bg started"
    assert event.text == original_text
    assert seen["text"] == "/background /health-all"


@pytest.mark.asyncio
async def test_herresearch_shortcut_rewrites_into_background_prompt_without_args(monkeypatch):
    runner = _make_runner()
    event = _make_event("/github-discovery")

    seen = {}

    async def _capture_background_command(captured_event):
        seen["text"] = captured_event.text
        return "bg started"

    runner._handle_background_command = _capture_background_command

    original_text = event.text
    result = await runner._handle_herresearch_shortcut_command(event)

    assert result == "bg started"
    assert event.text == original_text
    assert seen["text"] == "/background /github-discovery"


@pytest.mark.asyncio
async def test_herwiki_shortcut_executes_deterministic_helper(monkeypatch):
    runner = _make_runner()
    runner.config.quick_commands = {"wiki-lint": {"enabled": True}}
    event = _make_event("/wiki-lint")

    seen = {}

    async def _capture(argv, *, timeout):
        seen["argv"] = argv
        seen["timeout"] = timeout
        return "lint ok"

    runner._run_deterministic_shortcut_command = _capture

    result = await runner._handle_herwiki_shortcut_command(event)

    assert result == "lint ok"
    assert seen["timeout"] == 120
    assert seen["argv"] == [
        "node",
        "/workspace/hermes-agent-plugin/bin/herwiki-sdtk-wiki-tool",
        "lint",
    ]


@pytest.mark.asyncio
async def test_herwiki_search_shortcut_requires_and_forwards_query(monkeypatch):
    runner = _make_runner()
    runner.config.quick_commands = {"wiki-search": {"enabled": True}}
    event = _make_event("/wiki-search graph runtime")

    seen = {}

    async def _capture(argv, *, timeout):
        seen["argv"] = argv
        return "search ok"

    runner._run_deterministic_shortcut_command = _capture

    result = await runner._handle_herwiki_shortcut_command(event)

    assert result == "search ok"
    assert seen["argv"] == [
        "node",
        "/workspace/hermes-agent-plugin/bin/herwiki-sdtk-wiki-tool",
        "search",
        "--query",
        "graph runtime",
    ]

    usage_result = await runner._handle_herwiki_shortcut_command(_make_event("/wiki-search"))
    assert usage_result == "Usage: /wiki-search <query>"


@pytest.mark.asyncio
async def test_herwiki_lint_shortcut_formats_json_summary(monkeypatch):
    runner = _make_runner()
    runner.config.quick_commands = {"wiki-lint": {"enabled": True}}
    event = _make_event("/wiki-lint")

    async def _capture(argv, *, timeout):
        return json.dumps({
            "status": "completed",
            "action": "lint",
            "report_paths": ["/tmp/wiki/.sdtk/wiki/reports/lint-report.md"],
            "warnings": [],
            "errors": [],
            "stdout": "[wiki] Lint report: /tmp/wiki/.sdtk/wiki/reports/lint-report.md\n[wiki] Findings: 7",
        })

    runner._run_deterministic_shortcut_command = _capture

    result = await runner._handle_herwiki_shortcut_command(event)

    assert '"status"' not in result
    assert "Status: completed" in result
    assert "Action: lint" in result
    assert "- Findings: 7" in result
    assert "Reports:" in result
    assert "/tmp/wiki/.sdtk/wiki/reports/lint-report.md" in result


@pytest.mark.asyncio
async def test_herwiki_search_shortcut_formats_compact_summary(monkeypatch):
    runner = _make_runner()
    runner.config.quick_commands = {"wiki-search": {"enabled": True}}
    event = _make_event("/wiki-search graph runtime")

    async def _capture(argv, *, timeout):
        return json.dumps({
            "status": "completed",
            "action": "search",
            "query": "graph runtime",
            "result_count": 2,
            "total_matches": 12,
            "search_meta": {"scanned_files": 381},
            "warnings": [],
            "errors": [],
            "search_results": [
                {"title": "Runtime A", "path": "wiki/a.md", "score": 99},
                {"title": "Runtime B", "path": "wiki/b.md", "score": 88},
            ],
        })

    runner._run_deterministic_shortcut_command = _capture

    result = await runner._handle_herwiki_shortcut_command(event)

    assert '"search_results"' not in result
    assert "Status: completed" in result
    assert "Results: 2 shown / 12 total" in result
    assert "Scanned files: 381" in result
    assert "1. Runtime A (score 99)" in result
    assert "   wiki/a.md" in result


@pytest.mark.asyncio
async def test_underscored_alias_for_hyphenated_builtin_not_flagged(monkeypatch):
    """Telegram autocomplete sends /reload_mcp for the /reload-mcp built-in.
    That must NOT be flagged as unknown."""
    import gateway.run as gateway_run

    runner = _make_runner()
    # Prevent real MCP work; we only care that the unknown guard doesn't fire.
    async def _noop_reload(*_a, **_kw):
        return "mcp reloaded"

    runner._handle_reload_mcp_command = _noop_reload  # type: ignore[attr-defined]

    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )

    result = await runner._handle_message(_make_event("/reload_mcp"))

    # Whatever /reload_mcp returns, it must not be the unknown-command guard.
    if result is not None:
        assert "Unknown command" not in result


# ------------------------------------------------------------------
# command:<name> decision hook — deny / handled / rewrite
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_command_hook_can_deny_before_dispatch(monkeypatch):
    """A handler returning {"decision": "deny"} blocks a slash command early."""
    import gateway.run as gateway_run

    runner = _make_runner()
    runner._run_agent = AsyncMock(
        side_effect=AssertionError("denied slash command leaked to the agent")
    )
    runner._handle_status_command = AsyncMock(
        side_effect=AssertionError("denied slash command reached its handler")
    )
    runner.hooks.emit_collect = AsyncMock(
        return_value=[{"decision": "deny", "message": "Blocked by ACL"}]
    )

    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )

    result = await runner._handle_message(_make_event("/status"))

    assert result == "Blocked by ACL"
    runner._run_agent.assert_not_called()
    # The emit_collect call should use the canonical command name.
    call_args = runner.hooks.emit_collect.await_args
    assert call_args.args[0] == "command:status"


@pytest.mark.asyncio
async def test_command_hook_deny_without_message_uses_default(monkeypatch):
    """A deny decision with no message falls back to a generic blocked string."""
    import gateway.run as gateway_run

    runner = _make_runner()
    runner._handle_status_command = AsyncMock(
        side_effect=AssertionError("denied slash command reached its handler")
    )
    runner.hooks.emit_collect = AsyncMock(return_value=[{"decision": "deny"}])

    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )

    result = await runner._handle_message(_make_event("/status"))

    assert result is not None
    assert "blocked" in result.lower()


@pytest.mark.asyncio
async def test_command_hook_can_mark_command_as_handled(monkeypatch):
    """A handled decision short-circuits dispatch cleanly with a custom reply."""
    import gateway.run as gateway_run

    runner = _make_runner()
    runner._handle_status_command = AsyncMock(
        side_effect=AssertionError("handled slash command reached its handler")
    )
    runner.hooks.emit_collect = AsyncMock(
        return_value=[{"decision": "handled", "message": "Already handled upstream"}]
    )

    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )

    result = await runner._handle_message(_make_event("/status"))

    assert result == "Already handled upstream"


@pytest.mark.asyncio
async def test_command_hook_allow_decision_is_passthrough(monkeypatch):
    """A handler returning {"decision": "allow"} must NOT prevent normal dispatch."""
    import gateway.run as gateway_run

    runner = _make_runner()
    runner._handle_status_command = AsyncMock(return_value="status: ok")
    runner.hooks.emit_collect = AsyncMock(
        return_value=[{"decision": "allow"}]
    )

    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )

    result = await runner._handle_message(_make_event("/status"))

    assert result == "status: ok"
    runner._handle_status_command.assert_awaited_once()


@pytest.mark.asyncio
async def test_command_hook_non_dict_return_values_ignored(monkeypatch):
    """Hook return values that aren't dicts must not break dispatch."""
    import gateway.run as gateway_run

    runner = _make_runner()
    runner._handle_status_command = AsyncMock(return_value="status: ok")
    runner.hooks.emit_collect = AsyncMock(
        return_value=["some string", 42, None, {}]
    )

    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )

    result = await runner._handle_message(_make_event("/status"))

    assert result == "status: ok"


@pytest.mark.asyncio
async def test_command_hook_fires_for_plugin_registered_command(monkeypatch):
    """Plugin-registered slash commands should also trigger command:<name> hooks."""
    import gateway.run as gateway_run

    runner = _make_runner()
    runner._run_agent = AsyncMock(
        side_effect=AssertionError("plugin command leaked to the agent")
    )
    runner.hooks.emit_collect = AsyncMock(
        return_value=[{"decision": "handled", "message": "intercepted"}]
    )

    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )
    # Stub plugin command lookup so is_gateway_known_command() recognizes /metricas.
    from hermes_cli import plugins as _plugins_mod

    monkeypatch.setattr(
        _plugins_mod,
        "get_plugin_commands",
        lambda: {"metricas": {"description": "Metrics", "args_hint": "dias:7"}},
    )

    result = await runner._handle_message(_make_event("/metricas dias:7"))

    assert result == "intercepted"
    # Hook event name uses the plugin command as canonical.
    call_args = runner.hooks.emit_collect.await_args
    assert call_args.args[0] == "command:metricas"
    # Args are passed through in both "args" and "raw_args" keys.
    ctx = call_args.args[1]
    assert ctx["raw_args"] == "dias:7"


@pytest.mark.asyncio
async def test_command_hook_rewrite_routes_to_plugin(monkeypatch):
    """A rewrite decision should re-resolve the command and route to the new one."""
    import gateway.run as gateway_run

    runner = _make_runner()
    runner._run_agent = AsyncMock(
        side_effect=AssertionError("rewritten command leaked to the agent")
    )

    call_log = []

    async def _emit_collect(event_type, ctx):
        call_log.append(event_type)
        if event_type == "command:status":
            return [
                {
                    "decision": "rewrite",
                    "command_name": "metricas",
                    "raw_args": "dias:7",
                }
            ]
        return []

    runner.hooks.emit_collect = AsyncMock(side_effect=_emit_collect)

    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )
    from hermes_cli import plugins as _plugins_mod

    monkeypatch.setattr(
        _plugins_mod,
        "get_plugin_commands",
        lambda: {"metricas": {"description": "Metrics", "args_hint": "dias:7"}},
    )
    monkeypatch.setattr(
        _plugins_mod,
        "get_plugin_command_handler",
        lambda name: (lambda args: f"metrics {args}") if name == "metricas" else None,
    )

    result = await runner._handle_message(_make_event("/status"))

    assert result == "metrics dias:7"
    # First emit_collect fires on the original command; after rewrite the
    # dispatcher does NOT re-fire for the new command (one decision per turn).
    assert call_log == ["command:status"]
