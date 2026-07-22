from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform
from gateway.control_plane_router import ControlPlaneRouter, HERSOCIAL_APPROVAL_BIN
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


OWNER_ID = "10001"
POST_KEY = "sdtk-pro-live-post-5"
DIGEST = "a" * 64


def _event(text: str, user_id: str = OWNER_ID) -> MessageEvent:
    return MessageEvent(
        text=text,
        message_id="hersocial-approval-test",
        source=SessionSource(
            platform=Platform.TELEGRAM,
            user_id=user_id,
            chat_id=user_id,
            user_name="owner",
            chat_type="dm",
        ),
    )


def _router(runner) -> ControlPlaneRouter:
    return ControlPlaneRouter(
        {"enabled": True, "owner_telegram_user_id": OWNER_ID, "command_timeout_seconds": 7},
        command_runner=runner,
    )


@pytest.mark.asyncio
async def test_exact_owner_approval_records_bounded_command() -> None:
    runner = AsyncMock(
        return_value=SimpleNamespace(
            returncode=0,
            stdout=(
                '{"status":"approved_pending_publish",'
                f'"post_key":"{POST_KEY}","content_sha256":"{DIGEST}"}}'
            ),
            stderr="",
        )
    )

    decision = await _router(runner).handle(
        _event(f"APPROVE HERSOCIAL POST {POST_KEY} {DIGEST}")
    )

    assert decision.handled is True
    assert "approval recorded" in (decision.response or "")
    assert runner.await_args.args == (
        [HERSOCIAL_APPROVAL_BIN, "--record-approval", POST_KEY, DIGEST],
        7,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        f"approve hersocial post {POST_KEY} {DIGEST}",
        f"APPROVE HERSOCIAL POST {POST_KEY}",
        f"APPROVE HERSOCIAL POST {POST_KEY} {'a' * 63}",
        f"APPROVE HERSOCIAL POST post_key_with_underscore {DIGEST}",
    ],
)
async def test_partial_or_natural_language_approval_fails_closed(text: str) -> None:
    runner = AsyncMock()
    decision = await _router(runner).handle(_event(text))
    assert decision.handled is True
    assert "Exact syntax" in (decision.response or "")
    runner.assert_not_awaited()


@pytest.mark.asyncio
async def test_non_owner_is_silently_dropped_before_command() -> None:
    runner = AsyncMock()
    decision = await _router(runner).handle(
        _event(f"APPROVE HERSOCIAL POST {POST_KEY} {DIGEST}", user_id="20002")
    )
    assert decision.handled is True
    assert decision.response is None
    runner.assert_not_awaited()


@pytest.mark.asyncio
async def test_helper_failure_reports_fail_closed_without_retry() -> None:
    runner = AsyncMock(return_value=SimpleNamespace(returncode=1, stdout="{}", stderr="blocked"))
    decision = await _router(runner).handle(
        _event(f"APPROVE HERSOCIAL POST {POST_KEY} {DIGEST}")
    )
    assert "failed closed" in (decision.response or "")
    assert runner.await_count == 1
