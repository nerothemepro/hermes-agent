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


def _event(text: str, user_id: str = OWNER_ID, chat_id: str = OWNER_ID) -> MessageEvent:
    return MessageEvent(
        text=text,
        message_id="hersocial-approval-test",
        source=SessionSource(
            platform=Platform.TELEGRAM,
            user_id=user_id,
            chat_id=chat_id,
            user_name="owner",
            chat_type="dm",
        ),
    )


def _router(runner, home_chat_id: str | None = None) -> ControlPlaneRouter:
    config = {"enabled": True, "owner_telegram_user_id": OWNER_ID, "command_timeout_seconds": 7}
    if home_chat_id is not None:
        config["home_telegram_chat_id"] = home_chat_id
    return ControlPlaneRouter(
        config,
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
async def test_exact_owner_approval_reports_confirmed_published_url() -> None:
    runner = AsyncMock(
        return_value=SimpleNamespace(
            returncode=0,
            stdout=(
                '{"status":"published",'
                f'"post_key":"{POST_KEY}","content_sha256":"{DIGEST}",'
                '"video_url":"https://www.facebook.com/reel/example"}'
            ),
            stderr="",
        )
    )

    decision = await _router(runner).handle(
        _event(f"APPROVE HERSOCIAL POST {POST_KEY} {DIGEST}")
    )

    assert decision.handled is True
    assert "published" in (decision.response or "")
    assert "https://www.facebook.com/reel/example" in (decision.response or "")


@pytest.mark.asyncio
async def test_published_response_without_https_permalink_fails_closed() -> None:
    runner = AsyncMock(
        return_value=SimpleNamespace(
            returncode=0,
            stdout=(
                '{"status":"published",'
                f'"post_key":"{POST_KEY}","content_sha256":"{DIGEST}",'
                '"video_url":"/reel/not-a-permalink"}'
            ),
            stderr="",
        )
    )

    decision = await _router(runner).handle(
        _event(f"APPROVE HERSOCIAL POST {POST_KEY} {DIGEST}")
    )

    assert "lacked a valid permalink" in (decision.response or "")


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


@pytest.mark.asyncio
async def test_home_group_exact_approval_records_bounded_command() -> None:
    group_id = "-100123"
    runner = AsyncMock(return_value=SimpleNamespace(
        returncode=0,
        stdout=(
            '{"status":"approved_pending_publish",'
            f'"post_key":"{POST_KEY}","content_sha256":"{DIGEST}"}}'
        ),
        stderr="",
    ))

    decision = await _router(runner, home_chat_id=group_id).handle(
        _event(f"APPROVE HERSOCIAL POST {POST_KEY} {DIGEST}", chat_id=group_id)
    )

    assert decision.handled is True
    assert "approval recorded" in (decision.response or "")
    assert runner.await_args.args[0] == [HERSOCIAL_APPROVAL_BIN, "--record-approval", POST_KEY, DIGEST]


@pytest.mark.asyncio
async def test_owner_control_command_outside_home_group_is_silently_dropped() -> None:
    runner = AsyncMock()
    decision = await _router(runner, home_chat_id="-100123").handle(
        _event(f"APPROVE HERSOCIAL POST {POST_KEY} {DIGEST}", chat_id="10001")
    )
    assert decision.handled is True
    assert decision.response is None
    runner.assert_not_awaited()


@pytest.mark.asyncio
async def test_normal_owner_message_outside_home_group_passthroughs() -> None:
    runner = AsyncMock()
    decision = await _router(runner, home_chat_id="-100123").handle(
        _event("Please draft a safe social post.", chat_id="10001")
    )
    assert decision.handled is False
    assert decision.response is None
    runner.assert_not_awaited()


def test_home_chat_env_supports_telegram_prefix(monkeypatch) -> None:
    monkeypatch.setenv("HERSOCIAL_HOME_GROUP", "telegram:-100123")
    router = ControlPlaneRouter.from_gateway_config({
        "control_plane_router": {
            "enabled": True,
            "owner_telegram_user_id": OWNER_ID,
            "home_telegram_chat_env": "HERSOCIAL_HOME_GROUP",
        }
    })
    assert router.home_chat_id == "-100123"
