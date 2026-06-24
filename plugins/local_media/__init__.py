"""Local media generation plugin for Hermes."""

from __future__ import annotations

from plugins.local_media.tools import (
    GENERATE_HERVID_PREVIEW_SCHEMA,
    GENERATE_HERVID_VIDEO_SCHEMA,
    GENERATE_LTX_VIDEO_SCHEMA,
    GENERATE_LTX_VIDEO_SEQUENCE_SCHEMA,
    GENERATE_VIDEO_SCHEMA,
    GENERATE_VIDEO_SEQUENCE_SCHEMA,
    handle_generate_hervid_preview,
    handle_generate_hervid_video,
    handle_generate_ltx_video,
    handle_generate_ltx_video_sequence,
    handle_generate_video,
    handle_generate_video_sequence,
)


def register(ctx) -> None:
    ctx.register_tool(
        name="generate_hervid_preview",
        toolset="local_media",
        schema=GENERATE_HERVID_PREVIEW_SCHEMA,
        handler=handle_generate_hervid_preview,
        emoji="image",
    )
    ctx.register_tool(
        name="generate_hervid_video",
        toolset="local_media",
        schema=GENERATE_HERVID_VIDEO_SCHEMA,
        handler=handle_generate_hervid_video,
        emoji="video",
    )
    ctx.register_tool(
        name="generate_video",
        toolset="local_media",
        schema=GENERATE_VIDEO_SCHEMA,
        handler=handle_generate_video,
        emoji="video",
    )
    ctx.register_tool(
        name="generate_ltx_video",
        toolset="local_media",
        schema=GENERATE_LTX_VIDEO_SCHEMA,
        handler=handle_generate_ltx_video,
        emoji="video",
    )
    ctx.register_tool(
        name="generate_ltx_video_sequence",
        toolset="local_media",
        schema=GENERATE_LTX_VIDEO_SEQUENCE_SCHEMA,
        handler=handle_generate_ltx_video_sequence,
        emoji="video",
    )
    ctx.register_tool(
        name="generate_video_sequence",
        toolset="local_media",
        schema=GENERATE_VIDEO_SEQUENCE_SCHEMA,
        handler=handle_generate_video_sequence,
        emoji="video",
    )
