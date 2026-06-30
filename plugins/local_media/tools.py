"""Agent-facing local media tools.

The tool intentionally wraps the existing media-pipeline CLI instead of exposing
terminal access to the model. It verifies that the pipeline returns a real file
inside /opt/data/hermes/generated-videos before handing the path back to Hermes.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any

from tools.registry import tool_error, tool_result

PIPELINE_SCRIPT = Path("/workspace/projects/media-pipeline/generate_video.py")
SEQUENCE_SCRIPT = Path("/workspace/projects/media-pipeline/generate_video_sequence.py")
LTX_PIPELINE_SCRIPT = Path("/workspace/projects/media-pipeline/generate_ltx_video.py")
LTX_SEQUENCE_SCRIPT = Path("/workspace/projects/media-pipeline/generate_ltx_video_sequence.py")
DEFAULT_ENV_FILE = Path("/opt/data/hermes/media-pipeline.env")
DEFAULT_VIDEO_DIR = Path("/opt/data/hermes/generated-videos")
SMOKE_IMAGE = Path("/workspace/projects/media-pipeline/test_assets/wan_i2v_smoke_input.png")
ALLOWED_VIDEO_DIRS = (DEFAULT_VIDEO_DIR.resolve(),)
ORCHESTRATOR_URL = os.environ.get("HVP_ORCHESTRATOR_URL", "http://localhost:8501")


GENERATE_HERVID_PREVIEW_SCHEMA: dict[str, Any] = {
    "name": "generate_hervid_preview",
    "description": (
        "ALWAYS call this tool FIRST when a user asks for a video. "
        "Generates a keyframe (composition image) from the user's brief, sends it to the user for approval, "
        "and returns a preview_id. If the user approves the composition, pass preview_id to "
        "generate_hervid_video to render the full video using the exact same creative parameters. "
        "If the user wants a different composition, call generate_hervid_preview again. "
        "Keyframe generation takes ~1-2 minutes (much faster than a full render)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "brief": {
                "type": "string",
                "description": (
                    "The user's video request in any language. "
                    "Example: 'video chú cáo chạy trong rừng mùa thu, toàn thân'."
                ),
            },
            "mode": {
                "type": "string",
                "enum": ["test", "standard", "quality"],
                "default": "quality",
                "description": "Render quality for the eventual video. test=fast/low-res preview; quality=full-res.",
            },
            "timeout_seconds": {
                "type": "integer",
                "default": 900,
                "minimum": 120,
                "maximum": 1800,
                "description": "Max wait time for keyframe generation. Use a higher value when ComfyUI is under load.",
            },
        },
        "required": ["brief"],
        "additionalProperties": False,
    },
}


GENERATE_HERVID_VIDEO_SCHEMA: dict[str, Any] = {
    "name": "generate_hervid_video",
    "description": (
        "PRIMARY Telegram-safe HerVid workflow. Render a full multi-shot HerVid video (8-60 min on RTX 3090). "
        "MANDATORY workflow: you MUST call generate_hervid_preview first, show the keyframe image to "
        "the user, wait for their approval, THEN call this tool with preview_id. "
        "DO NOT call this tool with only a brief and no preview_id — skipping the preview wastes a "
        "full render if composition is wrong (missing legs, wrong framing, wrong character). "
        "Exception: user explicitly says 'skip preview' or 'generate directly'. "
        "For ordinary HerVid Telegram requests, prefer this tool over generate_ltx_video."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "brief": {
                "type": "string",
                "description": (
                    "The user's video request in any language. Required if preview_id is not provided."
                ),
            },
            "preview_id": {
                "type": "string",
                "description": (
                    "ID returned by generate_hervid_preview. When provided, reuses the approved creative "
                    "parameters and skips the LM parsing step. Pass this after the user approves the keyframe."
                ),
            },
            "mode": {
                "type": "string",
                "enum": ["test", "standard", "quality"],
                "default": "quality",
                "description": (
                    "Render quality: test=fast (2s/shot, ~8 min); "
                    "standard=balanced; quality=best (recommended). "
                    "Always respected — even when preview_id is provided, this mode overrides the preview's stored mode."
                ),
            },
            "timeout_seconds": {
                "type": "integer",
                "default": 7200,
                "minimum": 300,
                "maximum": 14400,
                "description": "Max wait time. A 30s quality video takes 30-60 min on RTX 3090.",
            },
        },
        "required": [],
        "additionalProperties": False,
    },
}


GENERATE_LTX_VIDEO_SCHEMA: dict[str, Any] = {
    "name": "generate_ltx_video",
    "description": (
        "DIRECT operator path for a short realistic/cinematic/product/travel/social video through local ComfyUI LTX-2.3 I2V. "
        "This is NOT the normal HerVid Telegram workflow. For ordinary user-facing HerVid requests, prefer "
        "generate_hervid_video so the preview/approval flow runs first. "
        "Use this when HerVid needs LTX-2.3 instead of Wan2.1, especially for realistic marketing, product, "
        "travel, lifestyle, and social clips. Prefer mode=test for smoke checks; standard is the safe default "
        "for real use on RTX 3090. The tool saves the final mp4 under /opt/data/hermes/generated-videos and "
        "returns a MEDIA:/absolute/path directive suitable for Telegram delivery."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "Detailed English video prompt describing subject, motion over time, camera movement, lighting, and constraints such as no text/no watermark.",
            },
            "input_image_path": {
                "type": "string",
                "description": "Optional first-frame image path. If omitted, the pipeline generates a keyframe using the existing Hermes image/keyframe pipeline.",
            },
            "mode": {
                "type": "string",
                "enum": ["test", "standard", "quality"],
                "default": "test",
                "description": "test=1s 512x320 8fps; standard=3s 512x320 8fps; quality=5s 768x512 8fps and may require freeing VRAM first.",
            },
            "style": {
                "type": "string",
                "enum": ["realistic", "product", "travel", "social_ad", "anime"],
                "default": "realistic",
                "description": "Keyframe style hint when input_image_path is omitted. LTX is recommended for realistic/product/travel/social_ad; use Wan for anime action.",
            },
            "keyframe_engine": {
                "type": "string",
                "enum": ["auto", "flux", "animagine"],
                "default": "auto",
                "description": "Keyframe renderer used only when input_image_path is omitted.",
            },
            "width": {"type": "integer", "minimum": 256, "maximum": 1280, "description": "Optional width override. Keep 512 for safest RTX 3090 LTX smoke runs."},
            "height": {"type": "integer", "minimum": 256, "maximum": 720, "description": "Optional height override. Keep 320 for safest RTX 3090 LTX smoke runs."},
            "duration": {"type": "integer", "minimum": 1, "maximum": 5, "description": "Single-shot duration in seconds. Values above 5 are intentionally rejected for Phase 1."},
            "fps": {"type": "integer", "minimum": 8, "maximum": 24, "description": "Frames per second. 8 is the safest smoke default; 16 may need more VRAM."},
            "steps": {"type": "integer", "minimum": 1, "maximum": 30, "description": "Sampler steps. Lower values are faster and safer for smoke tests."},
            "seed": {"type": "integer", "minimum": 0, "maximum": 2147483647, "description": "Optional deterministic seed."},
            "timeout_seconds": {"type": "integer", "default": 1800, "minimum": 120, "maximum": 7200, "description": "Maximum wall-clock time for the LTX render."},
        },
        "required": ["prompt"],
        "additionalProperties": False,
    },
}


GENERATE_LTX_VIDEO_SEQUENCE_SCHEMA: dict[str, Any] = {
    "name": "generate_ltx_video_sequence",
    "description": (
        "Generate a LONG multi-shot LTX-2.3 video (roughly 10-75 seconds) by rendering several short shots "
        "and stitching them into one mp4. This is the DEFAULT tool for ANY multi-shot or long video in EVERY "
        "style — action sequences, cinematic scenes, realistic footage, AND animation / cartoon / 3D / Pixar "
        "/ anime / children's videos — i.e. whenever the user wants a video longer than a single ~5s LTX clip. "
        "ALWAYS use this tool for animation, cartoon and children's content; do NOT route those to the Wan2.1 "
        "generate_video_sequence tool. Only skip this tool if the user explicitly names the Wan2.1 pipeline. "
        "Each shot is rendered with the same engine as generate_ltx_video; by default the last frame "
        "of each shot becomes the first frame of the next shot so the action stays continuous. "
        "For the best long-form result, pass an explicit `shots` array where each item is the prompt for one "
        "consecutive ~5s beat of the action (e.g. 12 shots for a ~60s fight). If `shots` is omitted, the tool "
        "auto-splits `prompt` into shots based on total_duration_seconds. Rendering many shots is slow "
        "(each shot is a full LTX render on the RTX 3090), so this can take several minutes. The tool returns "
        "a MEDIA:/absolute/path to the final stitched video for Telegram delivery."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "Overall scene/action description. Used for the output name and as the fallback when `shots` is not provided.",
            },
            "shots": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Preferred. Ordered list of per-shot English prompts, one per ~5s beat. Provide 12-15 shots for a 60-75s video. Each shot should describe a distinct moment of motion that continues from the previous one.",
            },
            "total_duration_seconds": {
                "type": "integer",
                "minimum": 6,
                "maximum": 90,
                "default": 60,
                "description": "Target total length when `shots` is omitted. The tool computes shot count = ceil(total / shot_duration_seconds).",
            },
            "shot_duration_seconds": {
                "type": "integer",
                "minimum": 1,
                "maximum": 5,
                "default": 3,
                "description": "Length of each individual shot in seconds (capped at 5 by the single-shot LTX engine). Default 3: shorter shots keep faces sharp because each shot is anchored to one keyframe and the face softens/morphs late in a long take. Only raise toward 5 for slow, near-static shots.",
            },
            "continuity": {
                "type": "string",
                "enum": ["last_frame", "independent"],
                "default": "independent",
                "description": "independent (recommended for long videos) generates a fresh keyframe per shot so quality and style do NOT drift over many shots; pair it with character_note to keep the same subjects. last_frame chains each shot from the previous shot's final frame (smoother continuity but quality drifts badly past ~3-4 shots).",
            },
            "character_note": {
                "type": "string",
                "description": "Shared character/style anchor appended to EVERY shot prompt so independent shots keep the same subjects. Strongly recommended for independent mode. Example: 'the same two original anime samurai, one in a black robe, one in a red robe, consistent faces and weapons, medium shot with faces clearly visible, cinematic anime style, no text, no watermark'.",
            },
            "keyframe_seed": {
                "type": "integer",
                "minimum": 0,
                "maximum": 2147483647,
                "description": "Optional fixed seed for per-shot keyframe generation, for extra cross-shot character consistency.",
            },
            "input_image_path": {
                "type": "string",
                "description": "Optional first-frame image for shot 1. If omitted, a keyframe is generated automatically.",
            },
            "mode": {
                "type": "string",
                "enum": ["test", "standard", "quality"],
                "default": "standard",
                "description": "Per-shot render preset. standard (3s 512x320 8fps) is the safe default; test for fast plumbing checks; quality (5s 768x512 8fps) is heavier and slower per shot.",
            },
            "style": {
                "type": "string",
                "enum": ["realistic", "product", "travel", "social_ad"],
                "default": "realistic",
                "description": "Keyframe style for automatic keyframe generation. Use 'realistic' for ALL animation, cartoon, Pixar, children's and live-action content — it uses the Flux keyframe engine which handles every style correctly. 'anime' is NOT available here; the LTX pipeline always uses Flux for keyframes regardless of content type.",
            },
            "keyframe_engine": {
                "type": "string",
                "enum": ["auto", "flux"],
                "default": "auto",
                "description": "Keyframe renderer. Always 'auto' (resolves to Flux) for LTX sequences. 'animagine' is a legacy Wan-only path and must NOT be used here.",
            },
            "width": {"type": "integer", "minimum": 256, "maximum": 1280, "description": "Optional width override applied to every shot. Keep 512 for safe RTX 3090 runs."},
            "height": {"type": "integer", "minimum": 256, "maximum": 720, "description": "Optional height override applied to every shot. Keep 320 for safe RTX 3090 runs."},
            "fps": {"type": "integer", "minimum": 8, "maximum": 24, "description": "Optional fps override applied to every shot. All shots must share fps for clean stitching."},
            "steps": {"type": "integer", "minimum": 1, "maximum": 30, "description": "Optional sampler steps override applied to every shot."},
            "seed": {"type": "integer", "minimum": 0, "maximum": 2147483647, "description": "Optional base seed; shot N uses seed+N."},
            "timeout_seconds": {"type": "integer", "default": 7200, "minimum": 300, "maximum": 14400, "description": "Overall wall-clock budget for the whole sequence across all shots."},
        },
        "required": ["prompt"],
        "additionalProperties": False,
    },
}

GENERATE_VIDEO_SEQUENCE_SCHEMA: dict[str, Any] = {
    "name": "generate_video_sequence",
    "description": (
        "SPECIALIZED LEGACY tool for ORIGINAL JAPANESE SHONEN ANIME SWORD-FIGHT (samurai/katana) scenes ONLY, "
        "via the user's ComfyUI + Wan2.1 FLF2V pipeline. ALWAYS use generate_ltx_video_sequence instead for "
        "EVERY general request — including ALL animation, cartoon, 3D / Pixar, anime, children's, cinematic and "
        "realistic videos. Use THIS Wan tool ONLY when the user EXPLICITLY names the Wan2.1 pipeline OR "
        "explicitly asks for a samurai / katana sword-fight anime_action storyboard. WARNING: this tool OVERRIDES "
        "the prompt's subject with a hardcoded two-samurai duel storyboard, so it must NEVER be used for "
        "non-samurai content (e.g. a fox, a child, a product) — that content belongs in generate_ltx_video_sequence. "
        "Default duration is 20 seconds, default style_preset is anime_action, default control_mode is FLF2V "
        "start/end keyframes, default postprocess is ffmpeg_fps at 16fps. Returns a MEDIA:/absolute/path directive."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": (
                    "User's video idea. Short ideas are allowed; the sequence pipeline expands them "
                    "into original anime action shot prompts internally."
                ),
            },
            "duration_seconds": {
                "type": "integer",
                "default": 20,
                "minimum": 8,
                "maximum": 30,
                "description": "Requested final duration. Defaults to 20 seconds; values are clamped to 8-30.",
            },
            "mode": {
                "type": "string",
                "enum": ["test", "quality"],
                "default": "quality",
                "description": "quality for real videos; test for short plumbing checks.",
            },
            "shot_count": {
                "type": "integer",
                "default": 0,
                "minimum": 0,
                "maximum": 15,
                "description": "Optional explicit shot count. Use 0 or omit for anime_action auto timing, e.g. about 4 shots for 8s and 6 shots for 12s.",
            },
            "shot_duration_seconds": {
                "type": "number",
                "default": 0,
                "minimum": 0,
                "maximum": 4,
                "description": "Optional target source duration per shot. Omit for safe defaults; anime action normally uses short 1.5-2.5s shots.",
            },
            "frames_per_shot": {
                "type": "integer",
                "default": 0,
                "minimum": 0,
                "maximum": 33,
                "description": "Optional Wan frame count per shot. Omit for safe defaults; anime_action uses shorter shots than the old 33-frame baseline.",
            },
            "wan_steps_per_shot": {
                "type": "integer",
                "default": 0,
                "minimum": 0,
                "maximum": 30,
                "description": "Optional Wan sampler steps per shot. Omit for safe defaults; test mode stays low for plumbing checks.",
            },
            "motion_profile": {
                "type": "string",
                "enum": ["rapid", "balanced", "dramatic", "impact"],
                "default": "balanced",
                "description": "Anime action shot timing preset. balanced is the safe default; rapid cuts faster, dramatic holds longer, impact favors stronger action beats.",
            },
            "storyboard_mode": {
                "type": "string",
                "enum": ["auto", "intro_action", "action_core", "full_arc"],
                "default": "auto",
                "description": "Anime action storyboard selection. auto detects fight keywords and uses action_core for short sword-fight clips so blade clash and counter beats are included.",
            },
            "keyframe_quality_preset": {
                "type": "string",
                "enum": ["flux_default", "anime_action_v2"],
                "default": "flux_default",
                "description": "Optional Phase 3 keyframe prompt preset. anime_action_v2 adds shot-specific composition, pose readability, and character consistency language.",
            },
            "keyframe_engine": {
                "type": "string",
                "enum": ["auto", "flux", "animagine"],
                "default": "auto",
                "description": "Keyframe renderer. auto uses Animagine for anime_action only when the checkpoint is available; animagine requires animagine-xl-3.1.safetensors.",
            },
            "keyframe_frame_mode": {
                "type": "string",
                "enum": ["single_scene", "stylized_panel"],
                "description": "Animagine keyframe composition mode. single_scene forces one full-frame anime film still with no panels or collage; stylized_panel preserves the older stylized behavior. Defaults to single_scene for anime_action with explicit animagine.",
            },
            "keyframe_only_sequence": {
                "type": "boolean",
                "default": False,
                "description": "Generate start/end keyframes for every storyboard shot and stop before Wan video rendering.",
            },
            "existing_keyframe_dir": {
                "type": "string",
                "description": "Optional approved keyframe directory containing shot_XX_start.png and shot_XX_end.png. When set, skip keyframe generation/validation and render Wan from these existing images.",
            },
            "shot_prompt_strength": {
                "type": "string",
                "enum": ["light", "balanced", "strong"],
                "default": "balanced",
                "description": "Controls how strongly anime_action_v2 emphasizes readable poses, clear silhouettes, and restrained effects.",
            },
            "composition_profile": {
                "type": "string",
                "enum": ["auto", "establishing", "closeup", "action", "impact"],
                "default": "auto",
                "description": "Optional keyframe composition hint. auto selects a profile per storyboard beat.",
            },
            "character_consistency_note": {
                "type": "string",
                "default": "",
                "description": "Optional original character continuity note for repeated clothing, colors, weapons, and faces across shots.",
            },
            "style": {
                "type": "string",
                "default": "original_japanese_anime_action",
                "description": "Backward-compatible style label. Prefer style_preset for new calls.",
            },
            "style_preset": {
                "type": "string",
                "enum": ["default", "anime_action"],
                "default": "anime_action",
                "description": "anime_action removes photoreal language and uses original Japanese 2D cel-shaded action prompts.",
            },
            "control_mode": {
                "type": "string",
                "enum": ["i2v_last_frame", "flf2v"],
                "default": "flf2v",
                "description": "flf2v generates start/end keyframes per shot; i2v_last_frame keeps the older last-frame chaining path.",
            },
            "postprocess": {
                "type": "string",
                "enum": ["none", "ffmpeg_fps", "frame_interpolate"],
                "default": "ffmpeg_fps",
                "description": "Final FPS stage. frame_interpolate falls back with a warning if the interpolation model is missing.",
            },
            "target_fps": {
                "type": "integer",
                "default": 16,
                "minimum": 8,
                "maximum": 24,
                "description": "Final FPS target for post-processing. Defaults to 16.",
            },
            "continuity": {
                "type": "string",
                "enum": ["last_frame", "independent"],
                "default": "last_frame",
                "description": "last_frame chains each shot into the next; independent renders standalone shots then stitches them.",
            },
            "seed": {
                "type": "integer",
                "minimum": 0,
                "maximum": 2147483647,
                "description": "Optional deterministic sequence seed. Use this to reproduce a previously approved keyframe contact sheet.",
            },
            "timeout_seconds": {
                "type": "integer",
                "default": 14400,
                "minimum": 300,
                "maximum": 28800,
                "description": "Maximum wall-clock time for the full sequence.",
            },
        },
        "required": ["prompt"],
        "additionalProperties": False,
    },
}


GENERATE_VIDEO_SCHEMA: dict[str, Any] = {
    "name": "generate_video",
    "description": (
        "Generate a local video through the user's ComfyUI + Wan2.1 media pipeline. "
        "Use this when the user asks to create/generate/render a video locally. "
        "For normal user requests, expand short ideas into a detailed English visual prompt "
        "before calling this tool: include clear subject/action, cinematic anime/photoreal style "
        "when relevant, camera movement, close-up details, lighting, smooth motion, and avoid "
        "text, watermark, gore, or too many scene events in one short clip. "
        "Default to mode=quality and do not set use_smoke_image for real videos. "
        "The tool saves the final mp4 under /opt/data/hermes/generated-videos and "
        "returns a MEDIA:/absolute/path directive suitable for Telegram delivery."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": (
                    "Detailed English video prompt. If the user only gives a short idea, "
                    "rewrite it into a visual prompt with action, style, camera movement, "
                    "lighting, close-up detail, smooth motion, and negative constraints "
                    "such as no text, no watermark, and no gore."
                ),
            },
            "mode": {
                "type": "string",
                "enum": ["test", "quality"],
                "default": "quality",
                "description": "quality is the default for real user videos; test is only for short smoke tests.",
            },
            "style_preset": {
                "type": "string",
                "enum": ["default", "anime_action"],
                "default": "default",
                "description": "Use anime_action for original 2D cel-shaded anime action; default preserves the older prompt style.",
            },
            "keyframe_engine": {
                "type": "string",
                "enum": ["auto", "flux", "animagine"],
                "default": "auto",
                "description": "Keyframe renderer for generated keyframes. animagine requires animagine-xl-3.1.safetensors.",
            },
            "keyframe_frame_mode": {
                "type": "string",
                "enum": ["single_scene", "stylized_panel"],
                "description": "Animagine keyframe composition mode. single_scene forces one full-frame anime film still with no panels or collage; stylized_panel preserves the older stylized behavior. Defaults to single_scene for anime_action with explicit animagine.",
            },
            "input_image_path": {
                "type": "string",
                "description": (
                    "Optional existing keyframe image path. When provided, the tool skips "
                    "Flux/keyframe generation and runs Wan2.1 I2V directly."
                ),
            },
            "use_smoke_image": {
                "type": "boolean",
                "default": False,
                "description": (
                    "For smoke tests only: use the bundled Wan I2V smoke image if no "
                    "input_image_path is provided. Do not use for real user videos."
                ),
            },
            "timeout_seconds": {
                "type": "integer",
                "default": 1800,
                "minimum": 60,
                "maximum": 7200,
                "description": "Maximum wall-clock time to wait for the local pipeline.",
            },
        },
        "required": ["prompt"],
        "additionalProperties": False,
    },
}



def _coerce_ltx_mode(value: Any) -> str:
    mode = str(value or "test").strip().lower()
    return mode if mode in {"test", "standard", "quality"} else "test"


def _coerce_ltx_style(value: Any) -> str:
    style = str(value or "realistic").strip().lower()
    return style if style in {"realistic", "product", "travel", "social_ad", "anime"} else "realistic"


def _coerce_ltx_optional_int(value: Any, low: int, high: int) -> int | None:
    if value is None or value == "":
        return None
    try:
        number = int(value)
    except Exception:
        return None
    return max(low, min(high, number))


def handle_generate_hervid_preview(args: dict[str, Any], **kw) -> str:
    brief = str(args.get("brief") or "").strip()
    if not brief:
        return tool_error("brief is required")
    mode = str(args.get("mode") or "quality").strip().lower()
    if mode not in {"test", "standard", "quality"}:
        mode = "quality"
    timeout = int(args.get("timeout_seconds") or 900)

    body = json.dumps({"brief": brief, "mode": mode}).encode()
    req = urllib.request.Request(
        f"{ORCHESTRATOR_URL}/generate-preview",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            resp = json.loads(r.read())
    except Exception as exc:
        return tool_error(
            f"generate_hervid_preview: orchestrator unreachable at {ORCHESTRATOR_URL} — {exc}. "
            "Ensure start-services.sh has been run."
        )

    image_path = resp.get("image_path")
    preview_id = resp.get("preview_id")
    if not image_path or not preview_id:
        return tool_error(f"generate_hervid_preview: unexpected response: {resp}")

    return tool_result({
        "success": True,
        "preview_id": preview_id,
        "image_path": str(image_path),
        "media": f"MEDIA:{image_path}",
        "send_to_user": f"MEDIA:{image_path}",
        "prompt": resp.get("prompt", ""),
        "character_note": resp.get("character_note", ""),
        "mode": resp.get("mode", mode),
        "note": (
            "Show this keyframe to the user and ask if the composition looks good. "
            "If approved, call generate_hervid_video with this preview_id to render the full video. "
            "If the user wants a different composition, call generate_hervid_preview again."
        ),
    })


def handle_generate_hervid_video(args: dict[str, Any], **kw) -> str:
    preview_id = str(args.get("preview_id") or "").strip()
    brief = str(args.get("brief") or "").strip()
    if not preview_id and not brief:
        return tool_error("either brief or preview_id is required")

    # Auto-run the preview step when no preview_id is supplied.
    # This ensures the keyframe is always shown to the user before the full render,
    # even when the model skips the explicit generate_hervid_preview call.
    if not preview_id:
        preview_timeout = int(args.get("preview_timeout_seconds") or args.get("timeout_seconds") or 900)
        preview_result_str = handle_generate_hervid_preview(
            {"brief": brief, "mode": args.get("mode") or "quality", "timeout_seconds": preview_timeout},
            **kw,
        )
        try:
            preview_result = json.loads(preview_result_str)
        except Exception:
            return preview_result_str
        if "error" in preview_result:
            return preview_result_str
        preview_id = str(preview_result.get("preview_id") or "")
        image_path = preview_result.get("image_path", "")
        return tool_result({
            "success": True,
            "action": "preview_sent",
            "preview_id": preview_id,
            "image_path": str(image_path),
            "media": f"MEDIA:{image_path}",
            "send_to_user": f"MEDIA:{image_path}",
            "prompt": preview_result.get("prompt", ""),
            "note": (
                "Keyframe image sent to user. STOP — do not call generate_hervid_video yet. "
                "Show this image to the user and ask if the composition looks good. "
                f"When the user approves, call generate_hervid_video with preview_id={preview_id!r}. "
                "If the user wants changes, call generate_hervid_preview again with a revised brief."
            ),
        })

    mode = str(args.get("mode") or "quality").strip().lower()
    if mode not in {"test", "standard", "quality"}:
        mode = "quality"
    timeout = _coerce_seq_timeout(args.get("timeout_seconds"))

    # Submit to orchestrator — pass preview_id to skip LM step when available
    body = json.dumps({"brief": brief or "(from preview)", "mode": mode, "preview_id": preview_id or None}).encode()
    req = urllib.request.Request(
        f"{ORCHESTRATOR_URL}/generate-video",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            resp = json.loads(r.read())
    except Exception as exc:
        return tool_error(
            f"generate_hervid_video: orchestrator unreachable at {ORCHESTRATOR_URL} — {exc}. "
            "Ensure start-services.sh has been run."
        )

    job_id = resp.get("id")
    if not job_id:
        return tool_error(f"generate_hervid_video: orchestrator returned no job id: {resp}")

    # Poll until completed or failed
    deadline = time.time() + timeout
    last_status = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(
                f"{ORCHESTRATOR_URL}/pipeline-job/{job_id}", timeout=15
            ) as r:
                job = json.loads(r.read())
        except Exception:
            time.sleep(30)
            continue

        status = job.get("status")
        last_status = status

        if status == "completed":
            video_path = job.get("final_video_path")
            try:
                verified, size = _verify_video_path(video_path)
            except Exception as exc:
                return tool_error(f"generate_hervid_video completed but video invalid: {exc}")
            creative = job.get("creative") or {}
            return tool_result({
                "success": True,
                "status": "completed",
                "video_path": str(verified),
                "media": f"MEDIA:{verified}",
                "send_to_user": f"MEDIA:{verified}",
                "size_bytes": size,
                "runtime_seconds": job.get("runtime_seconds"),
                "prompt_used": creative.get("prompt"),
                "character_note": creative.get("character_note"),
                "duration_seconds": creative.get("total_duration_seconds"),
                "note": "Send the media field verbatim to deliver the video on Telegram.",
            })

        if status == "failed":
            errors = job.get("errors") or []
            logs = job.get("logs") or []
            detail = "; ".join(errors) or (logs[-1] if logs else "unknown error")
            return tool_error(f"generate_hervid_video failed: {detail}")

        time.sleep(30)

    return tool_error(
        f"generate_hervid_video timed out after {timeout}s (job {job_id} still {last_status})"
    )


def handle_generate_ltx_video(args: dict[str, Any], **kw) -> str:
    prompt = str(args.get("prompt") or "").strip()
    if not prompt:
        return tool_error("prompt is required")
    if not LTX_PIPELINE_SCRIPT.exists():
        return tool_error(f"LTX media pipeline script not found: {LTX_PIPELINE_SCRIPT}")

    mode = _coerce_ltx_mode(args.get("mode"))
    style = _coerce_ltx_style(args.get("style"))
    keyframe_engine = _coerce_keyframe_engine(args.get("keyframe_engine"))
    timeout = _coerce_timeout(args.get("timeout_seconds"))
    input_image = str(args.get("input_image_path") or "").strip()
    seed = _coerce_optional_seed(args.get("seed"))

    cmd = [sys.executable, str(LTX_PIPELINE_SCRIPT), "--prompt", prompt, "--mode", mode, "--style", style, "--keyframe-engine", keyframe_engine]
    if DEFAULT_ENV_FILE.exists():
        cmd.extend(["--env-file", str(DEFAULT_ENV_FILE)])
    if input_image:
        cmd.extend(["--input-image", input_image])
    for arg_name, cli_name, low, high in (
        ("width", "--width", 256, 1280),
        ("height", "--height", 256, 720),
        ("duration", "--duration", 1, 5),
        ("fps", "--fps", 8, 24),
        ("steps", "--steps", 1, 30),
    ):
        value = _coerce_ltx_optional_int(args.get(arg_name), low, high)
        if value is not None:
            cmd.extend([cli_name, str(value)])
    if seed is not None:
        cmd.extend(["--seed", str(seed)])

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(LTX_PIPELINE_SCRIPT.parent),
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        payload = _parse_pipeline_json(proc.stdout, proc.stderr)
        if proc.returncode != 0 or payload.get("status") != "completed":
            errors = payload.get("errors") if isinstance(payload.get("errors"), list) else []
            detail = "; ".join(str(item) for item in errors) or proc.stderr[-1600:] or proc.stdout[-1600:]
            return tool_error(f"generate_ltx_video failed: {detail}")
        video_path, size = _verify_video_path(payload.get("video_path"))
        result = {
            "success": True,
            "status": "completed",
            "workflow": payload.get("workflow"),
            "video_path": str(video_path),
            "media": f"MEDIA:{video_path}",
            "send_to_user": f"MEDIA:{video_path}",
            "size_bytes": size,
            "input_image_path": payload.get("input_image_path"),
            "keyframe_generated": payload.get("keyframe_generated"),
            "settings": payload.get("settings"),
            "models": payload.get("models"),
            "runtime_seconds": payload.get("runtime_seconds"),
            "warnings": payload.get("warnings"),
            "comfyui": payload.get("comfyui"),
            "note": "Send the media field verbatim to deliver the LTX video on Telegram.",
        }
        return tool_result(result)
    except subprocess.TimeoutExpired:
        return tool_error(f"generate_ltx_video timed out after {timeout} seconds")
    except Exception as exc:
        return tool_error(f"generate_ltx_video failed: {type(exc).__name__}: {exc}")


def _coerce_seq_timeout(value: Any) -> int:
    try:
        timeout = int(value)
    except Exception:
        timeout = 7200
    return max(300, min(14400, timeout))


def handle_generate_ltx_video_sequence(args: dict[str, Any], **kw) -> str:
    prompt = str(args.get("prompt") or "").strip()
    if not prompt:
        return tool_error("prompt is required")
    if not LTX_SEQUENCE_SCRIPT.exists():
        return tool_error(f"LTX sequence pipeline script not found: {LTX_SEQUENCE_SCRIPT}")

    mode = _coerce_ltx_mode(args.get("mode"))
    style = _coerce_ltx_style(args.get("style"))
    keyframe_engine = _coerce_keyframe_engine(args.get("keyframe_engine"))
    continuity = str(args.get("continuity") or "independent").strip().lower()
    if continuity not in {"last_frame", "independent"}:
        continuity = "independent"
    timeout = _coerce_seq_timeout(args.get("timeout_seconds"))
    input_image = str(args.get("input_image_path") or "").strip()
    character_note = " ".join(str(args.get("character_note") or "").split())
    seed = _coerce_optional_seed(args.get("seed"))
    keyframe_seed = _coerce_optional_seed(args.get("keyframe_seed"))
    total_duration = _coerce_ltx_optional_int(args.get("total_duration_seconds"), 6, 90) or 60
    shot_duration = _coerce_ltx_optional_int(args.get("shot_duration_seconds"), 1, 5) or 3

    shots = args.get("shots")
    shots_file: Path | None = None
    if isinstance(shots, list) and shots:
        cleaned = [" ".join(str(s).strip().split()) for s in shots if str(s).strip()]
        if cleaned:
            fd = tempfile.NamedTemporaryFile("w", suffix=".json", prefix="ltx_shots_", delete=False, encoding="utf-8")
            try:
                json.dump(cleaned, fd, ensure_ascii=False)
            finally:
                fd.close()
            shots_file = Path(fd.name)

    cmd = [
        sys.executable, str(LTX_SEQUENCE_SCRIPT),
        "--prompt", prompt,
        "--mode", mode,
        "--style", style,
        "--keyframe-engine", keyframe_engine,
        "--continuity", continuity,
        "--total-duration-seconds", str(total_duration),
        "--shot-duration-seconds", str(shot_duration),
    ]
    if DEFAULT_ENV_FILE.exists():
        cmd.extend(["--env-file", str(DEFAULT_ENV_FILE)])
    if shots_file is not None:
        cmd.extend(["--shots-file", str(shots_file)])
    if character_note:
        cmd.extend(["--character-note", character_note])
    if keyframe_seed is not None:
        cmd.extend(["--keyframe-seed", str(keyframe_seed)])
    if input_image:
        cmd.extend(["--input-image", input_image])
    for arg_name, cli_name, low, high in (
        ("width", "--width", 256, 1280),
        ("height", "--height", 256, 720),
        ("fps", "--fps", 8, 24),
        ("steps", "--steps", 1, 30),
    ):
        value = _coerce_ltx_optional_int(args.get(arg_name), low, high)
        if value is not None:
            cmd.extend([cli_name, str(value)])
    if seed is not None:
        cmd.extend(["--seed", str(seed)])

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(LTX_SEQUENCE_SCRIPT.parent),
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        payload = _parse_pipeline_json(proc.stdout, proc.stderr)
        if proc.returncode != 0 or payload.get("status") != "completed":
            errors = payload.get("errors") if isinstance(payload.get("errors"), list) else []
            detail = "; ".join(str(item) for item in errors) or proc.stderr[-1600:] or proc.stdout[-1600:]
            return tool_error(f"generate_ltx_video_sequence failed: {detail}")
        video_path, size = _verify_video_path(payload.get("final_video_path"))
        result = {
            "success": True,
            "status": "completed",
            "workflow": payload.get("workflow"),
            "final_video_path": str(video_path),
            "media": f"MEDIA:{video_path}",
            "send_to_user": f"MEDIA:{video_path}",
            "size_bytes": size,
            "shot_count": payload.get("shot_count"),
            "continuity": payload.get("continuity"),
            "manifest_path": payload.get("manifest_path"),
            "settings": payload.get("settings"),
            "runtime_seconds": payload.get("runtime_seconds"),
            "warnings": payload.get("warnings"),
            "note": "Send the media field verbatim to deliver the stitched LTX video on Telegram.",
        }
        return tool_result(result)
    except subprocess.TimeoutExpired:
        return tool_error(f"generate_ltx_video_sequence timed out after {timeout} seconds")
    except Exception as exc:
        return tool_error(f"generate_ltx_video_sequence failed: {type(exc).__name__}: {exc}")
    finally:
        if shots_file is not None:
            shots_file.unlink(missing_ok=True)


def _coerce_mode(value: Any) -> str:
    mode = str(value or "quality").strip().lower()
    return mode if mode in {"test", "quality"} else "quality"


def _coerce_timeout(value: Any) -> int:
    try:
        timeout = int(value)
    except Exception:
        timeout = 1800
    return max(60, min(7200, timeout))


def _parse_pipeline_json(stdout: str, stderr: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for raw in (stdout, stderr):
        text = raw.strip()
        if not text:
            continue
        # Fast path: the stream is exactly one JSON object.
        try:
            payload = json.loads(text)
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            pass
        # Robust fallback: the stream is JSON mixed with non-JSON log/progress
        # lines. Scan for an embedded JSON object, preferring the last one
        # (the final result), using raw_decode so multi-line indented JSON works.
        result: dict[str, Any] | None = None
        for idx, ch in enumerate(text):
            if ch != "{":
                continue
            try:
                payload, _ = decoder.raw_decode(text[idx:])
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                result = payload
        if result is not None:
            return result
    raise RuntimeError(
        "media pipeline did not return JSON; "
        f"stdout={stdout[-1000:]!r}; stderr={stderr[-1000:]!r}"
    )


def _verify_video_path(raw_path: Any) -> tuple[Path, int]:
    if not raw_path:
        raise RuntimeError("media pipeline completed without video_path")
    path = Path(str(raw_path)).expanduser()
    if not path.is_absolute():
        raise RuntimeError(f"video_path is not absolute: {path}")
    resolved = path.resolve()
    if not any(resolved == base or base in resolved.parents for base in ALLOWED_VIDEO_DIRS):
        raise RuntimeError(f"video_path is outside allowed output dirs: {resolved}")
    if resolved.suffix.lower() not in {".mp4", ".webm", ".mov", ".mkv"}:
        raise RuntimeError(f"video_path is not a video file: {resolved}")
    if not resolved.exists() or not resolved.is_file():
        raise RuntimeError(f"video file does not exist: {resolved}")
    size = resolved.stat().st_size
    if size <= 0:
        raise RuntimeError(f"video file is empty: {resolved}")
    return resolved, size


def handle_generate_video(args: dict[str, Any], **kw) -> str:
    prompt = str(args.get("prompt") or "").strip()
    if not prompt:
        return tool_error("prompt is required")
    if not PIPELINE_SCRIPT.exists():
        return tool_error(f"media pipeline script not found: {PIPELINE_SCRIPT}")

    mode = _coerce_mode(args.get("mode"))
    style_preset = _coerce_style_preset(args.get("style_preset") or "default")
    keyframe_engine = _coerce_keyframe_engine(args.get("keyframe_engine"))
    keyframe_frame_mode = _coerce_keyframe_frame_mode(args.get("keyframe_frame_mode"), style_preset, keyframe_engine)
    timeout = _coerce_timeout(args.get("timeout_seconds"))
    input_image = str(args.get("input_image_path") or "").strip()
    if not input_image and bool(args.get("use_smoke_image")):
        input_image = str(SMOKE_IMAGE)

    cmd = [sys.executable, str(PIPELINE_SCRIPT), "--prompt", prompt, "--mode", mode, "--style-preset", style_preset, "--keyframe-engine", keyframe_engine]
    if keyframe_frame_mode:
        cmd.extend(["--keyframe-frame-mode", keyframe_frame_mode])
    if DEFAULT_ENV_FILE.exists():
        cmd.extend(["--env-file", str(DEFAULT_ENV_FILE)])
    if input_image:
        cmd.extend(["--input-image", input_image])

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(PIPELINE_SCRIPT.parent),
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        payload = _parse_pipeline_json(proc.stdout, proc.stderr)
        if proc.returncode != 0 or payload.get("status") != "completed":
            errors = payload.get("errors") if isinstance(payload.get("errors"), list) else []
            detail = "; ".join(str(item) for item in errors) or proc.stderr[-1200:] or proc.stdout[-1200:]
            return tool_error(f"generate_video failed: {detail}")
        video_path, size = _verify_video_path(payload.get("video_path"))
        image_path = payload.get("image_path")
        result = {
            "success": True,
            "status": "completed",
            "video_path": str(video_path),
            "media": f"MEDIA:{video_path}",
            "send_to_user": f"MEDIA:{video_path}",
            "size_bytes": size,
            "image_path": image_path,
            "mode": mode,
            "style_preset": payload.get("style_preset"),
            "keyframe_engine": payload.get("keyframe_engine"),
            "keyframe_frame_mode": payload.get("keyframe_frame_mode"),
            "keyframe_checkpoint": payload.get("keyframe_checkpoint"),
            "keyframe_workflow": payload.get("keyframe_workflow"),
            "keyframe_sampler": payload.get("keyframe_sampler"),
            "keyframe_steps": payload.get("keyframe_steps"),
            "keyframe_cfg": payload.get("keyframe_cfg"),
            "keyframe_resolution": payload.get("keyframe_resolution"),
            "prompt_used": payload.get("prompt_used"),
            "comfyui": payload.get("comfyui"),
            "note": "Send the media field verbatim to deliver the video on Telegram.",
        }
        return tool_result(result)
    except subprocess.TimeoutExpired:
        return tool_error(f"generate_video timed out after {timeout} seconds")
    except Exception as exc:
        return tool_error(f"generate_video failed: {type(exc).__name__}: {exc}")

def _coerce_duration(value: Any) -> int:
    try:
        duration = int(value)
    except Exception:
        duration = 20
    return max(8, min(30, duration))


def _coerce_sequence_timeout(value: Any) -> int:
    try:
        timeout = int(value)
    except Exception:
        timeout = 14400
    return max(300, min(28800, timeout))


def _coerce_continuity(value: Any) -> str:
    continuity = str(value or "last_frame").strip().lower()
    return continuity if continuity in {"last_frame", "independent"} else "last_frame"


def _coerce_style(value: Any) -> str:
    style = str(value or "original_japanese_anime_action").strip()
    return style or "original_japanese_anime_action"


def _coerce_style_preset(value: Any) -> str:
    preset = str(value or "anime_action").strip().lower()
    return preset if preset in {"default", "anime_action"} else "anime_action"


def _coerce_control_mode(value: Any) -> str:
    mode = str(value or "flf2v").strip().lower()
    return mode if mode in {"i2v_last_frame", "flf2v"} else "flf2v"


def _coerce_postprocess(value: Any) -> str:
    mode = str(value or "ffmpeg_fps").strip().lower()
    return mode if mode in {"none", "ffmpeg_fps", "frame_interpolate"} else "ffmpeg_fps"


def _coerce_target_fps(value: Any) -> int:
    try:
        fps = int(value)
    except Exception:
        fps = 16
    return max(8, min(24, fps))


def _coerce_optional_int(value: Any, low: int, high: int) -> int:
    try:
        number = int(value)
    except Exception:
        number = 0
    return max(low, min(high, number))


def _coerce_optional_seed(value: Any) -> int | None:
    try:
        number = int(value)
    except Exception:
        return None
    if number < 0 or number > 2_147_483_647:
        return None
    return number


def _coerce_optional_float(value: Any, low: float, high: float) -> float:
    try:
        number = float(value)
    except Exception:
        number = 0.0
    return max(low, min(high, number))


def _coerce_motion_profile(value: Any) -> str:
    profile = str(value or "balanced").strip().lower()
    return profile if profile in {"rapid", "balanced", "dramatic", "impact"} else "balanced"


def _coerce_storyboard_mode(value: Any) -> str:
    mode = str(value or "auto").strip().lower()
    return mode if mode in {"auto", "intro_action", "action_core", "full_arc"} else "auto"


def _coerce_keyframe_quality_preset(value: Any) -> str:
    preset = str(value or "flux_default").strip().lower()
    return preset if preset in {"flux_default", "anime_action_v2"} else "flux_default"


def _coerce_keyframe_engine(value: Any) -> str:
    engine = str(value or "auto").strip().lower()
    return engine if engine in {"auto", "flux", "animagine"} else "auto"


def _coerce_keyframe_frame_mode(value: Any, style_preset: str, keyframe_engine: str) -> str:
    mode = str(value or "").strip().lower()
    if mode in {"single_scene", "stylized_panel"}:
        return mode
    if style_preset == "anime_action" and keyframe_engine == "animagine":
        return "single_scene"
    return ""


def _coerce_shot_prompt_strength(value: Any) -> str:
    strength = str(value or "balanced").strip().lower()
    return strength if strength in {"light", "balanced", "strong"} else "balanced"


def _coerce_composition_profile(value: Any) -> str:
    profile = str(value or "auto").strip().lower()
    return profile if profile in {"auto", "establishing", "closeup", "action", "impact"} else "auto"


def handle_generate_video_sequence(args: dict[str, Any], **kw) -> str:
    prompt = str(args.get("prompt") or "").strip()
    if not prompt:
        return tool_error("prompt is required")
    if not SEQUENCE_SCRIPT.exists():
        return tool_error(f"media sequence script not found: {SEQUENCE_SCRIPT}")

    mode = _coerce_mode(args.get("mode"))
    duration = _coerce_duration(args.get("duration_seconds"))
    timeout = _coerce_sequence_timeout(args.get("timeout_seconds"))
    continuity = _coerce_continuity(args.get("continuity"))
    style = _coerce_style(args.get("style"))
    style_preset = _coerce_style_preset(args.get("style_preset"))
    control_mode = _coerce_control_mode(args.get("control_mode"))
    postprocess = _coerce_postprocess(args.get("postprocess"))
    target_fps = _coerce_target_fps(args.get("target_fps"))
    shot_count = _coerce_optional_int(args.get("shot_count"), 0, 15)
    shot_duration_seconds = _coerce_optional_float(args.get("shot_duration_seconds"), 0.0, 4.0)
    frames_per_shot = _coerce_optional_int(args.get("frames_per_shot"), 0, 33)
    wan_steps_per_shot = _coerce_optional_int(args.get("wan_steps_per_shot"), 0, 30)
    motion_profile = _coerce_motion_profile(args.get("motion_profile"))
    storyboard_mode = _coerce_storyboard_mode(args.get("storyboard_mode"))
    keyframe_quality_preset = _coerce_keyframe_quality_preset(args.get("keyframe_quality_preset"))
    keyframe_engine = _coerce_keyframe_engine(args.get("keyframe_engine"))
    keyframe_frame_mode = _coerce_keyframe_frame_mode(args.get("keyframe_frame_mode"), style_preset, keyframe_engine)
    keyframe_only_sequence = bool(args.get("keyframe_only_sequence"))
    existing_keyframe_dir = str(args.get("existing_keyframe_dir") or "").strip()
    shot_prompt_strength = _coerce_shot_prompt_strength(args.get("shot_prompt_strength"))
    composition_profile = _coerce_composition_profile(args.get("composition_profile"))
    character_consistency_note = str(args.get("character_consistency_note") or "").strip()
    seed = _coerce_optional_seed(args.get("seed"))

    cmd = [
        sys.executable,
        str(SEQUENCE_SCRIPT),
        "--prompt",
        prompt,
        "--duration-seconds",
        str(duration),
        "--mode",
        mode,
        "--style",
        style,
        "--style-preset",
        style_preset,
        "--control-mode",
        control_mode,
        "--continuity",
        continuity,
        "--postprocess",
        postprocess,
        "--target-fps",
        str(target_fps),
        "--motion-profile",
        motion_profile,
        "--storyboard-mode",
        storyboard_mode,
        "--keyframe-quality-preset",
        keyframe_quality_preset,
        "--keyframe-engine",
        keyframe_engine,
        "--shot-prompt-strength",
        shot_prompt_strength,
        "--composition-profile",
        composition_profile,
    ]
    if keyframe_frame_mode:
        cmd.extend(["--keyframe-frame-mode", keyframe_frame_mode])
    if keyframe_only_sequence:
        cmd.append("--keyframe-only-sequence")
    if existing_keyframe_dir:
        cmd.extend(["--existing-keyframe-dir", existing_keyframe_dir])
    if character_consistency_note:
        cmd.extend(["--character-consistency-note", character_consistency_note])
    if shot_count > 0:
        cmd.extend(["--shot-count", str(shot_count)])
    if shot_duration_seconds > 0:
        cmd.extend(["--shot-duration-seconds", str(shot_duration_seconds)])
    if frames_per_shot > 0:
        cmd.extend(["--frames-per-shot", str(frames_per_shot)])
    if wan_steps_per_shot > 0:
        cmd.extend(["--wan-steps-per-shot", str(wan_steps_per_shot)])
    if seed is not None:
        cmd.extend(["--seed", str(seed)])

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(SEQUENCE_SCRIPT.parent),
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        payload = _parse_pipeline_json(proc.stdout, proc.stderr)
        if proc.returncode != 0 or payload.get("status") != "completed":
            errors = payload.get("errors") if isinstance(payload.get("errors"), list) else []
            detail = "; ".join(str(item) for item in errors) or proc.stderr[-1600:] or proc.stdout[-1600:]
            return tool_error(f"generate_video_sequence failed: {detail}")
        manifest_path = payload.get("manifest_path")
        if keyframe_only_sequence:
            result = {
                "success": True,
                "status": "completed",
                "video_path": None,
                "media": None,
                "send_to_user": None,
                "manifest_path": manifest_path,
                "keyframe_dir": payload.get("keyframe_dir"),
                "existing_keyframe_dir": payload.get("existing_keyframe_dir"),
                "keyframe_paths": payload.get("keyframe_paths"),
                "keyframe_contact_sheet_path": payload.get("keyframe_contact_sheet_path"),
                "duration_seconds": payload.get("duration_seconds_requested"),
                "shot_count": payload.get("shot_count"),
                "selected_shot_titles": payload.get("selected_shot_titles"),
                "storyboard_mode": payload.get("storyboard_mode"),
                "keyframe_quality_preset": payload.get("keyframe_quality_preset"),
                "keyframe_engine": payload.get("keyframe_engine"),
                "keyframe_frame_mode": payload.get("keyframe_frame_mode"),
                "keyframe_checkpoint": payload.get("keyframe_checkpoint"),
                "keyframe_workflow": payload.get("keyframe_workflow"),
                "keyframe_sampler": payload.get("keyframe_sampler"),
                "keyframe_steps": payload.get("keyframe_steps"),
                "keyframe_cfg": payload.get("keyframe_cfg"),
                "keyframe_resolution": payload.get("keyframe_resolution"),
                "warnings": payload.get("warnings"),
                "note": "Review the keyframe paths or contact sheet before running the full Wan render.",
            }
            return tool_result(result)
        video_path, size = _verify_video_path(payload.get("video_path"))
        result = {
            "success": True,
            "status": "completed",
            "video_path": str(video_path),
            "media": f"MEDIA:{video_path}",
            "send_to_user": f"MEDIA:{video_path}",
            "size_bytes": size,
            "manifest_path": manifest_path,
            "duration_seconds": payload.get("duration_seconds_requested"),
            "duration_seconds_actual": payload.get("duration_seconds_actual"),
            "duration_seconds_planned_source": payload.get("duration_seconds_planned_source"),
            "shot_count": payload.get("shot_count"),
            "shot_duration_seconds": payload.get("shot_duration_seconds"),
            "frames_per_shot": payload.get("frames_per_shot"),
            "wan_steps_per_shot": payload.get("wan_steps_per_shot"),
            "motion_profile": payload.get("motion_profile"),
            "storyboard_mode_requested": payload.get("storyboard_mode_requested"),
            "storyboard_mode": payload.get("storyboard_mode"),
            "selected_shot_titles": payload.get("selected_shot_titles"),
            "action_keywords_detected": payload.get("action_keywords_detected"),
            "action_core_selected": payload.get("action_core_selected"),
            "keyframe_quality_preset": payload.get("keyframe_quality_preset"),
            "keyframe_engine": payload.get("keyframe_engine"),
            "keyframe_frame_mode": payload.get("keyframe_frame_mode"),
            "keyframe_checkpoint": payload.get("keyframe_checkpoint"),
            "keyframe_workflow": payload.get("keyframe_workflow"),
            "keyframe_sampler": payload.get("keyframe_sampler"),
            "keyframe_steps": payload.get("keyframe_steps"),
            "keyframe_cfg": payload.get("keyframe_cfg"),
            "keyframe_resolution": payload.get("keyframe_resolution"),
            "keyframe_dir": payload.get("keyframe_dir"),
            "existing_keyframe_dir": payload.get("existing_keyframe_dir"),
            "shot_prompt_strength": payload.get("shot_prompt_strength"),
            "composition_profile": payload.get("composition_profile"),
            "character_consistency_note": payload.get("character_consistency_note"),
            "runtime_seconds": payload.get("runtime_seconds"),
            "continuity": payload.get("continuity"),
            "mode": payload.get("mode"),
            "style_preset": payload.get("style_preset"),
            "control_mode": payload.get("control_mode"),
            "postprocess_requested": payload.get("postprocess_requested"),
            "postprocess_mode": payload.get("postprocess_mode"),
            "effective_postprocess_mode": payload.get("effective_postprocess_mode"),
            "interpolation_model_name": payload.get("interpolation_model_name"),
            "source_fps": payload.get("source_fps"),
            "target_fps": payload.get("target_fps"),
            "actual_fps": payload.get("actual_fps"),
            "postprocessed_video_path": payload.get("postprocessed_video_path"),
            "warnings": payload.get("warnings"),
            "shot_videos": [shot.get("video_path") for shot in payload.get("shots", []) if isinstance(shot, dict)],
            "note": "Send the media field verbatim to deliver the final stitched video on Telegram.",
        }
        return tool_result(result)
    except subprocess.TimeoutExpired:
        return tool_error(f"generate_video_sequence timed out after {timeout} seconds")
    except Exception as exc:
        return tool_error(f"generate_video_sequence failed: {type(exc).__name__}: {exc}")

