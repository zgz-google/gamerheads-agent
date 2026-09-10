# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for VideoAgent, prompts, spec updates, and video production tools."""

import asyncio
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock

import pytest
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.tools import ToolContext
from google.genai import types

from app.agents.video_agent import (
    build_omni_prompt,
    generate_composite_video,
    generate_streamer_video,
    video_agent,
    video_agent_instruction,
)
from app.media.clips import get_ffmpeg_exe
from app.tools.spec_tools import update_composite_spec

# ============================================================================
# 1. Prompt Engineering Tests
# ============================================================================


def test_build_omni_prompt_devices():
    """Tests device-specific instructions and gaze constraints in Omni prompts."""
    # PC
    p_pc = build_omni_prompt("Reacts with shock", "Unbelievable!", 5, "PC")
    assert "<Image0> is the starting frame." in p_pc
    assert "Streamer plays on PC with keyboard and mouse on desk." in p_pc
    assert "Streamer looks naturally at the screen / desk." in p_pc
    assert (
        'Streamer dialogue: "Unbelievable!". Natural speaking motion and lip synchronization.'
        in p_pc
    )
    assert "Duration: 5 seconds." in p_pc

    # Console
    p_console = build_omni_prompt("Laughs", "Good game", 4, "Console")
    assert "Streamer holds a gamepad controller in hands." in p_console

    # Mobile (Vertical)
    p_mv = build_omni_prompt("Taps rapidly", "Watch this", 3, "Mobile (Vertical)")
    assert "Streamer holds a smartphone vertically in portrait mode." in p_mv

    # Mobile (Horizontal)
    p_mh = build_omni_prompt("Swipes", "Nice move", 3, "Mobile (Horizontal)")
    assert "Streamer holds a smartphone horizontally in landscape mode." in p_mh

    # Hands-free
    p_hf = build_omni_prompt("Waves", "Hello chat", 4, "Hands-free (No device)")
    assert "Streamer is completely hands-free with empty hands." in p_hf
    assert (
        "Streamer looks directly into the camera lens with engaging eye contact."
        in p_hf
    )


def test_build_omni_prompt_silence():
    """Tests that empty dialogue produces silence instruction."""
    p_silent = build_omni_prompt("Staring intensely ahead", "", 3, "PC")
    assert "Streamer remains silent." in p_silent


# ============================================================================
# 2. Spec Management Tests
# ============================================================================


@pytest.mark.asyncio
async def test_update_composite_spec():
    """Tests updating composite layout, audio, and subtitle specifications."""
    mock_ctx = MagicMock(spec=ToolContext)
    mock_ctx.state = {
        "spec": {
            "global": {"aspectRatio": "16:9"},
            "composite": {
                "layout": "pip",
                "pipPlacement": "bottom-right",
                "gameplayVolume": 0.8,
                "streamerVolume": 1.0,
                "subtitles": True,
            },
        },
        "artifacts": {},
    }

    res = await update_composite_spec(
        layout="stacked",
        pipPlacement="bottom-left",
        gameplayVolume=0.5,
        subtitles=False,
        tool_context=mock_ctx,
    )

    assert "Spec updated successfully" in res
    comp_spec = mock_ctx.state["spec"]["composite"]
    assert comp_spec["layout"] == "stacked"
    assert comp_spec["pipPlacement"] == "bottom-left"
    assert comp_spec["gameplayVolume"] == 0.5
    assert comp_spec["subtitles"] is False


def test_video_agent_instruction():
    """Tests that video_agent_instruction dynamically injects composite spec."""
    mock_ctx = MagicMock(spec=ReadonlyContext)
    mock_ctx.state = {
        "spec": {
            "composite": {
                "layout": "pip",
                "pipPlacement": "top-right",
                "gameplayVolume": 0.6,
                "streamerVolume": 1.2,
                "subtitles": True,
            }
        }
    }
    instruction = video_agent_instruction(mock_ctx)
    assert "PIP Placement: top-right" in instruction
    assert "Gameplay Volume: 0.6" in instruction
    assert "Streamer Volume: 1.2" in instruction


def test_video_agent_instruction_with_prerequisites_and_deliverables():
    """Tests that video_agent_instruction reports upstream prerequisites and video deliverables."""
    mock_ctx = MagicMock(spec=ReadonlyContext)
    mock_ctx.state = {
        "spec": {
            "global": {
                "footageUrl": "gameplay.mp4",
                "gamingDevice": "PC",
                "aspectRatio": "16:9",
            },
            "composite": {
                "layout": "pip",
                "pipPlacement": "bottom-right",
                "gameplayVolume": 0.8,
                "streamerVolume": 1.0,
                "subtitles": True,
            },
        },
        "artifacts": {
            "script": {
                "segments": [{"id": 1, "duration": 4, "dialogue": "Hello"}],
                "total_duration": 4,
            },
            "avatar": {
                "appearance": "Purple cat in hoodie",
            },
            "streamer_video": {
                "durationSeconds": 4,
                "segmentCount": 1,
            },
            "composite": {
                "durationSeconds": 4,
                "layout": "pip",
            },
        },
    }
    instruction = video_agent_instruction(mock_ctx)
    assert "Gameplay Footage: gameplay.mp4" in instruction
    assert "Gaming Platform: PC" in instruction
    assert "Commentary Script: Present (1 segments, 4s)" in instruction
    assert "Streamer Avatar: Present (Purple cat in hoodie)" in instruction
    assert "Streamer Reaction Video: Present (4s, 1 clips)" in instruction
    assert "Final Composite Video: Present (4s, layout: pip)" in instruction


def test_video_agent_model_retry_options():
    """Tests that video_agent is configured with retry options matching avatar and script agents."""
    assert video_agent.model is not None
    assert video_agent.model.retry_options is not None
    assert video_agent.model.retry_options.attempts == 3


# ============================================================================
# 3. Prerequisites & Blocked Execution Tests
# ============================================================================


@pytest.mark.asyncio
async def test_generate_streamer_video_prerequisites_missing():
    """Tests that generate_streamer_video blocks when script or avatar are missing."""
    mock_ctx = MagicMock(spec=ToolContext)
    mock_ctx.state = {
        "spec": {"global": {"aspectRatio": "16:9"}},
        "artifacts": {},
    }

    res = await generate_streamer_video(mock_ctx)
    assert "Cannot generate streamer video" in res
    assert "Prerequisites missing" in res


@pytest.mark.asyncio
async def test_generate_composite_video_prerequisites_missing():
    """Tests that generate_composite_video blocks when streamer_video is missing."""
    mock_ctx = MagicMock(spec=ToolContext)
    mock_ctx.state = {
        "spec": {"global": {"aspectRatio": "16:9", "footageUrl": "gameplay.mp4"}},
        "artifacts": {},
    }

    res = await generate_composite_video(mock_ctx)
    assert "Cannot generate composite" in res
    assert "Missing streamer reaction video" in res


@pytest.mark.asyncio
async def test_generate_composite_video_blocks_when_streamer_video_out_of_sync():
    """Tests that generate_composite_video refuses to composite if streamer_video is out of sync."""
    mock_ctx = MagicMock(spec=ToolContext)
    mock_ctx.state = {
        "spec": {"global": {"aspectRatio": "16:9", "footageUrl": "gameplay.mp4"}},
        "artifacts": {
            "script": {
                "segments": [{"id": 1, "duration": 3, "dialogue": "New dialogue"}],
                "_updated_at": 200.0,
            },
            "avatar": {
                "artifact_name": "avatar.png",
                "_updated_at": 100.0,
            },
            "streamer_video": {
                "artifact_name": "old_streamer.mp4",
                "_updated_at": 150.0,  # Older than script!
            },
        },
    }

    res = await generate_composite_video(mock_ctx)
    assert "Cannot generate composite" in res
    assert "Streamer reaction video is out of sync" in res


@pytest.mark.asyncio
async def test_generate_streamer_video_not_blocked_when_out_of_sync():
    """Tests that generate_streamer_video is NOT blocked when existing streamer_video is out of sync."""
    mock_ctx = MagicMock(spec=ToolContext)
    mock_ctx.state = {
        "spec": {"global": {"aspectRatio": "16:9"}},
        "artifacts": {
            "script": {
                "segments": [],  # Empty segments will test that it proceeds past prerequisite checks
                "_updated_at": 200.0,
            },
            "avatar": {
                "artifact_name": "avatar.png",
                "_updated_at": 100.0,
            },
            "streamer_video": {
                "artifact_name": "old_streamer.mp4",
                "_updated_at": 150.0,  # Older than script!
            },
        },
    }

    res = await generate_streamer_video(mock_ctx)
    # It should not say "Upstream assets are out of date", it should proceed to script validation
    assert "Upstream assets are out of date" not in res
    assert "Script contains no commentary segments" in res


# ============================================================================
# 4. End-to-End Execution with Mock Artifacts
# ============================================================================


async def _create_test_mp4(path: str, duration: int = 1) -> str:
    ffmpeg = get_ffmpeg_exe()
    cmd = [
        ffmpeg,
        "-f",
        "lavfi",
        "-i",
        f"color=c=black:s=320x240:d={duration}",
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency=1000:duration={duration}",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        path,
        "-y",
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    await proc.communicate()
    return path


@pytest.mark.asyncio
async def test_generate_streamer_and_composite_video_flow(monkeypatch):
    """Tests the full Stage 3 -> Stage 4 pipeline execution flow."""
    monkeypatch.setenv("MOCK_OMNI", "1")
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dummy avatar image
        avatar_img = os.path.join(tmpdir, "avatar.png")
        ffmpeg = get_ffmpeg_exe()
        await (
            await asyncio.create_subprocess_exec(
                ffmpeg,
                "-f",
                "lavfi",
                "-i",
                "color=c=red:s=320x240",
                "-vframes",
                "1",
                avatar_img,
                "-y",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        ).communicate()

        # Create dummy gameplay video
        gameplay_vid = os.path.join(tmpdir, "gameplay.mp4")
        await _create_test_mp4(gameplay_vid, duration=2)

        with open(avatar_img, "rb") as f:
            avatar_bytes = f.read()

        with open(gameplay_vid, "rb") as f:
            gameplay_bytes = f.read()

        saved_artifacts = {}

        mock_ctx = MagicMock(spec=ToolContext)
        mock_ctx.state = {
            "spec": {
                "global": {
                    "footageUrl": "gameplay.mp4",
                    "gamingDevice": "PC",
                    "aspectRatio": "16:9",
                },
                "composite": {
                    "layout": "pip",
                    "pipPlacement": "bottom-right",
                    "gameplayVolume": 0.8,
                    "streamerVolume": 1.0,
                    "subtitles": True,
                },
            },
            "artifacts": {
                "script": {
                    "segments": [
                        {
                            "id": 1,
                            "duration": 3,
                            "dialogue": "Let's go!",
                            "prompt": "Eyes wide",
                        },
                    ]
                },
                "avatar": {
                    "artifact_name": "avatar_portrait.png",
                    "mimeType": "image/png",
                },
            },
        }

        async def mock_load_artifact(name: str):
            if "avatar" in name:
                return types.Part(
                    inline_data=types.Blob(mime_type="image/png", data=avatar_bytes)
                )
            if "gameplay" in name:
                return types.Part(
                    inline_data=types.Blob(mime_type="video/mp4", data=gameplay_bytes)
                )
            if name in saved_artifacts:
                return types.Part(
                    inline_data=types.Blob(
                        mime_type="video/mp4", data=saved_artifacts[name]
                    )
                )
            return None

        async def mock_save_artifact(name: str, part: types.Part):
            if part and part.inline_data:
                saved_artifacts[name] = part.inline_data.data

        mock_ctx.load_artifact = AsyncMock(side_effect=mock_load_artifact)
        mock_ctx.save_artifact = AsyncMock(side_effect=mock_save_artifact)

        # 1. Run Stage 3: generate_streamer_video
        res_streamer = await generate_streamer_video(mock_ctx)
        assert (
            "Successfully generated continuous streamer reaction video" in res_streamer
        )
        assert "streamer_video" in mock_ctx.state["artifacts"]
        streamer_art_name = mock_ctx.state["artifacts"]["streamer_video"][
            "artifact_name"
        ]
        assert streamer_art_name in saved_artifacts

        # 2. Run Stage 4: generate_composite_video
        res_comp = await generate_composite_video(mock_ctx)
        assert "Successfully generated final composite reaction video" in res_comp
        assert "composite" in mock_ctx.state["artifacts"]
        comp_art_name = mock_ctx.state["artifacts"]["composite"]["artifact_name"]
        assert comp_art_name in saved_artifacts
