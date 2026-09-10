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

"""Unit tests for AvatarAgent utilities and specialist tools."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.tools import ToolContext
from google.genai import types

from app.agents.avatar_agent import (
    avatar_agent,
    avatar_agent_instruction,
    build_avatar_prompt,
    generate_golden_anchor_avatar,
    script_device_avatar_rules,
)


def test_script_device_avatar_rules_pc():
    """Tests device rules for PC platform."""
    action, gaze, negative = script_device_avatar_rules("PC")
    assert "keyboard and mouse visible in foreground" in action
    assert "looking slightly DOWN" in gaze
    assert "No handheld game controllers" in negative


def test_script_device_avatar_rules_console():
    """Tests device rules for Console platform."""
    action, gaze, negative = script_device_avatar_rules("Console")
    assert "holding a gaming controller / gamepad" in action
    assert "looking slightly DOWN" in gaze
    assert "No mobile phones, no keyboards" in negative


def test_script_device_avatar_rules_mobile_vertical():
    """Tests device rules for Mobile Vertical platform."""
    action, gaze, negative = script_device_avatar_rules("Mobile (Vertical)")
    assert "vertically (portrait mode)" in action
    assert "Only back of phone is visible" in action
    assert "looking slightly DOWN" in gaze
    assert "No gamepads, no game controllers" in negative


def test_script_device_avatar_rules_mobile_horizontal():
    """Tests device rules for Mobile Horizontal platform."""
    action, gaze, negative = script_device_avatar_rules("Mobile (Horizontal)")
    assert "horizontally (landscape mode)" in action
    assert "Only back of phone is visible" in action
    assert "looking slightly DOWN" in gaze
    assert "No gamepads, no game controllers" in negative


def test_script_device_avatar_rules_hands_free():
    """Tests device rules for Hands-free platform."""
    action, gaze, negative = script_device_avatar_rules("Hands-free (No device)")
    assert "hands completely empty and free" in action
    assert "DIRECTLY into the camera lens" in gaze
    assert "No controllers, no keyboards" in negative


def test_build_avatar_prompt():
    """Tests prompt construction with customized and default attributes."""
    prompt = build_avatar_prompt(
        appearance="A futuristic cyborg gamer with neon blue visor",
        setting="High-tech cyberpunk studio with holographic displays",
        device="PC",
        has_reference_image=False,
    )
    assert "Professional gaming livestreamer avatar portrait." in prompt
    assert "A futuristic cyborg gamer with neon blue visor" in prompt
    assert "High-tech cyberpunk studio with holographic displays" in prompt
    assert "keyboard and mouse visible in foreground" in prompt
    assert "Negative Prompt:" in prompt

    # With reference image
    ref_prompt = build_avatar_prompt(
        device="Hands-free (No device)",
        has_reference_image=True,
    )
    assert (
        "Maintain consistent identity and facial features with the reference image."
        in ref_prompt
    )
    assert "DIRECTLY into the camera lens" in ref_prompt


def test_avatar_agent_attributes():
    """Verifies that avatar_agent is properly configured with its tools."""
    assert avatar_agent.name == "avatar_agent"
    tool_names = [
        getattr(t, "__name__", None) or getattr(t, "name", "")
        for t in avatar_agent.tools
    ]
    assert tool_names == ["update_avatar_spec", "generate_golden_anchor_avatar"]


@pytest.mark.asyncio
async def test_update_avatar_spec():
    """Verifies update_avatar_spec updates avatar spec in session state."""
    from app.tools.spec_tools import update_avatar_spec

    mock_ctx = MagicMock(spec=ToolContext)
    mock_ctx.state = {"spec": {"avatar": {}}}
    res = await update_avatar_spec(
        appearance="Cyborg cat",
        setting="Neon loft",
        tool_context=mock_ctx,
    )
    assert "Spec updated successfully" in res
    assert mock_ctx.state["spec"]["avatar"]["appearance"] == "Cyborg cat"
    assert mock_ctx.state["spec"]["avatar"]["setting"] == "Neon loft"


def test_avatar_agent_instruction_dynamic_injection():
    """Tests that avatar_agent_instruction dynamically injects state into prompt."""
    mock_context = MagicMock(spec=ReadonlyContext)
    mock_context.state = {
        "spec": {
            "global": {
                "gamingDevice": "Console",
                "aspectRatio": "9:16",
            },
            "avatar": {
                "appearance": "Purple cat in hoodie",
                "referenceImageUrl": "ref_portrait.png",
                "setting": "Cozy loft room with neon sign",
            },
        },
        "artifacts": {
            "avatar": {
                "artifact_name": "avatar_1234.png",
                "appearance": "Purple cat in hoodie",
                "setting": "Cozy loft room with neon sign",
                "gamingDevice": "Console",
                "aspectRatio": "9:16",
            },
        },
    }

    instruction = avatar_agent_instruction(mock_context)
    assert "Appearance: Purple cat in hoodie" in instruction
    assert "Reference Image: ref_portrait.png" in instruction
    assert "Room Setting: Cozy loft room with neon sign" in instruction
    assert "Gaming Platform: Console" in instruction
    assert "Aspect Ratio: 9:16" in instruction
    assert "Existing Avatar Deliverable: avatar_1234.png (Ready)" in instruction
    assert "Current Deliverable Details:" in instruction
    assert "Setting: Cozy loft room with neon sign" in instruction


def test_avatar_agent_instruction_out_of_sync():
    """Tests that avatar_agent_instruction flags out-of-sync deliverable when spec was updated."""
    mock_context = MagicMock(spec=ReadonlyContext)
    mock_context.state = {
        "spec": {
            "global": {"gamingDevice": "PC"},
            "avatar": {
                "appearance": "Retro pixel knight",
                "_updated_at": 200.0,
            },
        },
        "artifacts": {
            "avatar": {
                "artifact_name": "pixel_01.png",
                "_updated_at": 100.0,
            },
        },
    }

    instruction = avatar_agent_instruction(mock_context)
    assert (
        "Existing Avatar Deliverable: pixel_01.png (⚠️ OUT OF SYNC: Avatar spec updated; portrait needs re-generation.)"
        in instruction
    )


@pytest.mark.asyncio
async def test_generate_golden_anchor_avatar_missing_prerequisites():
    """Tests that missing both appearance and referenceImageUrl blocks generation."""
    mock_ctx = MagicMock(spec=ToolContext)
    mock_ctx.state = {
        "spec": {
            "global": {"gamingDevice": "PC"},
            "avatar": {},
        }
    }

    res = await generate_golden_anchor_avatar(mock_ctx)
    assert "Cannot generate avatar: Missing streamer visual appearance" in res
    assert "inform the Director (Coordinator)" in res


@pytest.mark.asyncio
async def test_generate_golden_anchor_avatar_success():
    """Tests successful avatar generation saving artifact and updating state."""
    mock_ctx = MagicMock(spec=ToolContext)
    mock_ctx.state = {
        "spec": {
            "global": {
                "gamingDevice": "PC",
                "aspectRatio": "16:9",
            },
            "avatar": {
                "appearance": "Energetic esports player with red headset",
                "setting": "Pro streamer battle station",
            },
        },
        "artifacts": {},
    }
    mock_ctx.save_artifact = AsyncMock()

    fake_response = MagicMock()
    fake_cand = MagicMock()
    fake_part = MagicMock()
    fake_part.inline_data = types.Blob(
        mime_type="image/png",
        data=b"fake_avatar_png_bytes",
    )
    fake_cand.content.parts = [fake_part]
    fake_response.candidates = [fake_cand]

    with patch("app.agents.avatar_agent.get_global_genai_client") as mock_get_client:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.models.generate_content.return_value = fake_response

        res = await generate_golden_anchor_avatar(mock_ctx)

    assert "Successfully generated Golden Anchor avatar portrait!" in res
    assert "Aspect Ratio: 16:9" in res
    assert "Gaming Platform: PC" in res
    assert mock_ctx.save_artifact.called
    saved_filename, saved_part = mock_ctx.save_artifact.call_args[0]
    assert saved_filename.startswith("output_avatar_")
    assert saved_part.inline_data.data == b"fake_avatar_png_bytes"

    # Check state deliverable
    avatar_art = mock_ctx.state["artifacts"]["avatar"]
    assert avatar_art["artifact_name"] == saved_filename
    assert avatar_art["aspectRatio"] == "16:9"
    assert avatar_art["appearance"] == "Energetic esports player with red headset"
    assert avatar_art["gamingDevice"] == "PC"
    assert "_updated_at" in avatar_art


@pytest.mark.asyncio
async def test_generate_golden_anchor_avatar_with_reference_image():
    """Tests avatar generation when referenceImageUrl artifact is provided."""
    mock_ctx = MagicMock(spec=ToolContext)
    mock_ctx.state = {
        "spec": {
            "global": {"gamingDevice": "Mobile (Vertical)", "aspectRatio": "9:16"},
            "avatar": {
                "referenceImageUrl": "ref_uploaded.png",
                "setting": "Bedroom neon studio",
            },
        },
        "artifacts": {},
    }
    mock_ctx.save_artifact = AsyncMock()

    ref_part = types.Part(
        inline_data=types.Blob(
            mime_type="image/png",
            data=b"reference_source_bytes",
        )
    )
    mock_ctx.load_artifact = AsyncMock(return_value=ref_part)

    fake_response = MagicMock()
    fake_cand = MagicMock()
    fake_part = MagicMock()
    fake_part.inline_data = types.Blob(
        mime_type="image/png",
        data=b"generated_from_reference_bytes",
    )
    fake_cand.content.parts = [fake_part]
    fake_response.candidates = [fake_cand]

    with patch("app.agents.avatar_agent.get_global_genai_client") as mock_get_client:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.models.generate_content.return_value = fake_response

        res = await generate_golden_anchor_avatar(mock_ctx)

    assert "Successfully generated Golden Anchor avatar portrait!" in res
    assert mock_ctx.load_artifact.called
    avatar_art = mock_ctx.state["artifacts"]["avatar"]
    assert avatar_art["hasReferenceImage"] is True
    assert avatar_art["aspectRatio"] == "9:16"


@pytest.mark.asyncio
async def test_avatar_regeneration_triggers_downstream_out_of_sync():
    """Tests that regenerating avatar marks downstream streamer_video OUT_OF_SYNC in DAG."""
    from app.pipeline import render_pipeline_kanban

    mock_ctx = MagicMock(spec=ToolContext)
    mock_ctx.state = {
        "spec": {
            "global": {
                "gamingDevice": "PC",
                "aspectRatio": "16:9",
                "footageUrl": "gameplay.mp4",
            },
            "script": {"game": "Apex Legends"},
            "avatar": {
                "appearance": "New updated cyberpunk appearance",
            },
        },
        "artifacts": {
            "script": {"segments": [{"id": 1, "duration": 5}], "_updated_at": 100.0},
            "avatar": {"artifact_name": "avatar_old.png", "_updated_at": 100.0},
            "streamer_video": {"url": "video.mp4", "_updated_at": 100.0},
        },
    }
    mock_ctx.save_artifact = AsyncMock()

    fake_response = MagicMock()
    fake_cand = MagicMock()
    fake_part = MagicMock()
    fake_part.inline_data = types.Blob(
        mime_type="image/png",
        data=b"new_avatar_bytes",
    )
    fake_cand.content.parts = [fake_part]
    fake_response.candidates = [fake_cand]

    with patch("app.agents.avatar_agent.get_global_genai_client") as mock_get_client:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.models.generate_content.return_value = fake_response

        await generate_golden_anchor_avatar(mock_ctx)

    # State avatar artifact updated with newer timestamp > 100.0
    assert mock_ctx.state["artifacts"]["avatar"]["_updated_at"] > 100.0

    # Downstream Stage 3 status in pipeline
    kanban = render_pipeline_kanban(mock_ctx.state)
    assert "[Streamer Video]: ⚠️ OUT_OF_SYNC" in kanban
