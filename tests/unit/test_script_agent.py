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

"""Unit tests for ScriptAgent utilities and specialist tools."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from google.adk.tools import ToolContext
from google.genai import types

from app.agents.script_agent import (
    LineEditItem,
    Segment,
    clamp_segments,
    compute_timeline,
    edit_script_lines,
    script_agent,
    watch_gameplay_and_generate_script,
)


def test_segment_pydantic_schema():
    """Tests the Segment schema parsing and validation."""
    seg = Segment(
        id=1,
        startTime="00:00",
        endTime="00:05",
        duration=5,
        prompt="Streamer leans forward staring intently",
        dialogue="[Excited] Look at that shield break!",
    )
    assert seg.id == 1
    assert seg.duration == 5
    assert seg.prompt == "Streamer leans forward staring intently"
    assert seg.dialogue == "[Excited] Look at that shield break!"


def test_clamp_segments():
    """Tests that durations are strictly clamped into the [3, 10] range."""
    raw = [
        {"id": 1, "duration": 1, "prompt": "action 1", "dialogue": "line 1"},
        {"id": 2, "duration": 15, "prompt": "action 2", "dialogue": "line 2"},
        {"id": 3, "duration": 5.4, "prompt": "action 3", "dialogue": "line 3"},
        {"duration": "invalid", "prompt": "action 4", "dialogue": "line 4"},
    ]

    clamped = clamp_segments(raw)
    assert len(clamped) == 4
    # Clamped to minimum 3
    assert clamped[0]["duration"] == 3
    # Clamped to maximum 10
    assert clamped[1]["duration"] == 10
    # Rounded to 5
    assert clamped[2]["duration"] == 5
    # Defaulted to 6
    assert clamped[3]["duration"] == 6
    assert clamped[3]["id"] == 4


def test_compute_timeline():
    """Tests cumulative timing calculations for continuous shots."""
    segments = [
        {"id": 1, "duration": 4, "prompt": "p1", "dialogue": "d1"},
        {"id": 2, "duration": 6, "prompt": "p2", "dialogue": "d2"},
        {"id": 3, "duration": 5, "prompt": "p3", "dialogue": "d3"},
    ]

    timeline = compute_timeline(segments)
    assert len(timeline) == 3

    assert timeline[0]["start_seconds"] == 0
    assert timeline[0]["end_seconds"] == 4
    assert timeline[0]["startTime"] == "00:00"
    assert timeline[0]["endTime"] == "00:04"

    assert timeline[1]["start_seconds"] == 4
    assert timeline[1]["end_seconds"] == 10
    assert timeline[1]["startTime"] == "00:04"
    assert timeline[1]["endTime"] == "00:10"

    assert timeline[2]["start_seconds"] == 10
    assert timeline[2]["end_seconds"] == 15
    assert timeline[2]["startTime"] == "00:10"
    assert timeline[2]["endTime"] == "00:15"


@pytest.mark.asyncio
async def test_edit_script_lines_success():
    """Tests surgical line edits on an existing script."""
    mock_ctx = MagicMock(spec=ToolContext)
    initial_segments = [
        {
            "id": 1,
            "duration": 4,
            "startTime": "00:00",
            "endTime": "00:04",
            "prompt": "look down",
            "dialogue": "hey there",
        },
        {
            "id": 2,
            "duration": 5,
            "startTime": "00:04",
            "endTime": "00:09",
            "prompt": "frown",
            "dialogue": "bad play",
        },
    ]
    mock_ctx.state = {
        "artifacts": {
            "script": {
                "segments": initial_segments,
                "total_duration": 9,
            }
        }
    }

    edits = [
        LineEditItem(line=2, dialogue="[Laughing] What an insane clutch!"),
        LineEditItem(line=1, on_screen="Streamer grins broadly at desk"),
    ]

    res = await edit_script_lines(edits, mock_ctx)
    assert "Successfully updated line(s) [2, 1]" in res

    saved_segs = mock_ctx.state["artifacts"]["script"]["segments"]
    assert saved_segs[0]["prompt"] == "Streamer grins broadly at desk"
    assert saved_segs[0]["dialogue"] == "hey there"  # Untouched
    assert saved_segs[1]["dialogue"] == "[Laughing] What an insane clutch!"
    assert saved_segs[1]["prompt"] == "frown"  # Untouched
    assert saved_segs[0]["duration"] == 4  # Duration preserved


@pytest.mark.asyncio
async def test_edit_script_lines_no_script():
    """Tests error handling when attempting to edit before generating a script."""
    mock_ctx = MagicMock(spec=ToolContext)
    mock_ctx.state = {}

    res = await edit_script_lines([LineEditItem(line=1, dialogue="test")], mock_ctx)
    assert "Error: There is no script to edit yet" in res


@pytest.mark.asyncio
async def test_watch_gameplay_and_generate_script_missing_footage():
    """Tests error when no footageUrl has been set."""
    mock_ctx = MagicMock(spec=ToolContext)
    mock_ctx.state = {"spec": {}}

    res = await watch_gameplay_and_generate_script(mock_ctx)
    assert "Cannot generate script: No gameplay footage has been provided yet" in res
    assert "inform the Director (Coordinator)" in res


@pytest.mark.asyncio
async def test_watch_gameplay_and_generate_script_success():
    """Tests full script generation with mocked video artifact and Gemini model."""
    mock_ctx = MagicMock(spec=ToolContext)
    mock_ctx.state = {
        "spec": {
            "footageUrl": "imported_gameplay.mp4",
            "game": "Apex Legends",
            "gamingDevice": "PC",
            "cta": "Subscribe for more",
        }
    }

    mock_part = types.Part(
        inline_data=types.Blob(
            mime_type="video/mp4",
            data=b"fake_mp4_bytes",
        )
    )
    mock_ctx.load_artifact = AsyncMock(return_value=mock_part)

    fake_segments_json = json.dumps(
        [
            {
                "id": 1,
                "startTime": "00:00",
                "endTime": "00:05",
                "duration": 5,
                "prompt": "Streamer hands on keyboard and mouse, leaning forward",
                "dialogue": "[Excited] Dropping hot into Apex Legends!",
            },
            {
                "id": 2,
                "startTime": "00:05",
                "endTime": "00:10",
                "duration": 5,
                "prompt": "Streamer grins broadly, shaking head",
                "dialogue": "[Laughing] Hit that subscribe button for more!",
            },
        ]
    )

    fake_response = MagicMock()
    fake_response.text = fake_segments_json

    with patch("google.genai.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.models.generate_content.return_value = fake_response

        res = await watch_gameplay_and_generate_script(mock_ctx)

    assert "Successfully generated script with 2 segments" in res
    assert "Line 1 [00:00 - 00:05 (5s)]" in res
    assert "Line 2 [00:05 - 00:10 (5s)]" in res
    assert mock_ctx.state["artifacts"]["script"]["total_duration"] == 10
    assert len(mock_ctx.state["artifacts"]["script"]["segments"]) == 2


@pytest.mark.asyncio
async def test_watch_gameplay_and_generate_script_with_grounding_propagation():
    """Tests that searchGrounding propagates to model call and extracts groundingUrls."""
    mock_ctx = MagicMock(spec=ToolContext)
    mock_ctx.state = {
        "spec": {
            "footageUrl": "imported_gameplay.mp4",
            "game": "Apex Legends",
            "searchGrounding": True,
            "gameUrl": "https://ea.com/apex",
        }
    }
    mock_part = types.Part(
        inline_data=types.Blob(
            mime_type="video/mp4",
            data=b"fake_mp4_bytes",
        )
    )
    mock_ctx.load_artifact = AsyncMock(return_value=mock_part)

    fake_response = MagicMock()
    fake_response.text = json.dumps(
        [
            {
                "id": 1,
                "startTime": "00:00",
                "endTime": "00:05",
                "duration": 5,
                "prompt": "Streamer hands on desk",
                "dialogue": "[Excited] Apex Legends season 20!",
            }
        ]
    )
    chunk_mock = MagicMock()
    chunk_mock.web.uri = "https://ea.com/apex-news-123"
    fake_response.candidates = [
        MagicMock(grounding_metadata=MagicMock(grounding_chunks=[chunk_mock]))
    ]

    with patch("google.genai.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.models.generate_content.return_value = fake_response

        res = await watch_gameplay_and_generate_script(mock_ctx)

    assert "https://ea.com/apex-news-123" in res
    assert mock_ctx.state["artifacts"]["script"]["groundingUrls"] == [
        "https://ea.com/apex-news-123"
    ]


def test_script_agent_attributes():
    """Verifies that script_agent is properly configured with its tools."""
    assert script_agent.name == "script_agent"
    tool_names = [
        getattr(t, "__name__", None) or getattr(t, "name", "")
        for t in script_agent.tools
    ]
    assert tool_names == ["watch_gameplay_and_generate_script", "edit_script_lines"]


def test_build_script_prompt_search_grounding():
    """Tests that search grounding and researched facts are incorporated into the prompt."""
    from app.agents.script_agent import build_script_prompt

    prompt_no_grounding = build_script_prompt(
        title="Apex Legends",
        search_grounding=False,
    )
    assert "NO SEARCH GROUNDING" in prompt_no_grounding

    prompt_grounding = build_script_prompt(
        title="Apex Legends",
        search_grounding=True,
        game_url="https://ea.com/games/apex-legends",
        researched_facts="Season 20 introduces Upgrade Stations and Armor Core system.",
    )
    assert "GOOGLE SEARCH GROUNDING ACTIVE" in prompt_grounding
    assert "Upgrade Stations and Armor Core system" in prompt_grounding
