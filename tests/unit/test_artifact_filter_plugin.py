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

"""Unit tests for DeliverableArtifactFilterPlugin."""

import pytest
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions

from app.plugins.artifact_filter_plugin import (
    DEFAULT_DELIVERABLE_PREFIXES,
    DeliverableArtifactFilterPlugin,
)


@pytest.mark.asyncio
async def test_filter_plugin_removes_non_output_artifacts():
    """Verify that artifacts not starting with 'output_' are stripped from artifact_delta."""
    plugin = DeliverableArtifactFilterPlugin()

    delta = {
        # Non-deliverables: user uploads, URL imports, temporary files
        "asset_77a88b.mp4": 1,
        "imported_gameplay_clip.mp4": 1,
        "user_recording.mov": 1,
        "my_gameplay_screen.mp4": 1,
        "temp_game_facts.txt": 1,
        # Deliverables: Stage 2, 3, 4 outputs
        "output_avatar_a1b2c3d4.png": 1,
        "output_avatar_e5f6g7h8.jpg": 2,
        "output_streamer_video_9i0j1k2l.mp4": 1,
        "output_composite_3m4n5o6p.mp4": 1,
    }

    event = Event(
        invocation_id="test-inv-001",
        author="gamerheads_app",
        actions=EventActions(artifact_delta=delta),
    )

    result_event = await plugin.on_event_callback(
        invocation_context=None,
        event=event,
    )

    assert result_event is not None
    filtered_delta = result_event.actions.artifact_delta

    # All non-output_ artifacts must be filtered out
    assert "asset_77a88b.mp4" not in filtered_delta
    assert "imported_gameplay_clip.mp4" not in filtered_delta
    assert "user_recording.mov" not in filtered_delta
    assert "my_gameplay_screen.mp4" not in filtered_delta
    assert "temp_game_facts.txt" not in filtered_delta

    # All output_ artifacts must be retained
    assert filtered_delta["output_avatar_a1b2c3d4.png"] == 1
    assert filtered_delta["output_avatar_e5f6g7h8.jpg"] == 2
    assert filtered_delta["output_streamer_video_9i0j1k2l.mp4"] == 1
    assert filtered_delta["output_composite_3m4n5o6p.mp4"] == 1
    assert len(filtered_delta) == 4


@pytest.mark.asyncio
async def test_filter_plugin_handles_empty_or_none_actions():
    """Verify that events without actions or without artifact_delta pass through unharmed."""
    plugin = DeliverableArtifactFilterPlugin()

    event_no_actions = Event(invocation_id="test-inv-002", author="gamerheads_app")
    res1 = await plugin.on_event_callback(
        invocation_context=None, event=event_no_actions
    )
    assert res1 is None

    event_empty_delta = Event(
        invocation_id="test-inv-003",
        author="gamerheads_app",
        actions=EventActions(artifact_delta={}),
    )
    res2 = await plugin.on_event_callback(
        invocation_context=None, event=event_empty_delta
    )
    assert res2 is None


@pytest.mark.asyncio
async def test_filter_plugin_custom_prefixes():
    """Verify that custom allowed prefixes can be configured."""
    plugin = DeliverableArtifactFilterPlugin(allowed_prefixes=("golden_", "export_"))

    delta = {
        "output_avatar_123.png": 1,
        "golden_anchor.png": 1,
        "export_video.mp4": 1,
    }

    event = Event(
        invocation_id="test-inv-004",
        author="gamerheads_app",
        actions=EventActions(artifact_delta=delta),
    )

    await plugin.on_event_callback(invocation_context=None, event=event)
    filtered = event.actions.artifact_delta

    assert "output_avatar_123.png" not in filtered
    assert filtered["golden_anchor.png"] == 1
    assert filtered["export_video.mp4"] == 1
