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

"""Unit tests for the centralized PipelineManager."""

import pytest

from app.pipeline import (
    PIPELINE_STAGES,
    detect_impact,
    evaluate_stage_status,
    record_artifact,
    record_spec_param,
    render_pipeline_kanban,
)


def test_pipeline_stages_contracts():
    """Verify that all 4 stages are explicitly defined with their input/output contracts."""
    assert "script" in PIPELINE_STAGES
    assert "avatar" in PIPELINE_STAGES
    assert "streamer_video" in PIPELINE_STAGES
    assert "composite" in PIPELINE_STAGES

    # Verify Stage 3 consumes upstream artifacts
    stage3 = PIPELINE_STAGES["streamer_video"]
    assert "script" in stage3["artifact_inputs"]
    assert "avatar" in stage3["artifact_inputs"]
    assert stage3["output"] == "streamer_video"

    # Verify Stage 4 consumes streamer_video artifact
    stage4 = PIPELINE_STAGES["composite"]
    assert "streamer_video" in stage4["artifact_inputs"]
    assert stage4["output"] == "composite"


def test_record_spec_param_and_timestamps():
    """Verify that record_spec_param updates state and attaches timestamp."""
    state = {}
    t1 = 100.0
    changed = record_spec_param(state, "footageUrl", "clip.mp4", now=t1)
    assert changed is True
    assert state["spec"]["global"]["footageUrl"] == "clip.mp4"
    assert state["spec"]["global"]["_updated_at"] == t1

    # Unchanged value should return False
    changed_again = record_spec_param(state, "footageUrl", "clip.mp4", now=t1 + 10)
    assert changed_again is False
    assert state["spec"]["global"]["_updated_at"] == t1


def test_record_artifact_and_timestamps():
    """Verify that record_artifact attaches timestamp metadata."""
    state = {}
    t1 = 200.0
    record_artifact(state, "script", {"segments": [1, 2, 3]}, now=t1)
    assert state["artifacts"]["script"]["segments"] == [1, 2, 3]
    assert state["artifacts"]["script"]["_updated_at"] == t1


def test_detect_impact_for_spec_param():
    """Verify that changing a spec parameter cascades down the DAG to existing artifacts."""
    state = {
        "artifacts": {
            "script": {"segments": []},
            "streamer_video": {"url": "video.mp4"},
            "composite": {"url": "final.mp4"},
        }
    }
    # footageUrl impacts script, streamer_video, composite
    impacts = detect_impact("footageUrl", state)
    assert "script" in impacts
    assert "streamer_video" in impacts
    assert "composite" in impacts


def test_detect_impact_for_upstream_artifact():
    """Verify that updating script directly cascades down to streamer_video and composite."""
    state = {
        "artifacts": {
            "script": {"segments": []},
            "streamer_video": {"url": "video.mp4"},
            "composite": {"url": "final.mp4"},
        }
    }
    impacts = detect_impact("script", state)
    assert "streamer_video" in impacts
    assert "composite" in impacts
    assert "avatar" not in impacts


def test_evaluate_stage_status_transitions():
    """Verify deterministic stage status transitions based on state and timestamps."""
    state = {"spec": {"global": {}}, "artifacts": {}}

    # 1. Initially, script is BLOCKED because footageUrl is missing
    status_script = evaluate_stage_status("script", state)
    assert status_script["status"] == "BLOCKED"

    # 2. Add footage -> script becomes PENDING
    record_spec_param(state, "footageUrl", "clip.mp4", now=100.0)
    status_script = evaluate_stage_status("script", state)
    assert status_script["status"] == "PENDING"

    # 3. Generate script -> script becomes READY
    record_artifact(state, "script", {"segments": [{"id": 1}]}, now=110.0)
    status_script = evaluate_stage_status("script", state)
    assert status_script["status"] == "READY"

    # 4. Streamer video is BLOCKED because avatar is missing
    status_video = evaluate_stage_status("streamer_video", state)
    assert status_video["status"] == "BLOCKED"
    assert "streamer avatar" in status_video["summary"]

    # 5. Generate avatar -> Streamer video becomes PENDING
    record_artifact(state, "avatar", {"image": "anchor.png"}, now=120.0)
    status_video = evaluate_stage_status("streamer_video", state)
    assert status_video["status"] == "PENDING"

    # 6. Render streamer video -> READY
    record_artifact(state, "streamer_video", {"url": "v.mp4"}, now=130.0)
    status_video = evaluate_stage_status("streamer_video", state)
    assert status_video["status"] == "READY"

    # 7. Now update script at t=140.0 -> streamer_video becomes OUT_OF_SYNC!
    record_artifact(
        state, "script", {"segments": [{"id": 1, "dialogue": "New!"}]}, now=140.0
    )
    status_video = evaluate_stage_status("streamer_video", state)
    assert status_video["status"] == "OUT_OF_SYNC"


def test_render_pipeline_kanban():
    """Verify real-time kanban rendering."""
    state = {
        "spec": {
            "global": {"footageUrl": "test.mp4", "_updated_at": 100.0},
            "script": {},
            "avatar": {},
            "composite": {},
        },
        "artifacts": {
            "script": {"segments": [{"id": 1}], "_updated_at": 150.0},
            "avatar": {"image": "avatar.png", "_updated_at": 120.0},
            "streamer_video": {
                "artifact_name": "streamer.mp4",
                "clips": [{"artifact_name": "clip_1.mp4"}, {"artifact_name": "clip_2.mp4"}],
                "_updated_at": 130.0,
            },
        },
    }
    kanban = render_pipeline_kanban(state)
    assert "PRODUCTION PIPELINE REAL-TIME KANBAN" in kanban
    assert "[Commentary Script]: ✅ READY" in kanban
    assert "[Streamer Video]: ⚠️ OUT_OF_SYNC" in kanban
    assert "Clips: 2 segment clips saved as artifacts" in kanban


@pytest.mark.asyncio
async def test_adk_state_delta_tracking():
    """Verify that record_spec_param and record_artifact emit state_delta on ADK State across turns."""
    from google.adk.events import Event, EventActions
    from google.adk.sessions.in_memory_session_service import InMemorySessionService
    from google.adk.sessions.state import State

    service = InMemorySessionService()
    await service.create_session(app_name="app", user_id="u1", session_id="s1")

    # Turn 1: update footageUrl and record artifact
    sess1 = await service.get_session(app_name="app", user_id="u1", session_id="s1")
    actions1 = EventActions()
    state1 = State(value=sess1.state, delta=actions1.state_delta)
    record_spec_param(state1, "footageUrl", "clip.mp4", now=100.0)
    record_artifact(state1, "avatar", {"artifact_name": "avatar.png"}, now=100.0)

    assert "spec" in actions1.state_delta
    assert "artifacts" in actions1.state_delta
    ev1 = Event(author="test", actions=actions1)
    await service.append_event(session=sess1, event=ev1)

    # Turn 2: update additionalInstructions and avatar artifact
    sess2 = await service.get_session(app_name="app", user_id="u1", session_id="s1")
    assert sess2.state["spec"]["global"]["footageUrl"] == "clip.mp4"
    assert sess2.state["artifacts"]["avatar"]["artifact_name"] == "avatar.png"

    actions2 = EventActions()
    state2 = State(value=sess2.state, delta=actions2.state_delta)
    record_spec_param(state2, "additionalInstructions", "hype commentary", now=200.0)
    record_artifact(state2, "avatar", {"artifact_name": "avatar_v2.png"}, now=200.0)

    assert "spec" in actions2.state_delta
    assert (
        actions2.state_delta["spec"]["script"]["additionalInstructions"]
        == "hype commentary"
    )
    assert "artifacts" in actions2.state_delta
    assert (
        actions2.state_delta["artifacts"]["avatar"]["artifact_name"] == "avatar_v2.png"
    )

    ev2 = Event(author="test", actions=actions2)
    await service.append_event(session=sess2, event=ev2)

    # Verify storage session now contains both updates
    sess3 = await service.get_session(app_name="app", user_id="u1", session_id="s1")
    assert sess3.state["spec"]["global"]["footageUrl"] == "clip.mp4"
    assert sess3.state["spec"]["script"]["additionalInstructions"] == "hype commentary"
    assert sess3.state["artifacts"]["avatar"]["artifact_name"] == "avatar_v2.png"


def test_director_instruction_dynamic_injection():
    """Verify that director_instruction injects global spec, artifacts, and kanban."""
    from unittest.mock import MagicMock

    from google.adk.agents.readonly_context import ReadonlyContext

    from app.agent import director_instruction

    mock_ctx = MagicMock(spec=ReadonlyContext)
    mock_ctx.state = {
        "spec": {
            "global": {
                "footageUrl": "gameplay_epic.mp4",
                "gamingDevice": "Console",
                "aspectRatio": "9:16",
            },
            "script": {"game": "Apex Legends"},
        },
        "artifacts": {
            "script": {
                "segments": [{"id": 1, "duration": 5}],
                "total_duration": 5,
            },
            "avatar": {"artifact_name": "anchor_01.png"},
        },
    }

    instruction = director_instruction(mock_ctx)
    assert "【CURRENT PROJECT SETTINGS】" in instruction
    assert "Gameplay Footage: gameplay_epic.mp4" in instruction
    assert "Gaming Platform: Console" in instruction
    assert "Video Aspect Ratio: 9:16" in instruction
    assert "【PRODUCTION PIPELINE REAL-TIME KANBAN】" in instruction
    assert "[Commentary Script]: ✅ READY" in instruction
    assert "Deliverable: 1 commentary segments (5s total)" in instruction
    assert "[Streamer Avatar]: ✅ READY" in instruction
    assert "Deliverable: anchor_01.png" in instruction


def test_director_instruction_out_of_sync_alignment():
    """Verify that director_instruction aligns artifact sync tag with kanban out-of-sync status."""
    from unittest.mock import MagicMock

    from google.adk.agents.readonly_context import ReadonlyContext

    from app.agent import director_instruction

    mock_ctx = MagicMock(spec=ReadonlyContext)
    mock_ctx.state = {
        "spec": {
            "global": {"footageUrl": "gameplay.mp4"},
            "avatar": {
                "appearance": "cyberpunk girl",
                "setting": "neon room",
                "_updated_at": 200.0,  # Spec changed later
            },
        },
        "artifacts": {
            "avatar": {
                "artifact_name": "old_avatar.png",
                "setting": "old loft",
                "_updated_at": 100.0,  # Older artifact
            }
        },
    }

    instruction = director_instruction(mock_ctx)
    # Kanban section flags OUT OF SYNC
    assert (
        "[Streamer Avatar]: ⚠️ OUT_OF_SYNC - Avatar spec updated; portrait needs re-generation."
        in instruction
    )
    # Deliverable directly under the Stage card flags Stale
    assert "Deliverable (⚠️ Stale - Out of sync): old_avatar.png" in instruction
    assert "Room Setting: old loft" in instruction
