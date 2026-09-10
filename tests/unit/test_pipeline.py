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
    assert "Stage 2 avatar" in status_video["summary"]

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
            "streamer_video": {"url": "streamer.mp4", "_updated_at": 130.0},
        },
    }
    kanban = render_pipeline_kanban(state)
    assert "PRODUCTION PIPELINE REAL-TIME KANBAN" in kanban
    assert "Stage 1 [Commentary Script]: ✅ READY" in kanban
    assert "Stage 3 [Streamer Video]: ⚠️ OUT_OF_SYNC" in kanban
    assert "ACTIVE DIRECTOR FOCUS" in kanban
    assert "OUT OF SYNC" in kanban
