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

"""Unified Pipeline Manager for GamerHeads Production DAG.

Manages stages, input/output contracts, DAG impact detection, and real-time
pipeline state evaluations.
"""

import time
from collections import defaultdict
from typing import Any

# ============================================================================
# Stage Contract Definitions
# Each stage defines its input requirements (spec parameters and/or upstream artifacts)
# and its output deliverable target.
# ============================================================================
PIPELINE_STAGES: dict[str, dict[str, Any]] = {
    "script": {
        "name": "Commentary Script",
        "stage_num": 1,
        "spec_inputs": [
            "global.footageUrl",
            "script.game",
            "script.cta",
            "script.additionalInstructions",
            "script.gameUrl",
            "script.searchGrounding",
            "global.gamingDevice",
        ],
        "artifact_inputs": [],
        "output": "script",
        "description": "Timed commentary shot list analyzing gameplay video.",
    },
    "avatar": {
        "name": "Streamer Avatar",
        "stage_num": 2,
        "spec_inputs": [
            "avatar.appearance",
            "avatar.referenceImageUrl",
            "avatar.setting",
            "global.gamingDevice",
            "global.aspectRatio",
        ],
        "artifact_inputs": [],
        "output": "avatar",
        "description": "Golden Anchor portrait image establishing streamer likeness.",
    },
    "streamer_video": {
        "name": "Streamer Video",
        "stage_num": 3,
        "spec_inputs": [
            "global.aspectRatio",
            "global.gamingDevice",
        ],
        # The primary inputs to streamer video generation ARE the upstream artifacts!
        "artifact_inputs": ["script", "avatar"],
        "output": "streamer_video",
        "description": "Consecutive reaction clips rendered with lip-sync and visual continuity.",
    },
    "composite": {
        "name": "Video Composite",
        "stage_num": 4,
        "spec_inputs": [
            "global.footageUrl",
            "global.aspectRatio",
            "composite.layout",
            "composite.pipPlacement",
            "composite.stackedPlacement",
            "composite.gameplayVolume",
            "composite.streamerVolume",
            "composite.subtitles",
        ],
        # Composite's primary inputs are the streamer video and gameplay footage!
        "artifact_inputs": ["streamer_video"],
        "output": "composite",
        "description": "Final video overlay of streamer on top of gameplay footage.",
    },
}

# Scope mapping defining which domain scope each specification parameter belongs to.
PARAM_SCOPE_MAP: dict[str, str] = {
    # global scope
    "footageUrl": "global",
    "gamingDevice": "global",
    "aspectRatio": "global",
    # script scope
    "game": "script",
    "gameUrl": "script",
    "searchGrounding": "script",
    "cta": "script",
    "additionalInstructions": "script",
    # avatar scope
    "appearance": "avatar",
    "referenceImageUrl": "avatar",
    "setting": "avatar",
    # composite scope
    "layout": "composite",
    "pipPlacement": "composite",
    "stackedPlacement": "composite",
    "gameplayVolume": "composite",
    "streamerVolume": "composite",
    "subtitles": "composite",
}

# Default production specifications organized by scope.
DEFAULT_SPEC_SCOPES: dict[str, dict[str, Any]] = {
    "global": {
        "gamingDevice": "PC",
        "aspectRatio": "16:9",
    },
    "script": {
        "searchGrounding": False,
        "additionalInstructions": "",
    },
    "avatar": {},
    "composite": {
        "layout": "pip",
        "pipPlacement": "bottom-right",
        "gameplayVolume": 0.8,
        "streamerVolume": 1.0,
        "subtitles": True,
    },
}

# Parameter dependencies declaring downstream artifact impacts.
PARAM_DEPENDENCIES: dict[str, dict[str, Any]] = {
    "footageUrl": {
        "targets": ["script", "streamer_video", "composite"],
        "reason": "Gameplay video clip changed; commentary duration, scene beat timing, and visual reaction cues need to match the new video.",
    },
    "game": {
        "targets": ["script", "streamer_video"],
        "reason": "Game title changed; commentary gamer terminology and streamer spoken lines need adjustments.",
    },
    "gameUrl": {
        "targets": ["script", "streamer_video"],
        "reason": "Game reference URL changed; gameplay mechanics research and commentary dialogue need updating.",
    },
    "searchGrounding": {
        "targets": ["script", "streamer_video"],
        "reason": "Search grounding setting changed; commentary facts and streamer dialogue need updating.",
    },
    "cta": {
        "targets": ["script", "streamer_video"],
        "reason": "Call-to-action changed; the final commentary segment dialogue and streamer speech should be updated.",
    },
    "additionalInstructions": {
        "targets": ["script", "streamer_video"],
        "reason": "Script creative instructions or commentary tone changed; dialogue and streamer reactions need updating.",
    },
    "gamingDevice": {
        "targets": ["script", "avatar", "streamer_video"],
        "reason": "Gaming platform changed (e.g. PC vs Console); player physical actions (hands/controller) and streamer avatar props need alignment.",
    },
    "appearance": {
        "targets": ["avatar", "streamer_video"],
        "reason": "Streamer visual appearance changed; avatar likeness asset and streamer video need regeneration.",
    },
    "referenceImageUrl": {
        "targets": ["avatar", "streamer_video"],
        "reason": "Avatar reference image changed; visual likeness asset and streamer video need regeneration.",
    },
    "setting": {
        "targets": ["avatar", "streamer_video"],
        "reason": "Streamer background room setting changed; avatar scene backdrop and streamer video need regeneration.",
    },
    "aspectRatio": {
        "targets": ["avatar", "streamer_video", "composite"],
        "reason": "Video aspect ratio changed; avatar portrait framing, streamer video composition, and composite layout need re-framing.",
    },
    "layout": {
        "targets": ["composite"],
        "reason": "Composite layout style changed; video composite layout needs re-rendering.",
    },
    "pipPlacement": {
        "targets": ["composite"],
        "reason": "PIP placement position changed; video composite needs re-rendering.",
    },
    "stackedPlacement": {
        "targets": ["composite"],
        "reason": "Stacked placement position changed; video composite needs re-rendering.",
    },
    "gameplayVolume": {
        "targets": ["composite"],
        "reason": "Gameplay audio volume changed; audio mix needs re-rendering.",
    },
    "streamerVolume": {
        "targets": ["composite"],
        "reason": "Streamer audio volume changed; audio mix needs re-rendering.",
    },
    "subtitles": {
        "targets": ["composite"],
        "reason": "Subtitles toggle changed; video composite needs re-rendering.",
    },
}

# Artifact-to-artifact dependencies declaring which downstream artifacts consume which upstream artifacts.
ARTIFACT_DEPENDENCIES: dict[str, list[str]] = {
    "script": ["streamer_video"],
    "avatar": ["streamer_video"],
    "streamer_video": ["composite"],
    "composite": [],
}

ARTIFACT_CASCADE_REASONS: dict[str, str] = {
    "streamer_video": "Upstream script or avatar asset changed; streamer spoken dialogue and reactions need re-rendering.",
    "composite": "Upstream video track or footage changed; composite video needs re-compositing.",
}


# ============================================================================
# State Mutation and Recording Helpers
# ============================================================================


def record_spec_param(
    state: dict[str, Any], key: str, val: Any, now: float | None = None
) -> bool:
    """Updates a single spec parameter in state and records scope timestamp.

    Returns True if the value changed, False otherwise.
    """
    if now is None:
        now = time.time()
    spec = dict(state.get("spec", {}))
    scope = PARAM_SCOPE_MAP.get(key, "global")
    scope_dict = dict(spec.get(scope, {}))
    old_val = scope_dict.get(key)
    if old_val != val:
        scope_dict[key] = val
        scope_dict["_updated_at"] = now
        spec[scope] = scope_dict
        state["spec"] = spec
        return True
    return False


def record_artifact(
    state: dict[str, Any],
    artifact_name: str,
    artifact_data: Any,
    now: float | None = None,
) -> None:
    """Stores an artifact deliverable in state['artifacts'] with timestamp metadata."""
    if now is None:
        now = time.time()
    artifacts = dict(state.get("artifacts", {}))
    if isinstance(artifact_data, dict):
        artifact_data["_updated_at"] = now
        artifacts[artifact_name] = artifact_data
    else:
        artifacts[artifact_name] = artifact_data
    state["artifacts"] = artifacts


def get_artifact_timestamp(artifact: Any) -> float | None:
    """Extracts timestamp from artifact data if available."""
    if isinstance(artifact, dict):
        return artifact.get("_updated_at")
    return None


# ============================================================================
# Centralized DAG Impact Detection
# ============================================================================


def detect_impact(changed_item: str, state: dict[str, Any]) -> dict[str, list[str]]:
    """Detects existing downstream artifacts in state that are out of sync due to changed_item.

    changed_item can be a spec param (e.g. 'game', 'footageUrl') or an artifact (e.g. 'script', 'avatar').
    Returns a dict mapping artifact name -> list of explanatory reasons.
    """
    existing_artifacts = state.get("artifacts", {})
    impacted_targets: dict[str, list[str]] = defaultdict(list)

    if changed_item in PARAM_DEPENDENCIES:
        dep_info = PARAM_DEPENDENCIES[changed_item]
        targets = dep_info["targets"]
        reason = dep_info["reason"]
        queue = list(targets)
        visited = set()
        while queue:
            target = queue.pop(0)
            if target in visited:
                continue
            visited.add(target)
            if existing_artifacts.get(target):
                target_reason = (
                    reason
                    if target in targets
                    else ARTIFACT_CASCADE_REASONS.get(
                        target, "Upstream artifact dependency updated."
                    )
                )
                if target_reason not in impacted_targets[target]:
                    impacted_targets[target].append(target_reason)
                for downstream in ARTIFACT_DEPENDENCIES.get(target, []):
                    if downstream not in visited:
                        queue.append(downstream)
    elif changed_item in ARTIFACT_DEPENDENCIES:
        queue = list(ARTIFACT_DEPENDENCIES[changed_item])
        visited = set()
        while queue:
            target = queue.pop(0)
            if target in visited:
                continue
            visited.add(target)
            if existing_artifacts.get(target):
                reason = ARTIFACT_CASCADE_REASONS.get(
                    target, f"Upstream artifact '{changed_item}' updated."
                )
                if reason not in impacted_targets[target]:
                    impacted_targets[target].append(reason)
                for downstream in ARTIFACT_DEPENDENCIES.get(target, []):
                    if downstream not in visited:
                        queue.append(downstream)

    return dict(impacted_targets)


# ============================================================================
# Stage Evaluation and Real-Time Kanban Generator
# ============================================================================


def evaluate_stage_status(stage_name: str, state: dict[str, Any]) -> dict[str, Any]:
    """Evaluates the status of a specific stage against context.state.

    Returns a dict with:
    - 'status': 'READY' | 'OUT_OF_SYNC' | 'PENDING' | 'BLOCKED'
    - 'summary': Short human-readable explanation
    """
    spec = state.get("spec", {})
    artifacts = state.get("artifacts", {})
    global_spec = spec.get("global", {})
    script_spec = spec.get("script", {})
    avatar_spec = spec.get("avatar", {})

    existing_artifact = artifacts.get(stage_name)
    artifact_ts = get_artifact_timestamp(existing_artifact)

    if stage_name == "script":
        footage_url = global_spec.get("footageUrl")
        if not footage_url:
            return {
                "status": "BLOCKED",
                "summary": "Missing gameplay footage (footageUrl).",
            }
        if existing_artifact:
            # Check if upstream spec changed after script generation
            footage_ts = global_spec.get("_updated_at")
            script_spec_ts = script_spec.get("_updated_at")
            if (footage_ts and (artifact_ts is None or footage_ts > artifact_ts)) or (
                script_spec_ts and (artifact_ts is None or script_spec_ts > artifact_ts)
            ):
                return {
                    "status": "OUT_OF_SYNC",
                    "summary": "Script was generated with earlier footage/spec; needs re-generation.",
                }
            count = (
                len(existing_artifact.get("segments", []))
                if isinstance(existing_artifact, dict)
                else (
                    len(existing_artifact) if isinstance(existing_artifact, list) else 1
                )
            )
            return {
                "status": "READY",
                "summary": f"Ready ({count} commentary segments).",
            }
        return {
            "status": "PENDING",
            "summary": "Footage available. Ready to generate commentary script.",
        }

    elif stage_name == "avatar":
        has_prompt = bool(
            avatar_spec.get("appearance") or avatar_spec.get("referenceImageUrl")
        )
        if not has_prompt:
            if existing_artifact:
                return {
                    "status": "READY",
                    "summary": "Avatar portrait asset is ready.",
                }
            return {
                "status": "PENDING",
                "summary": "Waiting for streamer visual appearance or reference image.",
            }
        if existing_artifact:
            avatar_spec_ts = avatar_spec.get("_updated_at")
            if avatar_spec_ts and (artifact_ts is None or avatar_spec_ts > artifact_ts):
                return {
                    "status": "OUT_OF_SYNC",
                    "summary": "Avatar spec updated; portrait needs re-generation.",
                }
            return {
                "status": "READY",
                "summary": "Avatar Golden Anchor portrait is ready.",
            }
        return {
            "status": "PENDING",
            "summary": "Avatar spec available. Ready to generate Golden Anchor portrait.",
        }

    elif stage_name == "streamer_video":
        has_script = bool(artifacts.get("script"))
        has_avatar = bool(artifacts.get("avatar"))
        if not has_script or not has_avatar:
            missing = []
            if not has_script:
                missing.append("commentary script")
            if not has_avatar:
                missing.append("streamer avatar")
            return {
                "status": "BLOCKED",
                "summary": f"Prerequisites missing: requires {' and '.join(missing)}.",
            }
        if existing_artifact:
            script_ts = get_artifact_timestamp(artifacts.get("script"))
            avatar_ts = get_artifact_timestamp(artifacts.get("avatar"))
            if (script_ts and (artifact_ts is None or script_ts > artifact_ts)) or (
                avatar_ts and (artifact_ts is None or avatar_ts > artifact_ts)
            ):
                return {
                    "status": "OUT_OF_SYNC",
                    "summary": "Upstream script or avatar updated; streamer video must be re-rendered.",
                }
            return {
                "status": "READY",
                "summary": "Streamer video clips rendered and ready.",
            }
        return {
            "status": "PENDING",
            "summary": "Script and avatar are both ready. Can start rendering streamer video.",
        }

    elif stage_name == "composite":
        has_streamer = bool(artifacts.get("streamer_video"))
        has_footage = bool(global_spec.get("footageUrl"))
        if not has_streamer or not has_footage:
            missing = []
            if not has_footage:
                missing.append("gameplay footage")
            if not has_streamer:
                missing.append("Stage 3 streamer video (streamer reaction video)")
            return {
                "status": "BLOCKED",
                "summary": f"Prerequisites missing: requires {' and '.join(missing)}.",
            }
        if existing_artifact:
            streamer_ts = get_artifact_timestamp(artifacts.get("streamer_video"))
            composite_spec_ts = spec.get("composite", {}).get("_updated_at")
            if (streamer_ts and (artifact_ts is None or streamer_ts > artifact_ts)) or (
                composite_spec_ts
                and (artifact_ts is None or composite_spec_ts > artifact_ts)
            ):
                return {
                    "status": "OUT_OF_SYNC",
                    "summary": "Streamer video or composite settings updated; composite needs re-render.",
                }
            return {
                "status": "READY",
                "summary": "Final composite reaction video is ready.",
            }
        return {
            "status": "PENDING",
            "summary": "Streamer video ready. Can composite PIP reaction video.",
        }

    return {"status": "UNKNOWN", "summary": "Unrecognized stage"}


def render_pipeline_kanban(state: dict[str, Any]) -> str:
    """Renders a real-time production Kanban markdown block based on current session state.

    Injected dynamically into the Director's system instruction on every turn.
    """
    stages = ["script", "avatar", "streamer_video", "composite"]
    lines = ["【PRODUCTION PIPELINE REAL-TIME KANBAN】"]

    has_out_of_sync = False
    out_of_sync_stages = []
    ready_stages = []
    pending_or_blocked = []

    for stage_key in stages:
        meta = PIPELINE_STAGES[stage_key]
        stage_name = meta["name"]
        eval_result = evaluate_stage_status(stage_key, state)
        status = eval_result["status"]
        summary = eval_result["summary"]

        if status == "READY":
            emoji = "✅"
            ready_stages.append(stage_name)
        elif status == "OUT_OF_SYNC":
            emoji = "⚠️"
            has_out_of_sync = True
            out_of_sync_stages.append(stage_name)
        elif status == "PENDING":
            emoji = "⏳"
            pending_or_blocked.append(stage_name)
        else:  # BLOCKED
            emoji = "🛑"
            pending_or_blocked.append(stage_name)

        lines.append(f"- [{stage_name}]: {emoji} {status} - {summary}")

        # Embed Deliverable details directly under each stage card
        artifacts = state.get("artifacts", {})
        if stage_key in artifacts:
            art_val = artifacts[stage_key]
            stale_tag = " (⚠️ Stale - Out of sync)" if status == "OUT_OF_SYNC" else ""
            if stage_key == "script":
                segs = (
                    art_val.get("segments", [])
                    if isinstance(art_val, dict)
                    else (art_val if isinstance(art_val, list) else [])
                )
                total_dur = (
                    art_val.get(
                        "total_duration",
                        segs[-1].get("end_seconds", 0) if segs else 0,
                    )
                    if isinstance(art_val, dict) and segs
                    else 0
                )
                lines.append(
                    f"  * Deliverable{stale_tag}: {len(segs)} commentary segments ({total_dur}s total)"
                )
                for s in segs:
                    dialogue = s.get("dialogue", "")
                    diag_str = f': "{dialogue}"' if dialogue else ""
                    lines.append(
                        f"    - Line {s.get('id')} [{s.get('startTime', '00:00')} - {s.get('endTime', '00:00')}]{diag_str}"
                    )
            elif stage_key == "avatar":
                if isinstance(art_val, dict):
                    art_fn = art_val.get("artifact_name", "avatar.png")
                    lines.append(f"  * Deliverable{stale_tag}: {art_fn}")
                    if "appearance" in art_val:
                        lines.append(f"    - Appearance: {art_val.get('appearance')}")
                    if "setting" in art_val:
                        lines.append(f"    - Room Setting: {art_val.get('setting')}")
                else:
                    lines.append(f"  * Deliverable{stale_tag}: {art_val}")
            elif stage_key == "streamer_video":
                sv_name = (
                    art_val.get("artifact_name", "streamer_video.mp4")
                    if isinstance(art_val, dict)
                    else str(art_val)
                )
                lines.append(f"  * Deliverable{stale_tag}: {sv_name}")
            elif stage_key == "composite":
                comp_name = (
                    art_val.get("artifact_name", "composite_final.mp4")
                    if isinstance(art_val, dict)
                    else str(art_val)
                )
                lines.append(f"  * Deliverable{stale_tag}: {comp_name}")
        else:
            lines.append("  * Deliverable: None created yet")

    lines.append("--------------------------------------------------")
    if has_out_of_sync:
        lines.append(
            f"👉 ACTIVE DIRECTOR FOCUS: Downstream deliverables ({', '.join(out_of_sync_stages)}) are OUT OF SYNC with upstream changes!"
        )
        lines.append(
            "Assess user intent (direct modification vs. exploratory research) and apply director principles to resolve out-of-sync deliverables. Present the reworked deliverable to the user and ask for feedback before continuing. Keep internal agents invisible."
        )
    elif "composite" in [
        s for s in stages if evaluate_stage_status(s, state)["status"] == "READY"
    ]:
        lines.append(
            "👉 ACTIVE DIRECTOR FOCUS: Production is 100% complete! Present the final reaction video to the user and ask for their feedback."
        )
    else:
        script_st = evaluate_stage_status("script", state)["status"]
        avatar_st = evaluate_stage_status("avatar", state)["status"]
        video_st = evaluate_stage_status("streamer_video", state)["status"]

        if script_st == "BLOCKED":
            lines.append(
                "👉 ACTIVE DIRECTOR FOCUS: Awaiting gameplay footage to draft the commentary script."
            )
        elif script_st == "READY" and avatar_st != "READY":
            lines.append(
                "👉 ACTIVE DIRECTOR FOCUS: Commentary script is ready! Present the script to the user, ask for feedback on lines and pacing, and confirm they are satisfied before moving on to design the streamer avatar portrait."
            )
        elif avatar_st == "READY" and script_st != "READY":
            lines.append(
                "👉 ACTIVE DIRECTOR FOCUS: Streamer avatar portrait is ready! Present the visual style and setup to the user, ask for feedback on appearance and room setting, and confirm they are satisfied before drafting the commentary script."
            )
        elif script_st == "READY" and avatar_st == "READY" and video_st != "READY":
            lines.append(
                "👉 ACTIVE DIRECTOR FOCUS: Both commentary script and streamer avatar portrait are ready! Confirm user approval, then proceed to render the streamer reaction video."
            )
        elif video_st == "READY":
            lines.append(
                "👉 ACTIVE DIRECTOR FOCUS: Streamer reaction video is ready! Present the video clip to the user, ask for feedback, and confirm approval before generating the final composite video."
            )
        else:
            lines.append(
                "👉 ACTIVE DIRECTOR FOCUS: Guide the user forward to complete the remaining production deliverables."
            )

    return "\n".join(lines)
