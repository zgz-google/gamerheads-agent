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

"""Specification Management Tool for GamerHeads Director."""

from collections import defaultdict
from typing import Any

from google.adk.tools import ToolContext

from app.pipeline import (
    ARTIFACT_CASCADE_REASONS,
    ARTIFACT_DEPENDENCIES,
    DEFAULT_SPEC_SCOPES,
    PARAM_DEPENDENCIES,
    PARAM_SCOPE_MAP,
    PIPELINE_STAGES,
    detect_impact,
    record_spec_param,
)

__all__ = [
    "ARTIFACT_CASCADE_REASONS",
    "ARTIFACT_DEPENDENCIES",
    "DEFAULT_SPEC_SCOPES",
    "PARAM_DEPENDENCIES",
    "PARAM_SCOPE_MAP",
    "PIPELINE_STAGES",
    "detect_impact",
    "update_avatar_spec",
    "update_global_spec",
    "update_script_spec",
]


def _apply_spec_updates(
    incoming: dict[str, Any], tool_context: ToolContext | None
) -> str:
    """Internal helper to apply spec updates and perform centralized DAG impact detection."""
    if tool_context is None:
        return "Warning: No tool context available to persist spec."

    state = tool_context.state
    spec = dict(state.get("spec", {}))
    spec_modified = False

    # Initialize default scopes if not already present
    for scope_name, defaults in DEFAULT_SPEC_SCOPES.items():
        scope_dict = dict(spec.get(scope_name, {}))
        scope_changed = False
        for def_k, def_v in defaults.items():
            if def_k not in scope_dict:
                scope_dict[def_k] = def_v
                scope_changed = True
        if scope_changed:
            spec[scope_name] = scope_dict
            spec_modified = True

    if spec_modified or "spec" not in state:
        state["spec"] = spec

    updated = []
    changed_params: list[str] = []

    for key, val in incoming.items():
        if val is not None:
            if record_spec_param(state, key, val):
                changed_params.append(key)
                if key == "additionalInstructions":
                    updated.append("additionalInstructions updated")
                else:
                    updated.append(
                        f"{key}='{val}'" if isinstance(val, str) else f"{key}={val}"
                    )

    if not updated:
        return "No spec fields provided to update."

    msg_lines = [f"Spec updated successfully: {', '.join(updated)}."]

    # Smart Impact Analysis: Centralized via detect_impact
    impacted_targets: dict[str, list[str]] = defaultdict(list)
    for param in changed_params:
        impacts = detect_impact(param, state)
        for target, reasons in impacts.items():
            for r in reasons:
                if r not in impacted_targets[target]:
                    impacted_targets[target].append(r)

    if impacted_targets:
        msg_lines.append(
            "\n⚠️ Downstream Impact Detected (Existing Artifacts Out of Sync):"
        )
        for target, reasons in impacted_targets.items():
            msg_lines.append(f"- [{target}]: {' '.join(reasons)}")
        targets_list = ", ".join(f"[{t}]" for t in impacted_targets.keys())
        msg_lines.append(f"Deliverables requiring rework: {targets_list}.")

    return "\n".join(msg_lines)


async def update_global_spec(
    footageUrl: str | None = None,
    gamingDevice: str | None = None,
    aspectRatio: str | None = None,
    tool_context: ToolContext | None = None,
) -> str:
    """Updates global video production parameters in session state.

    Used by the Coordinator (Director) to settle cross-domain production settings:
    - footageUrl: The artifact name or URL of the gameplay clip (e.g. 'imported_123.mp4').
    - gamingDevice: 'PC', 'Console', 'Mobile (Vertical)', 'Mobile (Horizontal)', or 'Hands-free (No device)'.
    - aspectRatio: '9:16' (vertical) or '16:9' (landscape).

    Args:
        footageUrl: Artifact name or URL of the gameplay footage.
        gamingDevice: The platform the streamer plays on.
        aspectRatio: Aspect ratio of the finished video ('9:16' or '16:9').

    Returns:
        Confirmation message detailing updated global keys and downstream impact on existing artifacts.
    """
    incoming = {
        "footageUrl": footageUrl,
        "gamingDevice": gamingDevice,
        "aspectRatio": aspectRatio,
    }
    return _apply_spec_updates(incoming, tool_context)


async def update_avatar_spec(
    appearance: str | None = None,
    referenceImageUrl: str | None = None,
    setting: str | None = None,
    tool_context: ToolContext | None = None,
) -> str:
    """Updates streamer avatar likeness and environment specification in session state.

    Used by AvatarAgent to register or adjust streamer appearance, reference art, or room background:
    - appearance: Text description of what the streamer looks like (e.g. 'purple cat in a hoodie').
    - referenceImageUrl: The artifact name or URL of the avatar's visual reference image.
    - setting: Streamer room or background environment (e.g. 'Cozy cyberpunk loft with neon signs').

    Args:
        appearance: Visual description of the streamer avatar.
        referenceImageUrl: Artifact name or URL of streamer likeness reference.
        setting: Streamer room or background environment.

    Returns:
        Confirmation message detailing updated avatar keys and downstream impact on existing artifacts.
    """
    incoming = {
        "appearance": appearance,
        "referenceImageUrl": referenceImageUrl,
        "setting": setting,
    }
    return _apply_spec_updates(incoming, tool_context)


async def update_script_spec(
    game: str | None = None,
    cta: str | None = None,
    additionalInstructions: str | None = None,
    gameUrl: str | None = None,
    searchGrounding: bool | None = None,
    tool_context: ToolContext | None = None,
) -> str:
    """Updates commentary scriptwriting specification in session state.

    Used by ScriptAgent to register or adjust game context, call to action, or commentary tone:
    - game: The identified or user-stated name of the game (e.g. 'Apex Legends', 'Black Myth: Wukong').
    - cta: Call to action at the end of the video (e.g. 'Follow for more gameplay').
    - additionalInstructions: Tone, style, or specific creative direction for commentary dialogue.
    - gameUrl: Official game website or store URL for factual research.
    - searchGrounding: Whether Google Search grounding is enabled for factual research.

    Args:
        game: Name of the video game being played.
        cta: Call to action for viewers.
        additionalInstructions: Commentary tone, humor style, or special notes.
        gameUrl: Game URL for grounding facts.
        searchGrounding: Enable/disable Google Search grounding.

    Returns:
        Confirmation message detailing updated script keys and downstream impact on existing artifacts.
    """
    incoming = {
        "game": game,
        "cta": cta,
        "additionalInstructions": additionalInstructions,
        "gameUrl": gameUrl,
        "searchGrounding": searchGrounding,
    }
    return _apply_spec_updates(incoming, tool_context)
