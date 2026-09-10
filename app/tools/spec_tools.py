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
    "update_spec",
]


async def update_spec(
    footageUrl: str | None = None,
    game: str | None = None,
    referenceImageUrl: str | None = None,
    appearance: str | None = None,
    additionalInstructions: str | None = None,
    aspectRatio: str | None = None,
    gamingDevice: str | None = None,
    cta: str | None = None,
    gameUrl: str | None = None,
    searchGrounding: bool | None = None,
    tool_context: ToolContext | None = None,
) -> str:
    """Updates the production specification in session state.

    Call this tool whenever you settle or update any production setting:
    - footageUrl: The artifact name or URL of the gameplay clip (e.g. 'imported_123.mp4').
    - game: The identified or user-stated name of the game (e.g. 'Apex Legends', 'Black Myth: Wukong').
    - referenceImageUrl: The artifact name or URL of the avatar's visual reference image.
    - appearance: Text description of what the streamer looks like (e.g. 'a purple cat in a hoodie').
    - additionalInstructions: Tone, style, or specific creative direction from the user.
    - aspectRatio: '9:16' (vertical) or '16:9' (landscape).
    - gamingDevice: 'PC', 'Console', 'Mobile (Vertical)', 'Mobile (Horizontal)', or 'Hands-free (No device)'.
    - cta: Call to action at the end of the video (e.g. 'Follow for more gameplay').
    - gameUrl: Official game website or store URL for factual research.
    - searchGrounding: Whether Google Search grounding is enabled for factual research.

    Args:
        footageUrl: Artifact name or URL of the gameplay footage.
        game: Name of the video game being played.
        referenceImageUrl: Artifact name or URL of streamer likeness reference.
        appearance: Visual description of the streamer avatar.
        additionalInstructions: Creative direction or special instructions.
        aspectRatio: Aspect ratio of the finished video ('9:16' or '16:9').
        gamingDevice: The platform the streamer plays on.
        cta: Call to action for viewers.
        gameUrl: Game URL for grounding facts.
        searchGrounding: Enable/disable Google Search grounding.

    Returns:
        Confirmation message detailing which spec keys were updated and any downstream impact on existing artifacts.
    """
    if tool_context is None:
        return "Warning: No tool context available to persist spec."

    state = tool_context.state
    spec = state.setdefault("spec", {})

    # Initialize default scopes if not already present
    for scope_name, defaults in DEFAULT_SPEC_SCOPES.items():
        scope_dict = spec.setdefault(scope_name, {})
        for def_k, def_v in defaults.items():
            scope_dict.setdefault(def_k, def_v)

    updated = []
    changed_params: list[str] = []

    incoming = {
        "footageUrl": footageUrl,
        "game": game,
        "referenceImageUrl": referenceImageUrl,
        "appearance": appearance,
        "additionalInstructions": additionalInstructions,
        "aspectRatio": aspectRatio,
        "gamingDevice": gamingDevice,
        "cta": cta,
        "gameUrl": gameUrl,
        "searchGrounding": searchGrounding,
    }

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
        msg_lines.append(
            "Action: Please inform the user that these existing artifacts were built with earlier parameters and are now out of sync, and ask or invoke the appropriate specialist agent to re-align or regenerate them."
        )

    return "\n".join(msg_lines)
