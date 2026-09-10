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

# ============================================================================
# Parameter Dependency Graph (DAG)
# Maps configurable parameters to the downstream artifacts they impact.
# ============================================================================
PARAM_DEPENDENCIES: dict[str, dict[str, Any]] = {
    "footageUrl": {
        "targets": ["script", "streamer_video"],
        "reason": "Gameplay video clip changed; commentary duration, scene beat timing, and visual reaction cues need to match the new video.",
    },
    "game": {
        "targets": ["script"],
        "reason": "Game title changed; commentary gamer terminology, mechanics, and references may need adjustments.",
    },
    "gameUrl": {
        "targets": ["script"],
        "reason": "Game reference URL changed; gameplay mechanics and official lore grounding may need re-fetching.",
    },
    "searchGrounding": {
        "targets": ["script"],
        "reason": "Search grounding setting changed; factual research may need re-running.",
    },
    "cta": {
        "targets": ["script"],
        "reason": "Call-to-action changed; the final commentary segment dialogue should be updated.",
    },
    "additionalInstructions": {
        "targets": ["script"],
        "reason": "Script creative instructions or commentary tone changed; dialogue and streamer reaction style need updating.",
    },
    "gamingDevice": {
        "targets": ["script", "avatar", "streamer_video"],
        "reason": "Gaming platform changed (e.g. PC vs Console); player physical actions (hands/controller) and streamer avatar props need alignment.",
    },
    "appearance": {
        "targets": ["avatar", "streamer_video"],
        "reason": "Streamer visual appearance changed; avatar likeness asset needs regeneration.",
    },
    "referenceImageUrl": {
        "targets": ["avatar", "streamer_video"],
        "reason": "Avatar reference image changed; visual likeness asset needs regeneration.",
    },
    "setting": {
        "targets": ["avatar", "streamer_video"],
        "reason": "Streamer background room setting changed; avatar scene backdrop needs regeneration.",
    },
    "aspectRatio": {
        "targets": ["streamer_video"],
        "reason": "Video aspect ratio changed; composite layout and video composition need re-framing.",
    },
}


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
            old_val = spec.get(key)
            if old_val != val:
                spec[key] = val
                changed_params.append(key)
                if key == "additionalInstructions":
                    updated.append("additionalInstructions updated")
                else:
                    updated.append(
                        f"{key}='{val}'" if isinstance(val, str) else f"{key}={val}"
                    )

    # Synchronize structured configuration namespaces
    config = state.setdefault("config", {})
    config["global"] = {
        "footageUrl": spec.get("footageUrl"),
        "gamingDevice": spec.get("gamingDevice", "PC"),
        "aspectRatio": spec.get("aspectRatio", "16:9"),
    }
    config["script"] = {
        "gameTitle": spec.get("game"),
        "gameUrl": spec.get("gameUrl"),
        "searchGrounding": spec.get("searchGrounding", False),
        "cta": spec.get("cta"),
        "additionalInstructions": spec.get("additionalInstructions", ""),
    }
    config["avatar"] = {
        "appearance": spec.get("appearance"),
        "referenceImageUrl": spec.get("referenceImageUrl"),
        "setting": spec.get("setting"),
    }
    config["composite"] = {
        "layout": spec.get("layout", "pip"),
        "pipPlacement": spec.get("pipPlacement", "bottom-right"),
        "stackedPlacement": spec.get("stackedPlacement"),
        "gameplayVolume": spec.get("gameplayVolume", 0.8),
        "streamerVolume": spec.get("streamerVolume", 1.0),
        "subtitles": spec.get("subtitles", True),
    }

    if not updated:
        return "No spec fields provided to update."

    msg_lines = [f"Spec updated successfully: {', '.join(updated)}."]

    # Smart Impact Analysis: Only warn if affected artifacts ALREADY EXIST in state
    existing_artifacts = state.get("artifacts", {})
    impacted_targets: dict[str, list[str]] = defaultdict(list)

    for param in changed_params:
        dep_info = PARAM_DEPENDENCIES.get(param)
        if not dep_info:
            continue
        targets = dep_info["targets"]
        reason = dep_info["reason"]
        for target in targets:
            # Check if this artifact is already generated/present
            if existing_artifacts.get(target):
                if reason not in impacted_targets[target]:
                    impacted_targets[target].append(reason)

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
