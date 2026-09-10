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

from google.adk.tools import ToolContext


async def update_spec(
    footageUrl: str | None = None,
    game: str | None = None,
    referenceImageUrl: str | None = None,
    appearance: str | None = None,
    additionalInstructions: str | None = None,
    aspectRatio: str | None = None,
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

    Args:
        footageUrl: Artifact name or URL of the gameplay footage.
        game: Name of the video game being played.
        referenceImageUrl: Artifact name or URL of streamer likeness reference.
        appearance: Visual description of the streamer avatar.
        additionalInstructions: Creative direction or special instructions.
        aspectRatio: Aspect ratio of the finished video ('9:16' or '16:9').

    Returns:
        Confirmation message detailing which spec keys were updated.
    """
    if tool_context is None:
        return "Warning: No tool context available to persist spec."

    state = tool_context.state
    spec = state.setdefault("spec", {})
    updated = []

    if footageUrl is not None:
        old = spec.get("footageUrl")
        if old != footageUrl:
            spec["footageUrl"] = footageUrl
            state["script_state"] = "stale"
            state["streamer_video_state"] = "stale"
            state["script_confirmed"] = False
            updated.append(f"footageUrl='{footageUrl}' (invalidated downstream script/video)")

    if game is not None:
        spec["game"] = game
        updated.append(f"game='{game}'")

    if referenceImageUrl is not None:
        old = spec.get("referenceImageUrl")
        if old != referenceImageUrl:
            spec["referenceImageUrl"] = referenceImageUrl
            state["avatar_state"] = "stale"
            state["streamer_video_state"] = "stale"
            state["avatar_confirmed"] = False
            updated.append(f"referenceImageUrl='{referenceImageUrl}' (invalidated downstream avatar/video)")

    if appearance is not None:
        spec["appearance"] = appearance
        state["avatar_state"] = "stale"
        updated.append(f"appearance='{appearance}'")

    if additionalInstructions is not None:
        spec["additionalInstructions"] = additionalInstructions
        updated.append("additionalInstructions updated")

    if aspectRatio is not None:
        spec["aspectRatio"] = aspectRatio
        state["streamer_video_state"] = "stale"
        updated.append(f"aspectRatio='{aspectRatio}'")

    if not updated:
        return "No spec fields provided to update."

    return f"Spec updated successfully: {', '.join(updated)}."
