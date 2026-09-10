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

"""AvatarAgent: Specialist agent for streamer avatar generation (The Golden Anchor)."""

from __future__ import annotations

import base64
import os
import uuid

import aiohttp
from dotenv import load_dotenv
from google import genai
from google.adk.agents import Agent
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.models import Gemini
from google.adk.tools import ToolContext
from google.genai import types

from app.pipeline import evaluate_stage_status, record_artifact
from app.tools.spec_tools import update_avatar_spec

load_dotenv()

MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
AVATAR_MODEL = os.getenv("AVATAR_MODEL", "gemini-3.1-flash-lite-image")

SAFETY_BLOCK_NONE = [
    types.SafetySetting(
        category=category,
        threshold=types.HarmBlockThreshold.BLOCK_NONE,
    )
    for category in [
        types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        types.HarmCategory.HARM_CATEGORY_HARASSMENT,
        types.HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,
    ]
]


def resolve_model_id(model: str | None, fallback: str) -> str:
    """Resolves model ID handling retired names."""
    m = (model or fallback or "").strip()
    if m == "gemini-3.6-flash-lite":
        return "gemini-3.5-flash-lite"
    if m == "gemini-3.1-flash-image":
        return "gemini-3.1-flash-lite-image"
    return m


def get_global_genai_client() -> genai.Client:
    """Returns a GenAI client pointed at the global endpoint where image models live."""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if api_key:
        return genai.Client(api_key=api_key)
    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    return genai.Client(vertexai=True, project=project, location="global")


# ============================================================================
# 1. Prompt Engineering & Load-Bearing Device Rules
# ============================================================================


def script_device_avatar_rules(device: str) -> tuple[str, str, str]:
    """Returns platform-specific body posture, gaze direction, and negative exclusions.

    Load-bearing rules:
    - Gaze: looking down at monitor/phone for gaming platforms; looking directly into lens for Hands-free.
    - Device: explicit action and setting rules per platform.
    - Negative constraints: strictly exclude conflicting peripherals (e.g. no controllers when on PC).
    """
    gaze = "- Gaze: Streamer is looking slightly DOWN (at the monitor/phone), NOT directly into the lens."
    action = ""
    negative = ""

    if device == "Mobile (Vertical)":
        action = "\n- Action: Streamer is holding and playing a mobile phone vertically (portrait mode). Only back of phone is visible."
        negative = "No gamepads, no game controllers, no keyboards, no mouse."
    elif device == "Mobile (Horizontal)":
        action = "\n- Action: Streamer is holding and playing a mobile phone horizontally (landscape mode). Only back of phone is visible."
        negative = "No gamepads, no game controllers, no keyboards, no mouse."
    elif device == "PC":
        action = "\n- Setting: Streamer is at a gaming desk setup with keyboard and mouse visible in foreground."
        negative = "No handheld game controllers, no mobile phones, no gamepads."
    elif device == "Console":
        action = "\n- Action: Streamer is holding a gaming controller / gamepad with both hands."
        negative = "No mobile phones, no keyboards, no mouse."
    elif device == "Hands-free (No device)":
        action = "\n- Setting: Streamer is sitting in their streaming room, hands completely empty and free."
        gaze = "- Gaze: Streamer looks DIRECTLY into the camera lens with engaging eye contact."
        negative = (
            "No controllers, no keyboards, no mouse, no desk blocking view, no phones."
        )
    else:  # Default to PC
        action = "\n- Setting: Streamer is at a gaming desk setup with keyboard and mouse visible in foreground."
        negative = "No handheld game controllers, no mobile phones, no gamepads."

    return action, gaze, negative


def build_avatar_prompt(
    appearance: str = "",
    setting: str = "",
    device: str = "PC",
    has_reference_image: bool = False,
) -> str:
    """Constructs the photorealistic Golden Anchor avatar prompt.

    Ports the original GamerHeads prompt engineering rules:
    - Cinematic photorealistic lighting and medium eye-level framing.
    - Reference persona consistency when reference image is provided.
    - Platform-specific gaze, action, and negative constraints.
    """
    action_instruction, gaze_instruction, negative_extra = script_device_avatar_rules(
        device
    )

    identity_line = (
        "\nMaintain consistent identity and facial features with the reference image."
        if has_reference_image
        else ""
    )
    final_appearance = appearance.strip() or (
        "Consistent with reference persona"
        if has_reference_image
        else "Energetic young adult gamer in modern gaming attire"
    )
    final_setting = setting.strip() or (
        "Professional streaming room with warm accent lighting"
        if has_reference_image
        else "Modern streaming room with RGB accent glow and gaming gear"
    )

    return f"""Professional gaming livestreamer avatar portrait.
Photorealistic, cinematic lighting, eye-level framing, medium shot.{identity_line}
- Appearance: {final_appearance}
- Background: {final_setting}
{gaze_instruction}{action_instruction}
High detail, sharp focus, vibrant aesthetic.
Negative Prompt: Blurry, distorted, low quality, CGI game graphics, animated overlays, cartoonish. {negative_extra}""".strip()


# ============================================================================
# 2. Specialist Tools for AvatarAgent
# ============================================================================


async def generate_golden_anchor_avatar(tool_context: ToolContext) -> str:
    """Generates the Golden Anchor streamer portrait avatar and saves it to the session Artifact Store.

    This tool reads the streamer visual specification (appearance, referenceImageUrl, setting)
    and global parameters (gamingDevice, aspectRatio) from session state, constructs the
    disciplined avatar prompt, invokes image generation, and saves the deliverable into
    session artifacts.

    Call this tool when:
    - Initial avatar creation is needed (Stage 2).
    - The user wants to adjust/change the streamer's appearance, room setting, or device.

    Returns:
        A confirmation string with the saved artifact name, aspect ratio, and visual summary.
    """
    state = tool_context.state
    spec = state.get("spec", {})
    global_spec = spec.get("global", {})
    avatar_spec = spec.get("avatar", {})

    appearance = avatar_spec.get("appearance", "")
    ref_image_target = avatar_spec.get("referenceImageUrl")
    setting = avatar_spec.get("setting", "")
    device = global_spec.get("gamingDevice", "PC")
    aspect_ratio = global_spec.get("aspectRatio", "16:9")

    # 1. Prerequisite check: Must have appearance or reference image
    if not appearance and not ref_image_target:
        return (
            "Cannot generate avatar: Missing streamer visual appearance or reference image. "
            "Drawing a Golden Anchor portrait requires knowing what the streamer looks like (appearance) "
            "or having a visual likeness reference image (referenceImageUrl). "
            "Please inform the Director (Coordinator) that streamer appearance or a reference image is required, "
            "so the Director can ask the user."
        )

    # 2. Retrieve reference image bytes if provided
    ref_bytes: bytes | None = None
    ref_mime = "image/png"
    if ref_image_target:
        try:
            part = await tool_context.load_artifact(ref_image_target)
            if part and part.inline_data and part.inline_data.data:
                raw_data = part.inline_data.data
                if isinstance(raw_data, str):
                    clean_str = raw_data.strip().rstrip("=")
                    padded = clean_str + "=" * ((4 - (len(clean_str) % 4)) % 4)
                    ref_bytes = base64.b64decode(padded)
                else:
                    ref_bytes = raw_data
                if part.inline_data.mime_type:
                    ref_mime = part.inline_data.mime_type
        except Exception:
            pass

        if not ref_bytes and (
            ref_image_target.startswith("http://")
            or ref_image_target.startswith("https://")
        ):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(ref_image_target) as resp:
                        if resp.status == 200:
                            ref_bytes = await resp.read()
                            ct = resp.headers.get("Content-Type")
                            if ct:
                                ref_mime = ct.split(";")[0].strip()
            except Exception as e:
                return f"Error downloading reference image from {ref_image_target}: {e}"

    # 3. Build disciplined prompt
    prompt = build_avatar_prompt(
        appearance=appearance,
        setting=setting,
        device=device,
        has_reference_image=bool(ref_bytes),
    )

    # 4. Generate avatar image via Google GenAI (Global endpoint, multimodal generate_content)
    model_id = resolve_model_id(os.getenv("AVATAR_MODEL"), AVATAR_MODEL)
    client = get_global_genai_client()

    parts: list[types.Part] = [types.Part.from_text(text=prompt)]
    if ref_bytes:
        parts.append(types.Part.from_bytes(data=ref_bytes, mime_type=ref_mime))

    image_bytes: bytes | None = None
    out_mime = "image/jpeg"

    try:
        response = client.models.generate_content(
            model=model_id,
            contents=parts,
            config=types.GenerateContentConfig(
                temperature=0.5,
                response_modalities=["IMAGE", "TEXT"],
                image_config=types.ImageConfig(
                    aspect_ratio=aspect_ratio, image_size="1K"
                ),
                safety_settings=SAFETY_BLOCK_NONE,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    disable=True
                ),
            ),
        )

        for cand in response.candidates or []:
            if cand.content and cand.content.parts:
                for p in cand.content.parts:
                    if p.inline_data and p.inline_data.data:
                        raw = p.inline_data.data
                        if isinstance(raw, str):
                            clean = raw.strip().rstrip("=")
                            padded = clean + "=" * ((4 - (len(clean) % 4)) % 4)
                            image_bytes = base64.b64decode(padded)
                        else:
                            image_bytes = raw
                        if p.inline_data.mime_type:
                            out_mime = p.inline_data.mime_type
                        break
                if image_bytes:
                    break

    except Exception as err:
        return f"Error executing avatar image generation model ({model_id}): {err}"

    if not image_bytes:
        return f"Error: Avatar image model ({model_id}) did not return any image data."

    # 5. Save generated image into ADK Artifact Service
    ext = "jpg" if "jpeg" in out_mime.lower() else "png"
    filename = f"avatar_{uuid.uuid4().hex[:8]}.{ext}"
    artifact_part = types.Part(
        inline_data=types.Blob(
            mime_type=out_mime,
            data=image_bytes,
        )
    )
    await tool_context.save_artifact(filename, artifact_part)

    # 6. Record deliverable in state["artifacts"]["avatar"]
    record_artifact(
        state,
        "avatar",
        {
            "artifact_name": filename,
            "mimeType": out_mime,
            "aspectRatio": aspect_ratio,
            "appearance": appearance or "Consistent with reference persona",
            "setting": setting,
            "gamingDevice": device,
            "hasReferenceImage": bool(ref_bytes),
        },
    )

    return (
        f"Successfully generated Golden Anchor avatar portrait!\n\n"
        f"- Artifact Name: {filename}\n"
        f"- Aspect Ratio: {aspect_ratio}\n"
        f"- Gaming Platform: {device}\n"
        f"- Appearance: {appearance or 'Consistent with reference persona'}\n"
        f"- Room Setting: {setting or 'Default studio'}\n\n"
        f"The Golden Anchor portrait is locked in session state as the visual continuity anchor for Stage 3 video rendering."
    )


# ============================================================================
# 3. AvatarAgent Definition & Dynamic Instruction
# ============================================================================

AVATAR_AGENT_INSTRUCTION = """You are the expert Character Designer and Art Director (AvatarAgent).
Your sole purpose is creating and refining the streamer's "Golden Anchor" portrait image and managing the avatar visual specification (appearance, referenceImageUrl, setting).

【THE GOLDEN ANCHOR CONTRACT】
- In GamerHeads' Diamond DAG, the Golden Anchor avatar is Stage 2.
- The avatar image serves as the conditioning anchor (<Image0>) for all subsequent reaction video clips rendered in Stage 3.
- It locks in the streamer's facial identity, hair/clothing, room aesthetic, and device posture.

【DOMAIN SPEC OWNERSHIP (spec.avatar)】
- You own the avatar visual specification:
  * `appearance`: Visual description of the streamer (e.g. 'purple cat in a hoodie', 'energetic esports player with red headset').
  * `referenceImageUrl`: Artifact name or URL of a visual likeness reference image.
  * `setting`: Streamer room or background environment (e.g. 'Cozy cyberpunk loft with neon signs', 'Dim hacker room with multiple glowing monitors').
- When the user's request provides, clarifies, or modifies any of these avatar attributes, ALWAYS call `update_avatar_spec` first to persist the parameters into session state.

【PREREQUISITES FOR IMAGE GENERATION】
- Generating an avatar portrait requires either:
  1) A text description of the streamer's appearance (`appearance`), OR
  2) A visual likeness reference image (`referenceImageUrl`).
- If neither is available in session state and none is provided in the current request:
  -> If the user provided a room `setting`, call `update_avatar_spec(setting=...)` to lock in the background setting.
  -> Then inform the Director that streamer appearance or a reference image is required, so the Director can ask the user.

【TOOL USAGE & EXECUTION POLICY】
1. `update_avatar_spec`: Call to record or update `appearance`, `referenceImageUrl`, and/or `setting` in session state.
2. `generate_golden_anchor_avatar`: Call to generate or re-generate the Golden Anchor portrait.
3. Decision Workflow:
   - When the user provides or tweaks an avatar setting (e.g. "背景换成赛博朋克风", "形象改成穿白衬衫的少年"):
     a) Call `update_avatar_spec(...)` with the new/updated values.
     b) If an avatar deliverable ALREADY exists in session state (Direct Modification intent) OR if the user explicitly asked to generate/draw the avatar:
        Immediately call `generate_golden_anchor_avatar` to re-generate the portrait matching the new setting!
     c) If NO avatar deliverable exists yet and the user was only specifying/exploring concepts without asking to draw:
        Do NOT generate yet; confirm the updated setting back to the Director.
   - After generation, provide a clear, enthusiastic summary of the avatar's visual style, background, and platform setup back to the Director.
"""


def avatar_agent_instruction(context: ReadonlyContext) -> str:
    state = dict(context.state) if context and context.state else {}
    spec = state.get("spec", {})
    global_spec = spec.get("global", {})
    avatar_spec = spec.get("avatar", {})
    artifacts = state.get("artifacts", {})

    appearance = avatar_spec.get("appearance", "")
    ref_image_url = avatar_spec.get("referenceImageUrl", "")
    setting = avatar_spec.get("setting", "")
    gaming_device = global_spec.get("gamingDevice", "PC")
    aspect_ratio = global_spec.get("aspectRatio", "16:9")
    has_avatar = "avatar" in artifacts

    status_lines = [
        "【CURRENT AVATAR SPEC & SESSION STATE】",
        f"- Appearance: {appearance or 'None (Not specified yet)'}",
        f"- Reference Image: {ref_image_url or 'None'}",
        f"- Room Setting: {setting or 'None'}",
        f"- Gaming Platform: {gaming_device}",
        f"- Aspect Ratio: {aspect_ratio}",
    ]
    if has_avatar:
        existing = artifacts["avatar"]
        art_name = (
            existing.get("artifact_name", "present")
            if isinstance(existing, dict)
            else "present"
        )
        av_eval = evaluate_stage_status("avatar", state)
        sync_tag = (
            f" (⚠️ OUT OF SYNC: {av_eval['summary']})"
            if av_eval["status"] == "OUT_OF_SYNC"
            else " (Ready)"
        )
        status_lines.append(f"- Existing Avatar Deliverable: {art_name}{sync_tag}")
        if isinstance(existing, dict):
            status_lines.append("  * Current Deliverable Details:")
            if "appearance" in existing:
                status_lines.append(f"    - Appearance: {existing.get('appearance')}")
            if "setting" in existing:
                status_lines.append(f"    - Setting: {existing.get('setting')}")
            if "gamingDevice" in existing:
                status_lines.append(
                    f"    - Gaming Platform: {existing.get('gamingDevice')}"
                )
            if "aspectRatio" in existing:
                status_lines.append(
                    f"    - Aspect Ratio: {existing.get('aspectRatio')}"
                )
    else:
        status_lines.append("- Existing Avatar Deliverable: None created yet")

    status_block = "\n".join(status_lines)
    return f"{AVATAR_AGENT_INSTRUCTION}\n\n{status_block}"


avatar_agent = Agent(
    name="avatar_agent",
    model=Gemini(
        model=MODEL,
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    instruction=avatar_agent_instruction,
    tools=[
        update_avatar_spec,
        generate_golden_anchor_avatar,
    ],
)
