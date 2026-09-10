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

"""VideoAgent: Specialist agent for video generation (Stage 3) and composite post-production (Stage 4)."""

from __future__ import annotations

import base64
import os
import shutil
import tempfile
import uuid

from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.models import Gemini
from google.adk.tools import ToolContext
from google.genai import types

from app.media.clips import extract_last_frame, normalize_clip
from app.media.composite import composite_streamer_over_gameplay
from app.media.omni import omni_interaction
from app.media.stitch import concat_clips
from app.media.subtitles import build_ass_from_segments
from app.pipeline import evaluate_stage_status, record_artifact
from app.tools.spec_tools import update_composite_spec

load_dotenv()

MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest")


# ============================================================================
# 1. Prompt Engineering for Omni Flash (Stage 3)
# ============================================================================


def build_omni_prompt(
    visual_prompt: str,
    dialogue: str,
    duration_seconds: int,
    gaming_device: str = "PC",
) -> str:
    """Constructs the disciplined Omni generation prompt for image-to-video.

    <Image0> is the starting frame (the Golden Anchor avatar on segment 0,
    the previous clip's last frame on every segment thereafter).
    """
    device_instruction = "Streamer is seated at a gaming setup."
    gaze_instruction = "Streamer looks naturally at the screen / desk."

    if gaming_device == "Hands-free (No device)":
        device_instruction = "Streamer is completely hands-free with empty hands."
        gaze_instruction = "Streamer looks directly into the camera lens with engaging eye contact."
    elif gaming_device == "PC":
        device_instruction = "Streamer plays on PC with keyboard and mouse on desk."
    elif gaming_device == "Console":
        device_instruction = "Streamer holds a gamepad controller in hands."
    elif gaming_device == "Mobile (Vertical)":
        device_instruction = "Streamer holds a smartphone vertically in portrait mode."
    elif gaming_device == "Mobile (Horizontal)":
        device_instruction = "Streamer holds a smartphone horizontally in landscape mode."

    clean_dialogue = " ".join(dialogue.split()).strip() if dialogue else ""
    dialogue_line = (
        f'Streamer dialogue: "{clean_dialogue}". Natural speaking motion and lip synchronization.'
        if clean_dialogue
        else "Streamer remains silent."
    )

    return (
        f"<Image0> is the starting frame. In a single continuous shot with static camera:\n"
        f"The livestreamer performs: {visual_prompt}\n"
        f"{device_instruction} {gaze_instruction}\n"
        f"{dialogue_line}\n"
        f"Duration: {duration_seconds} seconds.\n"
        f"Audio guidelines: Streamer spoken voice and vocal reactions only. "
        f"No background music. No game sound effects. No scene cuts. No camera movement."
    ).strip()


# ============================================================================
# 2. Specialist Tools for VideoAgent
# ============================================================================


async def _extract_artifact_bytes(tool_context: ToolContext, artifact_name_or_url: str) -> bytes | None:
    """Helper to retrieve raw bytes from an ADK artifact or local file."""
    try:
        part = await tool_context.load_artifact(artifact_name_or_url)
        if part and part.inline_data and part.inline_data.data:
            raw_data = part.inline_data.data
            if isinstance(raw_data, str):
                clean_str = raw_data.strip().rstrip("=")
                padded = clean_str + "=" * ((4 - (len(clean_str) % 4)) % 4)
                return base64.b64decode(padded)
            return raw_data
    except Exception:
        pass

    if os.path.exists(artifact_name_or_url):
        with open(artifact_name_or_url, "rb") as f:
            return f.read()

    return None


async def generate_streamer_video(tool_context: ToolContext) -> str:
    """Renders the Stage 3 continuous streamer reaction video.

    Executes the serial Continuity Chain across all script segments:
    - Segment 0 starts from the Golden Anchor avatar portrait.
    - Segment N starts from Segment N-1's last frame.
    Stitches all clips into a single video deliverable stored in state['artifacts']['streamer_video'].

    Call this tool when:
    - Stage 1 script and Stage 2 avatar are both approved.
    - The user wants to generate the streamer's reaction video.

    Returns:
        Confirmation string with artifact name, duration, and segment count.
    """
    state = tool_context.state
    artifacts = state.get("artifacts", {})
    spec = state.get("spec", {})
    global_spec = spec.get("global", {})

    # 1. Prerequisite Checks
    eval_status = evaluate_stage_status("streamer_video", state)
    if eval_status["status"] == "BLOCKED":
        return f"Cannot generate streamer video: {eval_status['summary']}"
    if eval_status["status"] == "OUT_OF_SYNC":
        return f"Cannot generate streamer video: Upstream assets are out of date ({eval_status['summary']}). Please regenerate script or avatar first."

    script_data = artifacts.get("script")
    avatar_data = artifacts.get("avatar")
    if not script_data or not avatar_data:
        return "Cannot generate streamer video: Missing script or avatar deliverables."

    segments = script_data.get("segments", [])
    if not segments:
        return "Cannot generate streamer video: Script contains no commentary segments."

    avatar_artifact_name = avatar_data.get("artifact_name")
    if not avatar_artifact_name:
        return "Cannot generate streamer video: Avatar artifact name missing in state."

    # 2. Retrieve Avatar Image Bytes
    avatar_bytes = await _extract_artifact_bytes(tool_context, avatar_artifact_name)
    if not avatar_bytes:
        return f"Cannot generate streamer video: Failed to load avatar image data ({avatar_artifact_name})."

    avatar_b64 = base64.b64encode(avatar_bytes).decode("utf-8")
    avatar_mime = avatar_data.get("mimeType", "image/png")
    golden_anchor_data_url = f"data:{avatar_mime};base64,{avatar_b64}"

    aspect_ratio = global_spec.get("aspectRatio", "16:9")
    gaming_device = global_spec.get("gamingDevice", "PC")

    work_dir = tempfile.mkdtemp(prefix="streamer_render_")
    rendered_clips: list[str] = []

    try:
        prev_pose_b64: str | None = None

        # 3. Continuity Chain Render Loop
        for index, seg in enumerate(segments):
            dur = int(seg.get("duration", 5))
            prompt_text = build_omni_prompt(
                visual_prompt=seg.get("prompt", "Streamer looks focused and reacts naturally."),
                dialogue=seg.get("dialogue", ""),
                duration_seconds=dur,
                gaming_device=gaming_device,
            )

            raw_path = os.path.join(work_dir, f"raw_{index}.mp4")
            norm_path = os.path.join(work_dir, f"clip_{index}.mp4")

            start_frame = prev_pose_b64 if (prev_pose_b64 and index > 0) else golden_anchor_data_url

            # Invoke Omni Flash (or synthetic mock in test mode)
            await omni_interaction(
                prompt=prompt_text,
                start_frame_base64=start_frame,
                duration_seconds=dur,
                aspect_ratio=aspect_ratio,
                dest_path=raw_path,
            )

            # Normalize frame rate, pixel format, and audio
            await normalize_clip(raw_path, norm_path, fps=30)
            rendered_clips.append(norm_path)

            # Extract last frame for continuity into next segment
            if index < len(segments) - 1:
                try:
                    last_frame_path = os.path.join(work_dir, f"last_frame_{index}.jpg")
                    await extract_last_frame(norm_path, last_frame_path)
                    with open(last_frame_path, "rb") as lf_file:
                        prev_pose_b64 = f"data:image/jpeg;base64,{base64.b64encode(lf_file.read()).decode('utf-8')}"
                except Exception:
                    # Fallback to golden anchor avatar if frame extraction fails
                    prev_pose_b64 = golden_anchor_data_url

        # 4. Concatenate All Clips
        concat_output_path = os.path.join(work_dir, "streamer_stitched.mp4")
        await concat_clips(rendered_clips, concat_output_path)

        # 5. Save Artifact into ADK Artifact Service
        with open(concat_output_path, "rb") as f:
            final_bytes = f.read()

        artifact_name = f"streamer_video_{uuid.uuid4().hex[:8]}.mp4"
        part = types.Part(
            inline_data=types.Blob(
                mime_type="video/mp4",
                data=final_bytes,
            )
        )
        await tool_context.save_artifact(artifact_name, part)

        total_dur = sum(int(s.get("duration", 5)) for s in segments)

        # 6. Record in state['artifacts']['streamer_video']
        record_artifact(
            state,
            "streamer_video",
            {
                "artifact_name": artifact_name,
                "durationSeconds": total_dur,
                "segmentCount": len(segments),
                "aspectRatio": aspect_ratio,
            },
        )

        return (
            f"Successfully generated continuous streamer reaction video!\n"
            f"- Artifact Name: {artifact_name}\n"
            f"- Total Duration: {total_dur}s\n"
            f"- Segments Rendered: {len(segments)}\n"
            f"- Aspect Ratio: {aspect_ratio}\n"
            f"Streamer track is ready for final composite over gameplay footage."
        )

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


async def generate_composite_video(tool_context: ToolContext) -> str:
    """Renders the Stage 4 final composite video over gameplay footage via FFmpeg.

    Fast (~3 seconds). Decoupled from Stage 3: adjusting PIP placement, layouts,
    audio volumes, or subtitles only calls this tool without re-rendering streamer clips.

    Call this tool when:
    - Stage 3 streamer_video deliverable is ready.
    - The user wants the final composite reaction video.
    - The user requests layout changes (e.g. PIP bottom-left), volume tweaks, or subtitle toggles.

    Returns:
        Confirmation string with artifact name, layout, and duration.
    """
    state = tool_context.state
    artifacts = state.get("artifacts", {})
    spec = state.get("spec", {})
    global_spec = spec.get("global", {})
    comp_spec = spec.get("composite", {})

    streamer_artifact_data = artifacts.get("streamer_video")
    if not streamer_artifact_data:
        return "Cannot generate composite: Missing Stage 3 streamer_video artifact. Run generate_streamer_video first."

    streamer_artifact_name = streamer_artifact_data.get("artifact_name")
    streamer_bytes = await _extract_artifact_bytes(tool_context, streamer_artifact_name)
    if not streamer_bytes:
        return f"Cannot generate composite: Failed to load streamer video data ({streamer_artifact_name})."

    layout = comp_spec.get("layout", "pip")
    footage_url = global_spec.get("footageUrl")

    if layout != "streamer-only" and not footage_url:
        return "Cannot generate composite: Missing gameplay footage. Ask user to upload or provide gameplay video URL."

    gameplay_bytes: bytes | None = None
    if layout != "streamer-only" and footage_url:
        gameplay_bytes = await _extract_artifact_bytes(tool_context, footage_url)
        if not gameplay_bytes:
            # Check if it is a local file path
            if os.path.exists(footage_url):
                with open(footage_url, "rb") as f:
                    gameplay_bytes = f.read()

    work_dir = tempfile.mkdtemp(prefix="composite_render_")
    try:
        streamer_file = os.path.join(work_dir, "streamer.mp4")
        with open(streamer_file, "wb") as f:
            f.write(streamer_bytes)

        gameplay_file: str | None = None
        if gameplay_bytes:
            gameplay_file = os.path.join(work_dir, "gameplay.mp4")
            with open(gameplay_file, "wb") as f:
                f.write(gameplay_bytes)

        aspect_ratio = global_spec.get("aspectRatio", "16:9")
        pip_placement = comp_spec.get("pipPlacement", "bottom-right")
        stacked_placement = comp_spec.get("stackedPlacement")
        gameplay_volume = float(comp_spec.get("gameplayVolume", 0.8))
        streamer_volume = float(comp_spec.get("streamerVolume", 1.0))
        subtitles_enabled = bool(comp_spec.get("subtitles", True))

        # Build Subtitles if enabled
        subtitles_ass_path: str | None = None
        script_data = artifacts.get("script")
        if subtitles_enabled and script_data and script_data.get("segments"):
            ass_content = build_ass_from_segments(script_data["segments"], aspect_ratio=aspect_ratio)
            subtitles_ass_path = os.path.join(work_dir, "subtitles.ass")
            with open(subtitles_ass_path, "w", encoding="utf-8") as f:
                f.write(ass_content)

        output_composite_path = os.path.join(work_dir, "final_composite.mp4")

        result = await composite_streamer_over_gameplay(
            streamer_path=streamer_file,
            gameplay_path=gameplay_file,
            output_path=output_composite_path,
            aspect_ratio=aspect_ratio,
            layout=layout,
            pip_placement=pip_placement,
            stacked_placement=stacked_placement,
            gameplay_volume=gameplay_volume,
            streamer_volume=streamer_volume,
            subtitles_ass_path=subtitles_ass_path,
        )

        with open(output_composite_path, "rb") as f:
            final_bytes = f.read()

        composite_artifact_name = f"final_video_{uuid.uuid4().hex[:8]}.mp4"
        part = types.Part(
            inline_data=types.Blob(
                mime_type="video/mp4",
                data=final_bytes,
            )
        )
        await tool_context.save_artifact(composite_artifact_name, part)

        record_artifact(
            state,
            "composite",
            {
                "artifact_name": composite_artifact_name,
                "layout": layout,
                "pipPlacement": pip_placement,
                "gameplayVolume": gameplay_volume,
                "streamerVolume": streamer_volume,
                "subtitles": subtitles_enabled,
                "durationSeconds": result["durationSeconds"],
            },
        )

        notes = []
        if result.get("driftSeconds") and abs(result["driftSeconds"]) >= 1:
            notes.append(f"- Timing Notice: Gameplay drift of {result['driftSeconds']}s was synchronized to streamer commentary.")

        return (
            f"Successfully generated final composite reaction video!\n"
            f"- Artifact Name: {composite_artifact_name}\n"
            f"- Layout: {layout} (PIP placement: {pip_placement})\n"
            f"- Aspect Ratio: {aspect_ratio}\n"
            f"- Audio Mix: Gameplay {gameplay_volume}x, Streamer {streamer_volume}x\n"
            f"- Subtitles: {'Enabled' if subtitles_enabled else 'Disabled'}\n"
            f"- Duration: {result['durationSeconds']}s\n"
            + ("\n".join(notes) if notes else "")
        )

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


# ============================================================================
# 3. VideoAgent Definition & Dynamic Instruction
# ============================================================================

VIDEO_AGENT_INSTRUCTION = """You are the expert Post-Production Video Director (VideoAgent).
You manage the video production and composite phases of GamerHeads:
1. Stage 3 (Streamer Video): Rendering continuous streamer reaction clips with Omni Flash.
2. Stage 4 (Composite Video): Fast FFmpeg compositing (PIP/stacked overlay, audio mixing, subtitles).

【DECISION PRINCIPLES】
- You own post-production composite settings (spec.composite): layout, pipPlacement, stackedPlacement, gameplayVolume, streamerVolume, subtitles.
- If the user provides or modifies any composite settings (e.g. 'put PIP in bottom-left', 'make gameplay louder', 'turn off subtitles'), ALWAYS call `update_composite_spec` first.
- If streamer_video is missing or needs generation, run `generate_streamer_video` first, then run `generate_composite_video`.
- If streamer_video is already ready and the user only changes layout, volume, or subtitles, NEVER re-render Stage 3. Call `generate_composite_video` directly (takes ~3s).
"""


def video_agent_instruction(context: ReadonlyContext) -> str:
    state = dict(context.state) if context and context.state else {}
    spec = state.get("spec", {})
    comp_spec = spec.get("composite", {})

    lines = [
        VIDEO_AGENT_INSTRUCTION,
        "\n【CURRENT COMPOSITE SPEC (OWNED BY VIDEO_AGENT)】",
        f"- Layout: {comp_spec.get('layout', 'pip')}",
        f"- PIP Placement: {comp_spec.get('pipPlacement', 'bottom-right')}",
        f"- Gameplay Volume: {comp_spec.get('gameplayVolume', 0.8)}",
        f"- Streamer Volume: {comp_spec.get('streamerVolume', 1.0)}",
        f"- Subtitles: {comp_spec.get('subtitles', True)}",
    ]
    return "\n".join(lines)


video_agent = Agent(
    name="video_agent",
    model=Gemini(model=MODEL),
    instruction=video_agent_instruction,
    tools=[
        update_composite_spec,
        generate_streamer_video,
        generate_composite_video,
    ],
)
