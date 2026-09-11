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
import contextlib
import logging
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

from app.media.clips import (
    extract_last_frame_data_url,
    normalize_clip,
    probe_duration,
)
from app.media.composite import composite_streamer_over_gameplay
from app.media.omni import omni_interaction
from app.media.stitch import concat_clips
from app.media.subtitles import build_ass_from_segments
from app.pipeline import evaluate_stage_status, record_artifact
from app.tools.spec_tools import update_composite_spec

load_dotenv()

logger = logging.getLogger(__name__)

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
        gaze_instruction = (
            "Streamer looks directly into the camera lens with engaging eye contact."
        )
    elif gaming_device == "PC":
        device_instruction = "Streamer plays on PC with keyboard and mouse on desk."
    elif gaming_device == "Console":
        device_instruction = "Streamer holds a gamepad controller in hands."
    elif gaming_device == "Mobile (Vertical)":
        device_instruction = "Streamer holds a smartphone vertically in portrait mode."
    elif gaming_device == "Mobile (Horizontal)":
        device_instruction = (
            "Streamer holds a smartphone horizontally in landscape mode."
        )

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


async def _extract_artifact_bytes(
    tool_context: ToolContext, artifact_name_or_url: str
) -> bytes | None:
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
    """Renders continuous streamer reaction video with lip-sync and continuity.

    Executes the serial Continuity Chain across all script segments:
    - Segment 0 starts from the Golden Anchor avatar portrait.
    - Segment N starts from Segment N-1's last frame.
    Stitches all clips into a single video deliverable stored in state['artifacts']['streamer_video'].

    Call this tool when:
    - Commentary script and streamer avatar are both ready.
    - The user wants to generate the streamer's reaction video.

    Returns:
        Confirmation string with duration and segment count.
    """
    state = tool_context.state
    artifacts = state.get("artifacts", {})
    spec = state.get("spec", {})
    global_spec = spec.get("global", {})

    # 1. Prerequisite Checks
    eval_status = evaluate_stage_status("streamer_video", state)
    if eval_status["status"] == "BLOCKED":
        return f"Cannot generate streamer video: {eval_status['summary']}"

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
    clip_artifacts: list[dict[str, Any]] = []

    try:
        logger.info(
            "[streamer] rendering %d segments at %s", len(segments), aspect_ratio
        )
        prev_pose_b64: str | None = None

        # 3. Continuity Chain Render Loop
        for index, seg in enumerate(segments):
            dur = int(seg.get("duration", 5))
            dialogue = seg.get("dialogue", "")
            logger.info(
                "[streamer] segment %d/%d (%ss): %s",
                index + 1,
                len(segments),
                dur,
                dialogue[:60],
            )

            prompt_text = build_omni_prompt(
                visual_prompt=seg.get(
                    "prompt", "Streamer looks focused and reacts naturally."
                ),
                dialogue=dialogue,
                duration_seconds=dur,
                gaming_device=gaming_device,
            )

            raw_path = os.path.join(work_dir, f"raw_{index}.mp4")
            norm_path = os.path.join(work_dir, f"clip_{index}.mp4")

            start_frame = prev_pose_b64 or golden_anchor_data_url

            # Invoke Omni Flash (or synthetic mock in test mode)
            try:
                await omni_interaction(
                    prompt=prompt_text,
                    start_frame_base64=start_frame,
                    duration_seconds=dur,
                    aspect_ratio=aspect_ratio,
                    dest_path=raw_path,
                    continuity=bool(prev_pose_b64),
                )
            except Exception as err:
                # This message is the only account of why a multi-minute render
                # died, and it otherwise reaches the model without ever reaching
                # the logs -- which makes the failure invisible to anyone reading them.
                logger.error(
                    "[streamer] segment %d/%d failed: %s", index + 1, len(segments), err
                )
                raise RuntimeError(
                    f"segment {index + 1} of {len(segments)} failed after retries ({err}). "
                    f"{len(rendered_clips)} segment(s) rendered before it."
                ) from err

            # Normalize frame rate, pixel format, and audio (canonical 24fps, PTS reset)
            await normalize_clip(raw_path, norm_path, fps=24)
            # Cloud Run's /tmp is RAM: the raw clip is dead weight once normalized,
            # and holding every one of them until the end doubles the peak.
            with contextlib.suppress(OSError):
                os.remove(raw_path)
            rendered_clips.append(norm_path)

            # Save individual normalized clip as an artifact
            with open(norm_path, "rb") as cf:
                clip_bytes = cf.read()

            seg_id = seg.get("id") if seg.get("id") is not None else (index + 1)
            clip_artifact_name = f"clip_{seg_id}_{uuid.uuid4().hex[:8]}.mp4"
            clip_part = types.Part(
                inline_data=types.Blob(
                    mime_type="video/mp4",
                    data=clip_bytes,
                )
            )
            await tool_context.save_artifact(clip_artifact_name, clip_part)
            clip_dur = probe_duration(norm_path)
            clip_artifacts.append(
                {
                    "index": index,
                    "segment_id": seg_id,
                    "artifact_name": clip_artifact_name,
                    "durationSeconds": round(clip_dur, 1) if clip_dur else dur,
                    "requestedSeconds": dur,
                    "dialogue": dialogue,
                }
            )

            # Extract last frame for continuity into next segment
            if index < len(segments) - 1:
                try:
                    prev_pose_b64 = await extract_last_frame_data_url(norm_path)
                except Exception as err:
                    # Not fatal: the next segment falls back to the golden anchor,
                    # which costs continuity on one cut rather than the whole run.
                    logger.warning(
                        "[streamer] could not read the last frame of segment %d; "
                        "falling back to the avatar (%s)",
                        index + 1,
                        err,
                    )
                    prev_pose_b64 = None

        # 4. Concatenate All Clips
        concat_output_path = os.path.join(work_dir, "streamer_stitched.mp4")
        await concat_clips(rendered_clips, concat_output_path)
        for clip in rendered_clips:
            with contextlib.suppress(OSError):
                os.remove(clip)

        # The script's durations are what we *asked* for. Nothing in the
        # Interactions request carries a length -- "Duration: 5 seconds." is a
        # sentence in the prompt -- so the clips come back as long as the model
        # felt like, and the sum of the script is a guess. The composite runs to
        # the real length, so report the real length.
        requested_dur = sum(int(s.get("duration", 5)) for s in segments)
        actual_dur = probe_duration(concat_output_path)
        total_dur = round(actual_dur, 1) if actual_dur else requested_dur
        if actual_dur and abs(actual_dur - requested_dur) >= 1:
            logger.warning(
                "[streamer] rendered %.1fs against a %ds script -- the clip "
                "generator does not honour the requested per-segment length",
                actual_dur,
                requested_dur,
            )

        # 5. Save Artifact into ADK Artifact Service
        with open(concat_output_path, "rb") as f:
            final_bytes = f.read()

        artifact_name = f"output_streamer_video_{uuid.uuid4().hex[:8]}.mp4"
        part = types.Part(
            inline_data=types.Blob(
                mime_type="video/mp4",
                data=final_bytes,
            )
        )
        await tool_context.save_artifact(artifact_name, part)

        # 6. Record in state['artifacts']['streamer_video']
        record_artifact(
            state,
            "streamer_video",
            {
                "artifact_name": artifact_name,
                "durationSeconds": total_dur,
                "requestedSeconds": requested_dur,
                "segmentCount": len(segments),
                "aspectRatio": aspect_ratio,
                "clips": clip_artifacts,
            },
        )

        return (
            f"Successfully generated continuous streamer reaction video!\n"
            f"- Total Duration: {total_dur}s\n"
            f"- Segments Rendered: {len(segments)}\n"
            f"- Aspect Ratio: {aspect_ratio}\n"
            f"- Clips Saved: {len(clip_artifacts)} individual clip artifacts preserved\n"
            f"Streamer reaction video is ready for final composite over gameplay footage."
        )

    except Exception as err:
        # A tool that raises reaches the Director as a stack trace it cannot act
        # on. Every other failure in this file is a sentence, so this one is too.
        logger.exception("[streamer] render failed")
        return f"Cannot generate streamer video: {err}"
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


async def generate_composite_video(tool_context: ToolContext) -> str:
    """Renders the final composite video over gameplay footage via FFmpeg.

    Fast (~3 seconds). Decoupled from reaction video rendering: adjusting PIP placement, layouts,
    audio volumes, or subtitles only calls this tool without re-rendering streamer clips.

    Call this tool when:
    - Streamer reaction video deliverable is ready.
    - The user wants the final composite reaction video.
    - The user requests layout changes (e.g. PIP bottom-left), volume tweaks, or subtitle toggles.

    Returns:
        Confirmation string with layout, audio mix, and duration.
    """
    state = tool_context.state
    artifacts = state.get("artifacts", {})
    spec = state.get("spec", {})
    global_spec = spec.get("global", {})
    comp_spec = spec.get("composite", {})

    streamer_artifact_data = artifacts.get("streamer_video")
    if not streamer_artifact_data:
        return "Cannot generate composite: Missing streamer reaction video. Run generate_streamer_video first."

    sv_status = evaluate_stage_status("streamer_video", state)
    if sv_status["status"] == "OUT_OF_SYNC":
        return f"Cannot generate composite: Streamer reaction video is out of sync with updated script or avatar ({sv_status['summary']}). Please re-render streamer reaction video via generate_streamer_video first."

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
            ass_content = build_ass_from_segments(
                script_data["segments"], aspect_ratio=aspect_ratio
            )
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

        composite_artifact_name = f"output_composite_{uuid.uuid4().hex[:8]}.mp4"
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
        drift = result.get("driftSeconds")
        if drift is not None and abs(drift) >= 1:
            if drift > 0:
                notes.append(
                    f"- The footage is {drift}s shorter than the streamer track, so its last frame is held until the streamer finishes."
                )
            else:
                notes.append(
                    f"- The footage is {abs(drift)}s longer than the streamer track, so its tail is not in the finished video."
                )

        if result.get("gameplayFit") == "contain":
            aspect = result.get("gameplayAspect") or 1.0
            shape = "vertical" if aspect < 1.0 else "wide"
            alt_aspect = "9:16" if aspect_ratio == "16:9" else "16:9"
            notes.append(
                f"- The footage is {shape} and this video is {aspect_ratio}, too different a shape to crop without throwing most of the picture away, so it is shown whole with black at the sides. Setting aspectRatio to {alt_aspect} would fill the frame instead."
            )

        return (
            f"Successfully generated final composite reaction video!\n"
            f"- Layout: {layout} (PIP placement: {pip_placement})\n"
            f"- Aspect Ratio: {aspect_ratio}\n"
            f"- Audio Mix: Gameplay {gameplay_volume}x, Streamer {streamer_volume}x\n"
            f"- Subtitles: {'Enabled' if subtitles_enabled else 'Disabled'}\n"
            f"- Duration: {result['durationSeconds']}s\n"
            + (("\n" + "\n".join(notes)) if notes else "")
        )

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


# ============================================================================
# 3. VideoAgent Definition & Dynamic Instruction
# ============================================================================

VIDEO_AGENT_INSTRUCTION = """You are the expert Post-Production Video Director & Compositor (VideoAgent) in the GamerHeads studio.
Your purpose is producing polished, lip-synced streamer reaction videos and seamless final composite reaction videos overlaid on gameplay footage.

【CORE PRODUCTION PRINCIPLES】

1. QUERY TRIAGE & SPEC-FIRST (Check update_composite_spec options BEFORE generating):
   - Whenever creator input arrives, FIRST check if the creator is asking to modify, configure, or provide any composite settings.
   - Check all available parameters in `update_composite_spec`:
     * `layout`: Video composition layout ('pip' for picture-in-picture, 'stacked' for split screen, 'streamer-only')
     * `pipPlacement`: Corner position for PIP streamer window ('bottom-right', 'bottom-left', 'top-right', 'top-left')
     * `stackedPlacement`: Alignment for stacked split screen ('top', 'bottom', 'left', 'right')
     * `gameplayVolume`: Gameplay background volume multiplier (e.g. 0.8)
     * `streamerVolume`: Streamer commentary voice volume multiplier (e.g. 1.0)
     * `subtitles`: Enable or disable burned-in subtitles (True/False)
   - If the request contains or modifies ANY of these spec options:
     -> You MUST call `update_composite_spec` FIRST to update session state before any video generation.

2. CONVERGENCE-FIRST & UPSTREAM PREREQUISITES (No script or avatar, no video):
   - Synthesizing streamer reaction video requires BOTH the commentary script (`script`) and the streamer avatar portrait (`avatar`).
   - If either is missing, NEVER attempt video generation. Record any provided composite settings via `update_composite_spec`, and inform the Director which upstream asset is missing (commentary script or avatar portrait) so the Director can coordinate with the creator.
   - For composite video generation with PIP or split-screen layouts, gameplay footage (`footageUrl`) is also required.

3. DECOUPLED POST-PRODUCTION (Protect expensive reaction footage synthesis):
   - Rendering streamer reaction video clips (image-to-video with lip synchronization and frame continuity) is compute-heavy (~2-3 minutes).
   - Compositing the final video (FFmpeg PIP overlay, audio mixing, subtitles) is fast and lightweight (~3 seconds).
   - When streamer reaction video is already rendered and the creator only adjusts composite settings (e.g. moving PIP corner, adjusting volume balance, toggling subtitles):
     -> Update settings via `update_composite_spec`.
     -> Call `generate_composite_video` directly.
     -> NEVER re-render `generate_streamer_video` for layout, volume, or subtitle adjustments!

4. DECOUPLED STAGING & DISCIPLINED EXECUTION (One video stage per request):
   - You handle two distinct production stages: Stage 3 (Streamer Reaction Video via `generate_streamer_video`) and Stage 4 (Composite Post-Production via `generate_composite_video`).
   - Execute ONLY the stage requested by the Director:
     * When asked to render or re-render streamer reaction video: call `generate_streamer_video`, report the generated reaction deliverable, and STOP. Do NOT automatically proceed to call `generate_composite_video` in the same turn—allow the Director and creator to review and confirm the reaction performance first.
     * When asked to composite or re-composite final video: update composite settings via `update_composite_spec` if requested, call `generate_composite_video`, and report the composite deliverable.

5. AUDIO & VISUAL HARMONY (Balanced mixing and readable framing):
   - Maintain clear audio balance: streamer commentary dialogue should always be crisp and prominent over gameplay sounds.
   - Ensure picture-in-picture streamer window placement respects gameplay visibility and does not obscure critical game UI or action.
   - When subtitles are enabled, ensure they are synchronized with commentary dialogue beats for high viewer engagement.

6. SPECIALIST DELIVERY (Concise upstream reporting to Director):
   You are an internal specialist reporting to the Director (main agent). Once all required production actions are complete and your job is finished, concisely synthesize the deliverable:
   - Provide the video duration, visual composition (layout style and placement), audio mix ratio, and subtitle status.
   - Share any notable gameplay video fit observations (e.g. vertical vs. horizontal framing fit, timing drift) so the Director can communicate them to the creator.
   - NEVER report artifact filenames, internal file paths, technical identifiers, or tool names to the Director. All media assets are already saved in session state.
"""


def video_agent_instruction(context: ReadonlyContext) -> str:
    state = context.state
    spec = state.get("spec", {})
    global_spec = spec.get("global", {})
    comp_spec = spec.get("composite", {})
    artifacts = state.get("artifacts", {})

    footage_url = global_spec.get("footageUrl", "")
    gaming_device = global_spec.get("gamingDevice", "PC")
    aspect_ratio = global_spec.get("aspectRatio", "16:9")

    layout = comp_spec.get("layout", "pip")
    pip_placement = comp_spec.get("pipPlacement", "bottom-right")
    stacked_placement = comp_spec.get("stackedPlacement")
    gameplay_volume = comp_spec.get("gameplayVolume", 0.8)
    streamer_volume = comp_spec.get("streamerVolume", 1.0)
    subtitles = comp_spec.get("subtitles", True)

    has_script = "script" in artifacts
    has_avatar = "avatar" in artifacts
    has_streamer = "streamer_video" in artifacts
    has_composite = "composite" in artifacts

    status_lines = [
        "【CURRENT PRODUCTION SPEC & SESSION STATE】",
        f"- Gameplay Footage: {footage_url or 'None (Not provided yet)'}",
        f"- Gaming Platform: {gaming_device}",
        f"- Video Aspect Ratio: {aspect_ratio}",
        f"- Video Layout: {layout}",
        f"- PIP Placement: {pip_placement}",
    ]
    if stacked_placement:
        status_lines.append(f"- Stacked Placement: {stacked_placement}")
    status_lines.extend(
        [
            f"- Gameplay Volume: {gameplay_volume}",
            f"- Streamer Volume: {streamer_volume}",
            f"- Subtitles: {'Enabled' if subtitles else 'Disabled'}",
        ]
    )

    # Upstream Prerequisites Status
    status_lines.append("【UPSTREAM PREREQUISITES STATUS】")
    if has_script:
        script_art = artifacts.get("script", {})
        seg_count = (
            len(script_art.get("segments", [])) if isinstance(script_art, dict) else 0
        )
        total_dur = (
            script_art.get("total_duration", 0) if isinstance(script_art, dict) else 0
        )
        sc_eval = evaluate_stage_status("script", state)
        sc_tag = (
            f" (⚠️ OUT OF SYNC: {sc_eval['summary']})"
            if sc_eval["status"] == "OUT_OF_SYNC"
            else " (Ready)"
        )
        status_lines.append(
            f"- Commentary Script: Present ({seg_count} segments, {total_dur}s){sc_tag}"
        )
    else:
        status_lines.append(
            "- Commentary Script: None drafted yet (Prerequisite for reaction video)"
        )

    if has_avatar:
        avatar_art = artifacts.get("avatar", {})
        av_eval = evaluate_stage_status("avatar", state)
        av_tag = (
            f" (⚠️ OUT OF SYNC: {av_eval['summary']})"
            if av_eval["status"] == "OUT_OF_SYNC"
            else " (Ready)"
        )
        av_desc = (
            avatar_art.get("appearance", "Persona established")
            if isinstance(avatar_art, dict)
            else "Persona established"
        )
        status_lines.append(f"- Streamer Avatar: Present ({av_desc}){av_tag}")
    else:
        status_lines.append(
            "- Streamer Avatar: None created yet (Prerequisite for reaction video)"
        )

    # Video Deliverables Status
    status_lines.append("【VIDEO DELIVERABLES STATUS】")
    sv_eval = evaluate_stage_status("streamer_video", state)
    if has_streamer:
        sv_art = artifacts.get("streamer_video", {})
        sv_tag = (
            f" (⚠️ OUT OF SYNC: {sv_eval['summary']})"
            if sv_eval["status"] == "OUT_OF_SYNC"
            else " (Ready)"
        )
        dur = sv_art.get("durationSeconds", 0) if isinstance(sv_art, dict) else 0
        segs = sv_art.get("segmentCount", 0) if isinstance(sv_art, dict) else 0
        status_lines.append(
            f"- Streamer Reaction Video: Present ({dur}s, {segs} clips){sv_tag}"
        )
    else:
        status_lines.append(
            f"- Streamer Reaction Video: Not generated yet ({sv_eval['summary']})"
        )

    comp_eval = evaluate_stage_status("composite", state)
    if has_composite:
        comp_art = artifacts.get("composite", {})
        comp_tag = (
            f" (⚠️ OUT OF SYNC: {comp_eval['summary']})"
            if comp_eval["status"] == "OUT_OF_SYNC"
            else " (Ready)"
        )
        dur = comp_art.get("durationSeconds", 0) if isinstance(comp_art, dict) else 0
        lay = comp_art.get("layout", "pip") if isinstance(comp_art, dict) else "pip"
        status_lines.append(
            f"- Final Composite Video: Present ({dur}s, layout: {lay}){comp_tag}"
        )
    else:
        status_lines.append(
            f"- Final Composite Video: Not generated yet ({comp_eval['summary']})"
        )

    status_block = "\n".join(status_lines)
    return f"{VIDEO_AGENT_INSTRUCTION}\n\n{status_block}"


video_agent = Agent(
    name="video_agent",
    model=Gemini(
        model=MODEL,
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    instruction=video_agent_instruction,
    tools=[
        update_composite_spec,
        generate_streamer_video,
        generate_composite_video,
    ],
)
