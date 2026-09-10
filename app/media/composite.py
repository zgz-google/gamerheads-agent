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

"""FFmpeg compositing utility: overlays streamer reaction video over gameplay footage."""

from __future__ import annotations

import asyncio
import os
import subprocess

from app.media.clips import get_ffmpeg_exe, has_audio_track


def get_video_duration(file_path: str) -> float:
    """Gets duration in seconds of a video file using ffmpeg."""
    ffmpeg = get_ffmpeg_exe()
    cmd = [ffmpeg, "-i", file_path]
    res = subprocess.run(cmd, capture_output=True, text=True)
    for line in res.stderr.split("\n"):
        if "Duration:" in line:
            parts = line.split("Duration:")[1].split(",")[0].strip()
            # Format: HH:MM:SS.ms
            h, m, s = parts.split(":")
            return float(h) * 3600 + float(m) * 60 + float(s)
    return 0.0


async def composite_streamer_over_gameplay(
    streamer_path: str,
    gameplay_path: str | None,
    output_path: str,
    aspect_ratio: str = "16:9",
    layout: str = "pip",
    pip_placement: str = "bottom-right",
    stacked_placement: str | None = None,
    gameplay_volume: float = 0.8,
    streamer_volume: float = 1.0,
    subtitles_ass_path: str | None = None,
) -> dict:
    """Composites streamer video over gameplay footage using FFmpeg.

    Args:
        streamer_path: Path to rendered streamer video.
        gameplay_path: Path to user gameplay video. Ignored if layout is 'streamer-only'.
        output_path: Target output video path.
        aspect_ratio: '16:9' or '9:16'.
        layout: 'pip', 'stacked', or 'streamer-only'.
        pip_placement: 'bottom-right', 'bottom-left', 'top-right', or 'top-left'.
        stacked_placement: 'top', 'bottom', 'left', 'right' (defaults based on aspect ratio).
        gameplay_volume: Multiplier for gameplay audio (e.g. 0.8).
        streamer_volume: Multiplier for streamer audio (e.g. 1.0).
        subtitles_ass_path: Optional path to .ass subtitle file to burn into video.

    Returns:
        Dict with duration, output path, and drift metadata.
    """
    ffmpeg = get_ffmpeg_exe()
    is_vertical = aspect_ratio == "9:16"
    width = 1080 if is_vertical else 1920
    height = 1920 if is_vertical else 1080

    streamer_dur = get_video_duration(streamer_path)

    # 1. streamer-only layout: only process streamer video
    if layout == "streamer-only" or not gameplay_path:
        vf_filters = [f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height}"]
        if subtitles_ass_path and os.path.exists(subtitles_ass_path):
            safe_ass = subtitles_ass_path.replace("\\", "/").replace(":", "\\:")
            vf_filters.append(f"ass='{safe_ass}'")

        cmd = [
            ffmpeg,
            "-i", streamer_path,
            "-vf", ",".join(vf_filters),
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-pix_fmt", "yuv420p",
            "-af", f"volume={streamer_volume}",
            "-c:a", "aac",
            output_path,
            "-y",
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"FFmpeg streamer-only composite failed: {stderr.decode('utf-8', errors='ignore')}")

        return {"outputPath": output_path, "durationSeconds": streamer_dur, "driftSeconds": 0.0}

    # 2. PIP or Stacked layout
    gameplay_dur = get_video_duration(gameplay_path)
    drift = streamer_dur - gameplay_dur

    streamer_has_audio = has_audio_track(streamer_path)
    gameplay_has_audio = has_audio_track(gameplay_path)

    # Filter graph construction
    filter_complex: list[str] = []

    if layout == "stacked":
        # Stacked layout
        st_place = stacked_placement or ("top" if is_vertical else "left")
        if is_vertical:
            # 1080x1920 split vertically: each 1080x960
            half_h = height // 2
            filter_complex.append(f"[1:v]scale={width}:{half_h}:force_original_aspect_ratio=increase,crop={width}:{half_h}[g_crop]")
            filter_complex.append(f"[0:v]scale={width}:{half_h}:force_original_aspect_ratio=increase,crop={width}:{half_h}[s_crop]")
            if st_place == "top":
                # Streamer top, gameplay bottom
                filter_complex.append("[s_crop][g_crop]vstack=inputs=2[v_base]")
            else:
                # Gameplay top, streamer bottom
                filter_complex.append("[g_crop][s_crop]vstack=inputs=2[v_base]")
        else:
            # 1920x1080 split horizontally: each 960x1080
            half_w = width // 2
            filter_complex.append(f"[1:v]scale={half_w}:{height}:force_original_aspect_ratio=increase,crop={half_w}:{height}[g_crop]")
            filter_complex.append(f"[0:v]scale={half_w}:{height}:force_original_aspect_ratio=increase,crop={half_w}:{height}[s_crop]")
            if st_place == "left":
                # Streamer left, gameplay right
                filter_complex.append("[s_crop][g_crop]hstack=inputs=2[v_base]")
            else:
                # Gameplay left, streamer right
                filter_complex.append("[g_crop][s_crop]hstack=inputs=2[v_base]")
    else:
        # PIP layout
        filter_complex.append(f"[1:v]scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height}[bg]")
        # Overlay size: ~30% of canvas width
        pip_w = int(width * 0.30)
        pip_margin = 32
        filter_complex.append(f"[0:v]scale={pip_w}:-2[pip_scaled]")

        if pip_placement == "bottom-left":
            overlay_coords = f"x={pip_margin}:y=main_h-overlay_h-{pip_margin}"
        elif pip_placement == "top-right":
            overlay_coords = f"x=main_w-overlay_w-{pip_margin}:y={pip_margin}"
        elif pip_placement == "top-left":
            overlay_coords = f"x={pip_margin}:y={pip_margin}"
        else:  # default to bottom-right
            overlay_coords = f"x=main_w-overlay_w-{pip_margin}:y=main_h-overlay_h-{pip_margin}"

        filter_complex.append(f"[bg][pip_scaled]overlay={overlay_coords}[v_base]")

    # Subtitles
    if subtitles_ass_path and os.path.exists(subtitles_ass_path):
        safe_ass = subtitles_ass_path.replace("\\", "/").replace(":", "\\:")
        filter_complex.append(f"[v_base]ass='{safe_ass}'[vout]")
        map_v = "[vout]"
    else:
        map_v = "[v_base]"

    # Audio mixing
    audio_inputs_count = 0
    audio_parts = []
    if streamer_has_audio:
        filter_complex.append(f"[0:a]volume={streamer_volume}[a_streamer]")
        audio_parts.append("[a_streamer]")
        audio_inputs_count += 1
    if gameplay_has_audio:
        filter_complex.append(f"[1:a]volume={gameplay_volume}[a_gameplay]")
        audio_parts.append("[a_gameplay]")
        audio_inputs_count += 1

    if audio_inputs_count == 2:
        filter_complex.append(f"{''.join(audio_parts)}amix=inputs=2:duration=first:dropout_transition=2[aout]")
        map_audio = ["[aout]"]
    elif audio_inputs_count == 1:
        filter_complex.append(f"{audio_parts[0]}acopy[aout]")
        map_audio = ["[aout]"]
    else:
        # Generate silent audio if neither has audio
        map_audio = []

    cmd = [
        ffmpeg,
        "-i", streamer_path,  # input 0
        "-i", gameplay_path,  # input 1
        "-filter_complex", ";".join(filter_complex),
        "-map", map_v,
    ]
    if map_audio:
        cmd.extend(["-map", map_audio[0], "-c:a", "aac"])
    else:
        cmd.extend(["-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo", "-c:a", "aac", "-shortest"])

    cmd.extend([
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-pix_fmt", "yuv420p",
        "-t", str(streamer_dur),  # End exactly when streamer finishes
        output_path,
        "-y",
    ])

    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg composite failed: {stderr.decode('utf-8', errors='ignore')}")

    return {
        "outputPath": output_path,
        "durationSeconds": streamer_dur,
        "driftSeconds": round(drift, 1),
    }
