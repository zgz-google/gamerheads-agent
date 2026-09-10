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

"""FFmpeg concatenation utility for stitching multiple normalized video clips."""

from __future__ import annotations

import asyncio

from app.media.clips import (
    CANONICAL,
    get_ffmpeg_exe,
    normalize_clip,
    probe_has_audio,
    probe_video_dimensions,
    probe_video_duration,
)


async def concat_clips(clip_paths: list[str], output_path: str) -> str:
    """Concatenates multiple normalized MP4 clips in order into a single video file.

    Uses filter_complex concat re-encoding (concat=n=N:v=1:a=1) rather than stream copy.
    A stream copy splices AAC tracks without touching their encoder priming, causing
    accumulated A/V drift and lip-sync desynchronization across multiple clips.

    Args:
        clip_paths: List of absolute paths to normalized MP4 video files.
        output_path: Destination path for the stitched video.

    Returns:
        The output_path.
    """
    if not clip_paths:
        raise ValueError("Cannot concatenate empty list of clips.")

    if len(clip_paths) == 1:
        return await normalize_clip(clip_paths[0], output_path)

    ffmpeg = get_ffmpeg_exe()
    inputs: list[str] = []
    pre: list[str] = []
    segments: list[str] = []

    # The first clip sets the geometry; everything else is fitted into it
    target = probe_video_dimensions(clip_paths[0])

    for i, clip in enumerate(clip_paths):
        inputs.extend(["-i", clip])

        video_label = f"[{i}:v]"
        dimensions = target if i == 0 else probe_video_dimensions(clip)
        if (
            target
            and dimensions
            and (dimensions[0] != target[0] or dimensions[1] != target[1])
        ):
            w, h = target
            pre.append(
                f"[{i}:v]scale={w}:{h}:force_original_aspect_ratio=decrease,"
                f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:black,setsar=1[v{i}]"
            )
            video_label = f"[v{i}]"

        if probe_has_audio(clip):
            segments.append(f"{video_label}[{i}:a]")
        else:
            duration = probe_video_duration(clip) or 0.0
            dur_str = f"{max(duration, 0.05):.3f}"
            pre.append(
                f"anullsrc=channel_layout=stereo:sample_rate={CANONICAL['sample_rate']}[si{i}]"
            )
            pre.append(f"[si{i}]atrim=duration={dur_str},asetpts=PTS-STARTPTS[a{i}]")
            segments.append(f"{video_label}[a{i}]")

    filter_complex = (
        (f"{';'.join(pre)};" if pre else "")
        + "".join(segments)
        + f"concat=n={len(clip_paths)}:v=1:a=1[v][a]"
    )

    cmd = [
        ffmpeg,
        "-y",
        *inputs,
        "-filter_complex",
        filter_complex,
        "-map",
        "[v]",
        "-map",
        "[a]",
        "-c:v",
        CANONICAL["video_codec"],
        "-preset",
        CANONICAL["preset"],
        "-crf",
        CANONICAL["crf"],
        "-pix_fmt",
        CANONICAL["pixel_format"],
        "-c:a",
        CANONICAL["audio_codec"],
        "-b:a",
        CANONICAL["audio_bitrate"],
        "-ar",
        CANONICAL["sample_rate"],
        "-ac",
        CANONICAL["channels"],
        "-movflags",
        "+faststart",
        output_path,
    ]

    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(
            f"Failed to concatenate clips: {stderr.decode('utf-8', errors='ignore')}"
        )

    return output_path
