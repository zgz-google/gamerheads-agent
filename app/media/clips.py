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

"""FFmpeg video clip utilities: frame extraction, normalization, and compression."""

from __future__ import annotations

import asyncio
import os
import subprocess
import tempfile

import imageio_ffmpeg


def get_ffmpeg_exe() -> str:
    """Returns the path to the bundled ffmpeg executable from imageio-ffmpeg."""
    return imageio_ffmpeg.get_ffmpeg_exe()


def has_audio_track(video_path: str) -> bool:
    """Checks if the video file contains an audio stream."""
    ffmpeg = get_ffmpeg_exe()
    cmd = [ffmpeg, "-i", video_path]
    res = subprocess.run(cmd, capture_output=True, text=True)
    return "Audio:" in res.stderr


async def extract_last_frame(video_path: str, output_image_path: str | None = None) -> str:
    """Extracts the final frame of a video clip.

    Used by the Continuity Chain to feed Segment N-1's last frame as Segment N's starting frame.

    Args:
        video_path: Path to the input video clip.
        output_image_path: Destination path for the extracted frame (JPEG/PNG). If None, generates temp path.

    Returns:
        Path to the extracted frame image file.
    """
    if not output_image_path:
        fd, output_image_path = tempfile.mkstemp(suffix=".jpg", prefix="last_frame_")
        os.close(fd)

    ffmpeg = get_ffmpeg_exe()

    # Primary strategy: seek to 0.1s before end of file
    cmd1 = [
        ffmpeg,
        "-sseof", "-0.1",
        "-i", video_path,
        "-update", "1",
        "-q:v", "2",
        output_image_path,
        "-y",
    ]

    proc = await asyncio.create_subprocess_exec(
        *cmd1, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    _, _ = await proc.communicate()

    if proc.returncode == 0 and os.path.exists(output_image_path) and os.path.getsize(output_image_path) > 0:
        return output_image_path

    # Fallback strategy: read first available frame if clip is very short
    cmd2 = [
        ffmpeg,
        "-i", video_path,
        "-vframes", "1",
        "-update", "1",
        "-q:v", "2",
        output_image_path,
        "-y",
    ]

    proc2 = await asyncio.create_subprocess_exec(
        *cmd2, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    _, stderr2 = await proc2.communicate()
    if proc2.returncode != 0 or not os.path.exists(output_image_path) or os.path.getsize(output_image_path) == 0:
        raise RuntimeError(f"Failed to extract last frame from {video_path}: {stderr2.decode('utf-8', errors='ignore')}")

    return output_image_path


async def normalize_clip(input_path: str, output_path: str, fps: int = 30) -> str:
    """Normalizes video frame rate, pixel format, timebase, and ensures standard AAC audio.

    Normalizing clips immediately after generation is critical for continuity chaining
    and seamless FFmpeg concatenation without audio/video drift or codec mismatches.

    Args:
        input_path: Path to raw input video.
        output_path: Path to normalized output video.
        fps: Target frame rate (default 30).

    Returns:
        The output_path.
    """
    ffmpeg = get_ffmpeg_exe()
    audio_present = has_audio_track(input_path)

    if audio_present:
        cmd = [
            ffmpeg,
            "-i", input_path,
            "-r", str(fps),
            "-pix_fmt", "yuv420p",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "22",
            "-c:a", "aac",
            "-ar", "44100",
            "-ac", "2",
            output_path,
            "-y",
        ]
    else:
        # Generate silent audio track matching video duration
        cmd = [
            ffmpeg,
            "-i", input_path,
            "-f", "lavfi",
            "-i", "anullsrc=r=44100:cl=stereo",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "22",
            "-r", str(fps),
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-shortest",
            output_path,
            "-y",
        ]

    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"Failed to normalize clip {input_path}: {stderr.decode('utf-8', errors='ignore')}")

    return output_path


async def compress_video(input_path: str, output_path: str, max_height: int = 720) -> str:
    """Compresses video for lightweight preview or upload.

    Args:
        input_path: Source video path.
        output_path: Compressed destination path.
        max_height: Maximum vertical resolution (default 720p).

    Returns:
        output_path.
    """
    ffmpeg = get_ffmpeg_exe()
    cmd = [
        ffmpeg,
        "-i", input_path,
        "-vf", f"scale=-2:min({max_height}\\,ih)",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "28",
        "-c:a", "aac",
        "-b:a", "128k",
        output_path,
        "-y",
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"Failed to compress video {input_path}: {stderr.decode('utf-8', errors='ignore')}")

    return output_path
