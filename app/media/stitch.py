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
import os
import shutil
import tempfile

from app.media.clips import get_ffmpeg_exe


async def concat_clips(clip_paths: list[str], output_path: str) -> str:
    """Concatenates multiple normalized MP4 clips in order into a single video file.

    Args:
        clip_paths: List of absolute paths to normalized MP4 video files.
        output_path: Destination path for the stitched video.

    Returns:
        The output_path.
    """
    if not clip_paths:
        raise ValueError("Cannot concatenate empty list of clips.")

    if len(clip_paths) == 1:
        shutil.copyfile(clip_paths[0], output_path)
        return output_path

    ffmpeg = get_ffmpeg_exe()

    # Create temporary concat list file
    fd, list_path = tempfile.mkstemp(suffix=".txt", prefix="ffmpeg_concat_")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        for p in clip_paths:
            abs_p = os.path.abspath(p).replace("'", "'\\''")
            f.write(f"file '{abs_p}'\n")

    try:
        # Fast path: stream copy
        cmd_copy = [
            ffmpeg,
            "-f", "concat",
            "-safe", "0",
            "-i", list_path,
            "-c", "copy",
            output_path,
            "-y",
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd_copy, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        _, _ = await proc.communicate()

        if proc.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return output_path

        # Fallback path: re-encode if stream copy fails
        cmd_reencode = [
            ffmpeg,
            "-f", "concat",
            "-safe", "0",
            "-i", list_path,
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            output_path,
            "-y",
        ]
        proc2 = await asyncio.create_subprocess_exec(
            *cmd_reencode, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        _, stderr2 = await proc2.communicate()
        if proc2.returncode != 0:
            raise RuntimeError(f"Failed to concatenate clips: {stderr2.decode('utf-8', errors='ignore')}")

        return output_path
    finally:
        if os.path.exists(list_path):
            os.remove(list_path)
