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

import asyncio
import os
import tempfile

import pytest

from app.media.clips import (
    extract_last_frame,
    get_ffmpeg_exe,
    has_audio_track,
    normalize_clip,
)
from app.media.composite import composite_streamer_over_gameplay
from app.media.stitch import concat_clips
from app.media.subtitles import build_ass_from_segments, format_ass_time


async def create_test_video(path: str, duration: int = 1, color: str = "blue") -> str:
    """Helper to generate a tiny synthetic mp4 clip with test audio."""
    ffmpeg = get_ffmpeg_exe()
    cmd = [
        ffmpeg,
        "-f", "lavfi",
        "-i", f"color=c={color}:s=320x240:d={duration}",
        "-f", "lavfi",
        "-i", f"sine=frequency=1000:duration={duration}",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        path,
        "-y",
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    await proc.communicate()
    return path


@pytest.mark.asyncio
async def test_ffmpeg_exe_found():
    exe = get_ffmpeg_exe()
    assert exe is not None
    assert os.path.exists(exe)


@pytest.mark.asyncio
async def test_clips_and_frame_extraction():
    with tempfile.TemporaryDirectory() as tmpdir:
        clip1 = os.path.join(tmpdir, "test1.mp4")
        await create_test_video(clip1, duration=1, color="red")

        assert os.path.exists(clip1)
        assert has_audio_track(clip1) is True

        # Extract last frame
        frame_path = os.path.join(tmpdir, "last_frame.jpg")
        await extract_last_frame(clip1, frame_path)
        assert os.path.exists(frame_path)
        assert os.path.getsize(frame_path) > 0

        # Normalize clip
        norm_path = os.path.join(tmpdir, "norm1.mp4")
        await normalize_clip(clip1, norm_path, fps=30)
        assert os.path.exists(norm_path)


@pytest.mark.asyncio
async def test_concat_clips():
    with tempfile.TemporaryDirectory() as tmpdir:
        clip1 = os.path.join(tmpdir, "c1.mp4")
        clip2 = os.path.join(tmpdir, "c2.mp4")
        await create_test_video(clip1, duration=1, color="red")
        await create_test_video(clip2, duration=1, color="green")

        out_path = os.path.join(tmpdir, "stitched.mp4")
        await concat_clips([clip1, clip2], out_path)
        assert os.path.exists(out_path)
        assert os.path.getsize(out_path) > 0


def test_subtitles_formatting():
    assert format_ass_time(0.0) == "0:00:00.00"
    assert format_ass_time(65.5) == "0:01:05.50"

    segments = [
        {"duration": 3, "dialogue": "[Laughing] What an epic start!"},
        {"duration": 5, "dialogue": "Watch this flank right here."},
    ]
    ass_text = build_ass_from_segments(segments, aspect_ratio="16:9")
    assert "[Script Info]" in ass_text
    assert "Dialogue: 0,0:00:00.00,0:00:03.00" in ass_text
    assert "What an epic start!" in ass_text
    assert "[Laughing]" not in ass_text  # stripped from visible text


@pytest.mark.asyncio
async def test_composite_pip_and_stacked():
    with tempfile.TemporaryDirectory() as tmpdir:
        streamer_vid = os.path.join(tmpdir, "streamer.mp4")
        gameplay_vid = os.path.join(tmpdir, "gameplay.mp4")
        await create_test_video(streamer_vid, duration=1, color="yellow")
        await create_test_video(gameplay_vid, duration=2, color="black")

        pip_out = os.path.join(tmpdir, "pip_out.mp4")
        res_pip = await composite_streamer_over_gameplay(
            streamer_path=streamer_vid,
            gameplay_path=gameplay_vid,
            output_path=pip_out,
            aspect_ratio="16:9",
            layout="pip",
            pip_placement="bottom-right",
        )
        assert os.path.exists(pip_out)
        assert res_pip["durationSeconds"] > 0

        stacked_out = os.path.join(tmpdir, "stacked_out.mp4")
        res_stacked = await composite_streamer_over_gameplay(
            streamer_path=streamer_vid,
            gameplay_path=gameplay_vid,
            output_path=stacked_out,
            aspect_ratio="9:16",
            layout="stacked",
            stacked_placement="top",
        )
        assert os.path.exists(stacked_out)
        assert res_stacked["durationSeconds"] > 0
