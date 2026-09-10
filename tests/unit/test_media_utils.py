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
    probe_duration,
)
from app.media.composite import composite_streamer_over_gameplay
from app.media.stitch import concat_clips
from app.media.subtitles import build_ass_from_segments, format_ass_time


async def create_test_video(path: str, duration: int = 1, color: str = "blue") -> str:
    """Helper to generate a tiny synthetic mp4 clip with test audio."""
    ffmpeg = get_ffmpeg_exe()
    cmd = [
        ffmpeg,
        "-f",
        "lavfi",
        "-i",
        f"color=c={color}:s=320x240:d={duration}",
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency=1000:duration={duration}",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
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
        # 1s streamer over 2s gameplay -> the streamer sets the length.
        assert round(res_stacked["durationSeconds"]) == 1


@pytest.mark.asyncio
async def test_composite_runs_exactly_as_long_as_the_streamer_track():
    """The finished video is the streamer's length in both directions of mismatch.

    Longer footage has its tail dropped; shorter footage holds its last frame.
    driftSeconds reports the gap either way without changing the length.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Case A: gameplay (3s) longer than streamer (1s) -> tail dropped, 1s out.
        streamer_1s = os.path.join(tmpdir, "streamer_1s.mp4")
        gameplay_3s = os.path.join(tmpdir, "gameplay_3s.mp4")
        await create_test_video(streamer_1s, duration=1, color="red")
        await create_test_video(gameplay_3s, duration=3, color="blue")

        out_a = os.path.join(tmpdir, "out_a.mp4")
        res_a = await composite_streamer_over_gameplay(
            streamer_path=streamer_1s,
            gameplay_path=gameplay_3s,
            output_path=out_a,
            layout="pip",
        )
        assert round(res_a["durationSeconds"]) == 1
        assert round(probe_duration(out_a)) == 1
        assert res_a["driftSeconds"] == -2.0

        # Case B: streamer (3s) longer than gameplay (1s) -> gameplay freezes, 3s out.
        streamer_3s = os.path.join(tmpdir, "streamer_3s.mp4")
        gameplay_1s = os.path.join(tmpdir, "gameplay_1s.mp4")
        await create_test_video(streamer_3s, duration=3, color="red")
        await create_test_video(gameplay_1s, duration=1, color="blue")

        out_b = os.path.join(tmpdir, "out_b.mp4")
        res_b = await composite_streamer_over_gameplay(
            streamer_path=streamer_3s,
            gameplay_path=gameplay_1s,
            output_path=out_b,
            layout="pip",
        )
        assert round(res_b["durationSeconds"]) == 3
        assert round(probe_duration(out_b)) == 3
        assert res_b["driftSeconds"] == 2.0


def test_plan_composite_geometry():
    from app.media.composite import plan_composite

    # 1. Picture-in-picture geometry test: 10% of 1920x1080 canvas area
    plan_pip = plan_composite(
        "16:9", "picture-in-picture", "bottom-right", None, 16 / 9
    )
    streamer_pip = plan_pip["streamer"]
    area_ratio = (streamer_pip["width"] * streamer_pip["height"]) / (1920 * 1080)
    assert abs(area_ratio - 0.1) < 0.02  # ~10% area

    # 2. 9:16 Stacked layout test: streamer is 35% of 1920 height = 672
    plan_stack_v = plan_composite("9:16", "stacked", "bottom-right", "top", 16 / 9)
    assert plan_stack_v["streamer"]["height"] == 672
    assert plan_stack_v["gameplay"]["height"] == 1920 - 672

    # 3. 16:9 Stacked layout test: streamer is 30% of 1920 width = 576
    plan_stack_h = plan_composite("16:9", "stacked", "bottom-right", "left", 16 / 9)
    assert plan_stack_h["streamer"]["width"] == 576
    assert plan_stack_h["gameplay"]["width"] == 1920 - 576


def test_subtitles_dejavu_font_and_scaling():
    from app.media.subtitles import SUBTITLE_FONT, build_ass_from_segments

    assert SUBTITLE_FONT == "DejaVu Sans"

    segments = [{"duration": 4, "dialogue": "[Gasping] Look at that dragon!"}]
    ass_1080p = build_ass_from_segments(segments, aspect_ratio="16:9")
    assert "DejaVu Sans" in ass_1080p
    assert "Look at that dragon!" in ass_1080p
    assert "[Gasping]" not in ass_1080p


@pytest.mark.asyncio
async def test_omni_silent_rejection(monkeypatch):
    from unittest.mock import AsyncMock, MagicMock, patch

    from app.media.omni import omni_interaction

    monkeypatch.delenv("MOCK_OMNI", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    mock_resp = AsyncMock()
    mock_resp.status = 200
    mock_resp.ok = True
    mock_resp.json = AsyncMock(
        return_value={"status": "completed", "usage": {"total_input_tokens": 0}}
    )

    mock_post_cm = MagicMock()
    mock_post_cm.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_post_cm.__aexit__ = AsyncMock(return_value=None)

    mock_session = MagicMock()
    mock_session.post = MagicMock(return_value=mock_post_cm)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)

    with patch("aiohttp.ClientSession", return_value=mock_session):
        with pytest.raises(RuntimeError) as exc:
            await omni_interaction(
                prompt="test",
                start_frame_base64="data:image/png;base64,AAAA",
                duration_seconds=3,
                aspect_ratio="16:9",
                dest_path="dummy.mp4",
                mock=False,
            )
        assert "silently rejected source image" in str(exc.value)
