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

"""FFmpeg video clip utilities: frame extraction, normalization, compression, and stream probing."""

from __future__ import annotations

import asyncio
import base64
import os
import re
import shutil
import subprocess
import tempfile
import uuid
from typing import Any

import imageio_ffmpeg

# Canonical encode profile guaranteed across all rendered clips
CANONICAL = {
    "fps": 24,
    "pixel_format": "yuv420p",
    "video_codec": "libx264",
    "crf": "18",
    "preset": "veryfast",
    "audio_codec": "aac",
    "audio_bitrate": "192k",
    "sample_rate": "44100",
    "channels": "2",
}


def get_ffmpeg_exe() -> str:
    """Returns the path to the ffmpeg executable."""
    env_path = os.getenv("FFMPEG_PATH")
    if env_path and os.path.exists(env_path):
        return env_path
    which_path = shutil.which("ffmpeg")
    if which_path:
        return which_path
    return imageio_ffmpeg.get_ffmpeg_exe()


def get_ffprobe_exe() -> str | None:
    """Returns the path to the ffprobe executable if available."""
    env_path = os.getenv("FFPROBE_PATH")
    if env_path and os.path.exists(env_path):
        return env_path
    which_path = shutil.which("ffprobe")
    if which_path:
        return which_path
    return None


def probe_video_info(video_path: str) -> dict[str, Any]:
    """Probes video metadata using ffprobe or ffmpeg fallback.

    Returns dict with keys:
        - width: int | None
        - height: int | None
        - duration: float | None
        - video_duration: float | None
        - has_audio: bool
        - fps: float | None
    """
    ffprobe = get_ffprobe_exe()
    if ffprobe:
        try:
            cmd = [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration:stream=width,height,duration,r_frame_rate,codec_type",
                "-of",
                "json",
                video_path,
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
            import json

            data = json.loads(res.stdout)
            streams = data.get("streams", [])
            format_info = data.get("format", {})

            width = None
            height = None
            v_dur = None
            has_audio = False
            fps = None

            for s in streams:
                if s.get("codec_type") == "video":
                    width = s.get("width")
                    height = s.get("height")
                    if s.get("duration"):
                        try:
                            v_dur = float(s["duration"])
                        except ValueError:
                            pass
                    r_fps = s.get("r_frame_rate", "")
                    if "/" in r_fps:
                        num, den = r_fps.split("/")
                        if den and float(den) > 0:
                            fps = float(num) / float(den)
                elif s.get("codec_type") == "audio":
                    has_audio = True

            dur = None
            if format_info.get("duration"):
                try:
                    dur = float(format_info["duration"])
                except ValueError:
                    pass
            if dur is None:
                dur = v_dur

            return {
                "width": width,
                "height": height,
                "duration": dur,
                "video_duration": v_dur if v_dur is not None else dur,
                "has_audio": has_audio,
                "fps": fps,
            }
        except Exception:
            pass

    # Fallback to ffmpeg -i stderr inspection
    ffmpeg = get_ffmpeg_exe()
    cmd = [ffmpeg, "-i", video_path]
    res = subprocess.run(cmd, capture_output=True, text=True)
    stderr = res.stderr

    dur = None
    width = None
    height = None
    has_audio = "Audio:" in stderr
    fps = None

    # Parse Duration: 00:01:23.45
    m_dur = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.?\d*)", stderr)
    if m_dur:
        h, m, s = m_dur.groups()
        dur = float(h) * 3600 + float(m) * 60 + float(s)

    # Parse Video stream dimensions: e.g. 1920x1080 or 1280x720
    m_dim = re.search(r"Stream #.*?Video:.*?(\d{2,5})x(\d{2,5})", stderr)
    if m_dim:
        width = int(m_dim.group(1))
        height = int(m_dim.group(2))

    m_fps = re.search(r"(\d+(?:\.\d+)?)\s*fps", stderr)
    if m_fps:
        fps = float(m_fps.group(1))

    return {
        "width": width,
        "height": height,
        "duration": dur,
        "video_duration": dur,
        "has_audio": has_audio,
        "fps": fps,
    }


def probe_video_dimensions(video_path: str) -> tuple[int, int] | None:
    """Returns (width, height) or None if probing fails."""
    info = probe_video_info(video_path)
    if info.get("width") and info.get("height"):
        return (info["width"], info["height"])
    return None


def probe_duration(video_path: str) -> float | None:
    """Returns duration in seconds or None if probing fails."""
    return probe_video_info(video_path).get("duration")


def probe_video_duration(video_path: str) -> float | None:
    """Returns video stream duration in seconds or None if probing fails."""
    return probe_video_info(video_path).get("video_duration")


def probe_has_audio(video_path: str) -> bool:
    """Checks if the video file contains an audio stream."""
    return probe_video_info(video_path).get("has_audio", False)


def has_audio_track(video_path: str) -> bool:
    """Backward compatibility alias for probe_has_audio."""
    return probe_has_audio(video_path)


async def normalize_clip(input_path: str, output_path: str, fps: int = 24) -> str:
    """Normalizes video clip into canonical format: CFR 24fps, PTS reset, AAC 44.1kHz stereo.

    Omni Flash hands back clips whose timestamps do not start at zero and whose AAC tracks
    carry encoder priming; concatenating those as-is accumulates a few milliseconds of A/V skew
    per clip until lips stop matching speech. Resetting PTS and resampling prevents this drift.

    Args:
        input_path: Path to raw input video.
        output_path: Path to normalized output video.
        fps: Target frame rate (default 24).

    Returns:
        The output_path.
    """
    ffmpeg = get_ffmpeg_exe()
    audio_present = probe_has_audio(input_path)

    vf = f"setpts=PTS-STARTPTS,fps={fps}"

    if audio_present:
        cmd = [
            ffmpeg,
            "-y",
            "-i",
            input_path,
            "-vf",
            vf,
            "-af",
            "asetpts=PTS-STARTPTS,aresample=async=1:first_pts=0",
            "-c:v",
            CANONICAL["video_codec"],
            "-preset",
            CANONICAL["preset"],
            "-crf",
            CANONICAL["crf"],
            "-pix_fmt",
            CANONICAL["pixel_format"],
            "-r",
            str(fps),
            "-c:a",
            CANONICAL["audio_codec"],
            "-b:a",
            CANONICAL["audio_bitrate"],
            "-ar",
            CANONICAL["sample_rate"],
            "-ac",
            CANONICAL["channels"],
            "-shortest",
            "-movflags",
            "+faststart",
            output_path,
        ]
    else:
        cmd = [
            ffmpeg,
            "-y",
            "-i",
            input_path,
            "-f",
            "lavfi",
            "-i",
            f"anullsrc=channel_layout=stereo:sample_rate={CANONICAL['sample_rate']}",
            "-vf",
            vf,
            "-af",
            "asetpts=PTS-STARTPTS,aresample=async=1:first_pts=0",
            "-c:v",
            CANONICAL["video_codec"],
            "-preset",
            CANONICAL["preset"],
            "-crf",
            CANONICAL["crf"],
            "-pix_fmt",
            CANONICAL["pixel_format"],
            "-r",
            str(fps),
            "-c:a",
            CANONICAL["audio_codec"],
            "-b:a",
            CANONICAL["audio_bitrate"],
            "-ar",
            CANONICAL["sample_rate"],
            "-ac",
            CANONICAL["channels"],
            "-shortest",
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
            f"Failed to normalize clip {input_path}: {stderr.decode('utf-8', errors='ignore')}"
        )

    return output_path


async def extract_last_frame(
    video_path: str, output_image_path: str | None = None
) -> str:
    """Extracts the final frame of a video clip using trailing-window overwrite.

    Decodes the final 0.5 seconds of the video stream with -update 1, continuously
    overwriting output_image_path until EOF so the final written image is guaranteed
    to be the true last frame without tight PTS-margin sensitivity. Falls back to full
    decode with -update 1 before raising.

    Used by the Continuity Chain to seed Segment N from Segment N-1's final pose.

    Args:
        video_path: Path to the input video clip.
        output_image_path: Destination path for extracted JPEG. If None, creates a temp file.

    Returns:
        Path to the extracted frame image file.
    """
    if not output_image_path:
        fd, output_image_path = tempfile.mkstemp(suffix=".jpg", prefix="last_frame_")
        os.close(fd)

    ffmpeg = get_ffmpeg_exe()
    v_dur = probe_video_duration(video_path) or 0.0

    # Primary: seek to (D_video - 0.5s) and continuously overwrite to EOF
    seek_time = max(0.0, v_dur - 0.5)
    cmd1 = [
        ffmpeg,
        "-y",
        "-ss",
        f"{seek_time:.3f}",
        "-i",
        video_path,
        "-update",
        "1",
        "-q:v",
        "2",
        output_image_path,
    ]

    proc1 = await asyncio.create_subprocess_exec(
        *cmd1, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    await proc1.communicate()

    if (
        proc1.returncode == 0
        and os.path.exists(output_image_path)
        and os.path.getsize(output_image_path) > 0
    ):
        return output_image_path

    # Fallback: full decode continuous overwrite (safe against corrupt seek indexes, low RSS)
    cmd2 = [
        ffmpeg,
        "-y",
        "-i",
        video_path,
        "-update",
        "1",
        "-q:v",
        "2",
        output_image_path,
    ]
    proc2 = await asyncio.create_subprocess_exec(
        *cmd2, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    _, stderr2 = await proc2.communicate()

    if (
        proc2.returncode == 0
        and os.path.exists(output_image_path)
        and os.path.getsize(output_image_path) > 0
    ):
        return output_image_path

    # Hard error on failure: do NOT silently fallback to the first frame!
    raise RuntimeError(
        f"Failed to extract last frame from {video_path}: {stderr2.decode('utf-8', errors='ignore')}"
    )


async def extract_last_frame_data_url(video_path: str) -> str:
    """Extracts the last frame and returns it as a JPEG data URL."""
    tmp_path = os.path.join(
        os.path.dirname(video_path), f"lastframe-{uuid.uuid4().hex[:8]}.jpg"
    )
    try:
        await extract_last_frame(video_path, tmp_path)
        with open(tmp_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


async def compress_video(
    input_path: str, output_path: str, max_height: int | None = None
) -> str:
    """Shrinks gameplay footage enough to ride inline in LLM generation:
    720p/2.5Mbps under 30s, 540p/1.5Mbps at or over 30s.
    """
    dims = probe_video_dimensions(input_path)
    if dims is None:
        raise RuntimeError(
            f"Not a valid or complete video file: {os.path.basename(input_path)} "
            "(upload the actual video file, not a web link)"
        )

    duration = probe_duration(input_path) or 0.0
    if max_height is not None:
        max_dim = max_height
        video_bitrate = "2000k"
    else:
        max_dim, video_bitrate = (720, "2500k") if duration <= 30 else (540, "1500k")

    vf = (
        f"scale='min({max_dim},iw)':'min({max_dim},ih)':force_original_aspect_ratio=decrease,"
        "pad=ceil(iw/2)*2:ceil(ih/2)*2:0:0:black"
    )

    ffmpeg = get_ffmpeg_exe()
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        input_path,
        "-vf",
        vf,
        "-c:v",
        "libx264",
        "-b:v",
        video_bitrate,
        "-c:a",
        "aac",
        "-b:a",
        "128k",
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
            f"Failed to compress video {input_path}: {stderr.decode('utf-8', errors='ignore')}"
        )

    return output_path
