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

"""FFmpeg compositing utility: overlays streamer reaction video over gameplay footage.

**The output is exactly as long as the streamer track.** Both directions of the
mismatch are handled in one chain with no branch: the gameplay is padded with a
clone of its last frame (`tpad`) when it is the shorter one, and `-t` cuts the
whole thing to the streamer's length either way. A gameplay tail with nobody
talking over it is dead air; a line of dialogue cut off halfway is a broken
video, and the call to action lives in the last segment.

This used to run to `max(streamer, gameplay)` and pad the streamer to match,
which turned every under-covering script into a finished video that froze the
streamer on their last frame -- silent, uncaptioned -- for the whole remaining
length of the footage.
"""

from __future__ import annotations

import asyncio
import math
import os
from typing import Any

from app.media.clips import (
    CANONICAL,
    get_ffmpeg_exe,
    probe_duration,
    probe_has_audio,
    probe_video_dimensions,
)

# Reference canvas dimensions (1080p either way)
COMPOSITE_FRAMES = {
    "16:9": {"width": 1920, "height": 1080},
    "9:16": {"width": 1080, "height": 1920},
}

PIP_AREA = 0.1  # Picture-in-picture takes exactly 10% of canvas area
PIP_MARGIN = 0.02  # 2% margin from frame edge
STACK_COLUMN = 0.30  # Streamer takes 30% width in 16:9 stacked layout
STACK_BAND = 0.35  # Streamer takes 35% height in 9:16 stacked layout
MIN_KEPT_FRACTION = 2 / 3  # Minimum kept fraction to prefer cover over contain


def even(n: float) -> int:
    """Rounds to nearest even integer for h.264 chroma alignment."""
    return max(2, round(n / 2) * 2)


def cover_keeps(src: float, dst: float) -> float:
    """The share of the source dimension a cover-crop into dst would keep."""
    return min(src, dst) / max(src, dst)


def plan_composite(
    aspect_ratio: str,
    layout: str,
    pip_placement: str,
    stacked_placement: str | None,
    streamer_aspect: float,
) -> dict[str, Any]:
    """Calculates frame rectangles for gameplay and streamer according to exact reference geometry."""
    frame = COMPOSITE_FRAMES.get(aspect_ratio, COMPOSITE_FRAMES["16:9"])
    w = frame["width"]
    h = frame["height"]

    if layout == "streamer-only":
        return {
            "frame": frame,
            "gameplay": None,
            "streamer": {"x": 0, "y": 0, "width": w, "height": h},
        }

    if layout == "stacked":
        if aspect_ratio == "9:16":
            split = even(h * STACK_BAND)
            top = (stacked_placement or "top") == "top"
            return {
                "frame": frame,
                "streamer": {
                    "x": 0,
                    "y": 0 if top else h - split,
                    "width": w,
                    "height": split,
                },
                "gameplay": {
                    "x": 0,
                    "y": split if top else 0,
                    "width": w,
                    "height": h - split,
                },
            }
        else:
            split = even(w * STACK_COLUMN)
            left = (stacked_placement or "left") != "right"
            return {
                "frame": frame,
                "streamer": {
                    "x": 0 if left else w - split,
                    "y": 0,
                    "width": split,
                    "height": h,
                },
                "gameplay": {
                    "x": split if left else 0,
                    "y": 0,
                    "width": w - split,
                    "height": h,
                },
            }

    # Picture-in-picture layout: area is 10% of total frame area
    pip_width = even(math.sqrt(w * h * PIP_AREA * streamer_aspect))
    pip_height = even(pip_width / streamer_aspect)
    margin = even(w * PIP_MARGIN)
    right = pip_placement in ("top-right", "bottom-right")
    top = pip_placement in ("top-left", "top-right")

    return {
        "frame": frame,
        "gameplay": {"x": 0, "y": 0, "width": w, "height": h},
        "streamer": {
            "x": w - pip_width - margin if right else margin,
            "y": margin if top else h - pip_height - margin,
            "width": pip_width,
            "height": pip_height,
        },
    }


def build_composite_graph(
    plan: dict[str, Any],
    streamer_seconds: float,
    gameplay_seconds: float | None,
    gameplay_has_audio: bool,
    gameplay_aspect: float | None,
    gameplay_volume: float,
    streamer_volume: float,
    subtitles_ass_path: str | None = None,
) -> tuple[str, str | None]:
    """Constructs the filter_complex graph string matching composite.ts."""
    frame = plan["frame"]
    w = frame["width"]
    h = frame["height"]
    chains: list[str] = []
    gameplay_fit: str | None = None

    def fit_rect(r: dict[str, int]) -> str:
        return (
            f"scale={r['width']}:{r['height']}:force_original_aspect_ratio=decrease,"
            f"pad={r['width']}:{r['height']}:(ow-iw)/2:(oh-ih)/2:black,setsar=1"
        )

    if plan["gameplay"]:
        g = plan["gameplay"]

        # Only pad when the gameplay is the shorter one. The half second is slack
        # against the probe's rounding, and -t cuts it off again. The streamer is
        # never padded: it is what sets the length, so it has no shortfall.
        gameplay_shortfall = (
            0.0
            if gameplay_seconds is None
            else max(0.0, streamer_seconds - gameplay_seconds)
        )
        gameplay_tpad = (
            f",tpad=stop_mode=clone:stop_duration={gameplay_shortfall + 0.5:.2f}"
            if gameplay_shortfall > 0
            else ""
        )

        target_aspect = g["width"] / g["height"]
        if (
            gameplay_aspect is None
            or cover_keeps(gameplay_aspect, target_aspect) >= MIN_KEPT_FRACTION
        ):
            gameplay_fit = "cover"
            place = (
                f"scale={g['width']}:{g['height']}:force_original_aspect_ratio=increase,"
                f"crop={g['width']}:{g['height']}"
            )
        else:
            gameplay_fit = "contain"
            place = (
                f"scale={g['width']}:{g['height']}:force_original_aspect_ratio=decrease,"
                f"pad={g['width']}:{g['height']}:(ow-iw)/2:(oh-ih)/2:black"
            )

        chains.append(
            f"[1:v]{place},setsar=1{gameplay_tpad},pad={w}:{h}:{g['x']}:{g['y']}:black[bg]"
        )
        chains.append(f"[0:v]{fit_rect(plan['streamer'])}[fg]")
        chains.append(
            f"[bg][fg]overlay={plan['streamer']['x']}:{plan['streamer']['y']}:eof_action=repeat[composed]"
        )
        video_tail = "[composed]"
    else:
        chains.append(f"[0:v]{fit_rect(plan['streamer'])}[composed]")
        video_tail = "[composed]"

    # Subtitles burned at finished composite resolution
    if subtitles_ass_path and os.path.exists(subtitles_ass_path):
        safe_ass = (
            subtitles_ass_path.replace("\\", "/")
            .replace(":", "\\:")
            .replace("'", "\\'")
        )
        chains.append(f"{video_tail}ass='{safe_ass}'[v]")
    else:
        chains.append(f"{video_tail}null[v]")

    # Audio mixing: amix with normalize=0 so volume values are applied directly
    chains.append(f"[0:a]volume={streamer_volume}[sa]")
    if plan["gameplay"] and gameplay_has_audio:
        chains.append(f"[1:a]volume={gameplay_volume}[ga]")
        chains.append("[sa][ga]amix=inputs=2:duration=longest:normalize=0[a]")
    else:
        chains.append("[sa]anull[a]")

    return ";".join(chains), gameplay_fit


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
) -> dict[str, Any]:
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
        Dict with duration, output path, drift metadata, and gameplay placement fit.
    """
    ffmpeg = get_ffmpeg_exe()

    streamer_dims = probe_video_dimensions(streamer_path)
    streamer_dur = probe_duration(streamer_path) or 0.0
    if not streamer_dur:
        raise RuntimeError(f"Could not read streamer track duration: {streamer_path}")

    streamer_aspect = (
        (streamer_dims[0] / streamer_dims[1])
        if streamer_dims and streamer_dims[1] > 0
        else (16 / 9)
    )

    # Standardize layout names ('pip' -> 'picture-in-picture')
    effective_layout = (
        "picture-in-picture" if layout in ("pip", "picture-in-picture") else layout
    )

    plan = plan_composite(
        aspect_ratio=aspect_ratio,
        layout=effective_layout,
        pip_placement=pip_placement,
        stacked_placement=stacked_placement,
        streamer_aspect=streamer_aspect,
    )

    gameplay_dur = None
    gameplay_has_audio = False
    gameplay_aspect = None

    if (
        gameplay_path
        and effective_layout != "streamer-only"
        and os.path.exists(gameplay_path)
    ):
        gameplay_dur = probe_duration(gameplay_path)
        gameplay_has_audio = probe_has_audio(gameplay_path)
        g_dims = probe_video_dimensions(gameplay_path)
        if g_dims and g_dims[1] > 0:
            gameplay_aspect = g_dims[0] / g_dims[1]

    filter_complex, gameplay_fit = build_composite_graph(
        plan=plan,
        streamer_seconds=streamer_dur,
        gameplay_seconds=gameplay_dur,
        gameplay_has_audio=gameplay_has_audio,
        gameplay_aspect=gameplay_aspect,
        gameplay_volume=gameplay_volume,
        streamer_volume=streamer_volume,
        subtitles_ass_path=subtitles_ass_path,
    )

    cmd = [
        ffmpeg,
        "-y",
        "-i",
        streamer_path,
        *(
            ["-i", gameplay_path]
            if gameplay_path and effective_layout != "streamer-only"
            else []
        ),
        "-filter_complex",
        filter_complex,
        "-map",
        "[v]",
        "-map",
        "[a]",
        "-t",
        f"{streamer_dur:.3f}",
        "-c:v",
        CANONICAL["video_codec"],
        "-preset",
        CANONICAL["preset"],
        "-crf",
        CANONICAL["crf"],
        "-pix_fmt",
        CANONICAL["pixel_format"],
        "-r",
        str(CANONICAL["fps"]),
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
            f"FFmpeg composite failed: {stderr.decode('utf-8', errors='ignore')}"
        )

    drift = (
        round(streamer_dur - gameplay_dur, 1)
        if gameplay_dur is not None and effective_layout != "streamer-only"
        else None
    )

    return {
        "outputPath": output_path,
        "durationSeconds": round(streamer_dur, 1),
        "streamerDurationSeconds": round(streamer_dur, 1),
        "gameplayDurationSeconds": round(gameplay_dur, 1)
        if gameplay_dur is not None
        else None,
        "driftSeconds": drift,
        "gameplayFit": gameplay_fit,
        "gameplayAspect": gameplay_aspect,
    }
