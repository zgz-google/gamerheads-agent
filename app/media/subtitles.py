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

"""Subtitle generation utilities for creating styled ASS subtitle files."""

from __future__ import annotations

import re
from typing import Any

# Default font: DejaVu Sans is bundled and available on headless Linux / container runtimes
SUBTITLE_FONT = "DejaVu Sans"


def format_ass_time(seconds: float) -> str:
    """Formats seconds into ASS timestamp format: H:MM:SS.cs."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centis = round((seconds - int(seconds)) * 100)
    if centis >= 100:
        secs += 1
        centis = 0
    return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"


def clean_dialogue_for_subtitles(dialogue: str) -> str:
    """Removes bracketed vocal direction cues (e.g. [Laughing], (gasp)) from visible subtitles."""
    cleaned = re.sub(r"\[.*?\]", "", dialogue)
    cleaned = re.sub(r"\(.*?\)", "", cleaned)
    return " ".join(cleaned.split()).strip()


def build_ass_from_segments(
    segments: list[dict[str, Any]],
    aspect_ratio: str = "16:9",
) -> str:
    """Constructs Advanced SubStation Alpha (.ass) subtitle file content from shot list segments.

    Dynamically calculates font size, outline, shadow, and margins based on video height:
      fontSize = ~4.2% of height (clamped 26..64)
      outline  = ~10% of fontSize (min 3)
      shadow   = ~6% of fontSize  (min 2)
      marginV  = ~7% of height    (min 40)

    Args:
        segments: List of segment dicts containing 'duration' and 'dialogue'.
        aspect_ratio: '16:9' (landscape 1920x1080) or '9:16' (portrait 1080x1920).

    Returns:
        The complete .ass file content as a string.
    """
    is_vertical = aspect_ratio == "9:16"
    res_x = 1080 if is_vertical else 1920
    res_y = 1920 if is_vertical else 1080

    font_size = max(26, min(64, round(res_y * 0.042)))
    outline = max(3, round(font_size * 0.1))
    shadow = max(2, round(font_size * 0.06))
    margin_v = max(40, round(res_y * 0.07))

    style_line = (
        f"Style: Default,{SUBTITLE_FONT},{font_size},&H00FFFFFF,&H000000FF,"
        f"&H00000000,&H80000000,1,0,0,0,100,100,0.4,0,1,{outline},{shadow},"
        f"2,80,80,{margin_v},1"
    )

    header = f"""[Script Info]
Title: GamerHeads Subtitles
ScriptType: v4.00+
WrapStyle: 0
ScaledBorderAndShadow: yes
PlayResX: {res_x}
PlayResY: {res_y}

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
{style_line}

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    lines = [header.strip()]
    cursor = 0.0

    for seg in segments:
        dur = float(seg.get("duration", 5))
        start_t = format_ass_time(cursor)
        end_t = format_ass_time(cursor + dur)
        cursor += dur

        raw_dialogue = seg.get("dialogue", "")
        clean_text = clean_dialogue_for_subtitles(raw_dialogue)
        if clean_text:
            escaped = (
                clean_text.replace("\n", "\\N").replace("{", "\\{").replace("}", "\\}")
            )
            lines.append(f"Dialogue: 0,{start_t},{end_t},Default,,0,0,0,,{escaped}")

    return "\n".join(lines) + "\n"
