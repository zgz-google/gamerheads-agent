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

"""MP4 Video Generation Tool for ADK Agent."""

import math
import os
import struct
import textwrap
import time
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from google.adk.tools import ToolContext
from google.genai import types
from PIL import Image, ImageDraw, ImageFont


def _pad_mp4_file(file_path: Path, target_size_mb: float) -> None:
    """Pads an MP4 file with a standard MPEG-4 'free' box up to target_size_mb bytes."""
    target_bytes = int(target_size_mb * 1024 * 1024)
    current_size = file_path.stat().st_size
    if current_size >= target_bytes:
        return
    pad_len = target_bytes - current_size
    if pad_len < 8:
        return
    with open(file_path, "ab") as f:
        f.write(struct.pack(">I", pad_len))
        f.write(b"free")
        chunk_size = 1024 * 1024
        remaining = pad_len - 8
        dummy_chunk = b"\x00" * min(chunk_size, remaining)
        while remaining > 0:
            write_size = min(chunk_size, remaining)
            if write_size == len(dummy_chunk):
                f.write(dummy_chunk)
            else:
                f.write(b"\x00" * write_size)
            remaining -= write_size


def _get_theme_colors(theme: str) -> dict:
    """Return color palettes based on chosen theme."""
    themes = {
        "gaming": {
            "bg_top": (15, 12, 35),
            "bg_bottom": (35, 15, 60),
            "primary": (0, 240, 255),  # Cyan neon
            "secondary": (255, 0, 128),  # Magenta neon
            "text": (255, 255, 255),
            "subtext": (180, 200, 240),
            "accent": (255, 215, 0),  # Gold
        },
        "cyberpunk": {
            "bg_top": (10, 10, 20),
            "bg_bottom": (20, 2, 35),
            "primary": (255, 230, 0),  # Cyber Yellow
            "secondary": (0, 255, 200),  # Bright Mint
            "text": (255, 255, 255),
            "subtext": (200, 220, 230),
            "accent": (255, 50, 100),
        },
        "minimal": {
            "bg_top": (240, 243, 246),
            "bg_bottom": (220, 225, 230),
            "primary": (30, 60, 120),
            "secondary": (60, 120, 200),
            "text": (20, 25, 35),
            "subtext": (80, 90, 105),
            "accent": (230, 70, 70),
        },
        "sunset": {
            "bg_top": (40, 15, 60),
            "bg_bottom": (120, 40, 60),
            "primary": (255, 150, 50),
            "secondary": (255, 90, 90),
            "text": (255, 255, 255),
            "subtext": (255, 220, 180),
            "accent": (255, 230, 120),
        },
    }
    return themes.get(theme.lower(), themes["gaming"])


async def create_mp4_video(
    title: str,
    description: str = "",
    duration_seconds: int = 4,
    theme: str = "gaming",
    orientation: str = "vertical",
    target_size_mb: float | None = None,
    output_filename: str | None = None,
    tool_context: ToolContext | None = None,
) -> str:
    """Generates a vertical (9:16) MP4 video clip with animated visuals, typography, and stylized branding.

    Use this tool whenever the user asks to generate, create, export, or output a video (MP4).
    Produces a 9:16 vertical video (720x1280) optimized for mobile, TikTok, Reels, and YouTube Shorts.

    Args:
        title: The primary headline or title text displayed in the video.
        description: An optional secondary subtitle, explanation, or summary.
        duration_seconds: The duration of the video in seconds (default: 4 seconds, between 2 and 10).
        theme: Visual theme style. Supported options: 'gaming', 'cyberpunk', 'minimal', 'sunset'.
        orientation: Video aspect ratio ('vertical' for 9:16 720x1280, or 'landscape' for 16:9 1280x720). Defaults to 'vertical'.
        target_size_mb: Optional target file size in megabytes (e.g. 60.0 for a ~60MB test video). When specified, the video container is padded with a standard MPEG-4 'free' box to achieve the target size for testing large-file playback and streaming limits.
        output_filename: Optional custom filename for the MP4 file (e.g. 'game_intro.mp4').

    Returns:
        A status string detailing the generated MP4 file path, resolution, duration, and file size.
    """
    duration = max(2, min(int(duration_seconds), 15))
    fps = 30
    n_frames = duration * fps

    # Default to 9:16 Vertical (720x1280); both divisible by 16 for H.264
    if orientation.lower() == "landscape":
        width, height = 1280, 720
    else:
        width, height = 720, 1280

    output_dir = Path.cwd() / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not output_filename:
        safe_title = "".join(c if c.isalnum() else "_" for c in title.lower())[
            :25
        ].strip("_")
        safe_title = safe_title or "clip"
        timestamp = int(time.time())
        output_filename = f"{safe_title}_{timestamp}.mp4"
    elif not output_filename.endswith(".mp4"):
        output_filename = f"{output_filename}.mp4"

    target_path = output_dir / output_filename
    colors = _get_theme_colors(theme)

    # Scalable typography
    font_title = ImageFont.load_default(size=42 if height > width else 48)
    font_subtitle = ImageFont.load_default(size=22)
    font_badge = ImageFont.load_default(size=18)
    font_ui = ImageFont.load_default(size=15 if height > width else 16)

    # Word wrapping for safe text fitting on vertical displays
    wrap_w_title = 18 if height > width else 28
    wrap_w_desc = 28 if height > width else 45
    wrapped_title = "\n".join(textwrap.wrap(title.upper(), width=wrap_w_title))
    wrapped_desc = (
        "\n".join(textwrap.wrap(description, width=wrap_w_desc)) if description else ""
    )

    # Base gradient background
    base_bg = Image.new("RGB", (width, height))
    draw_base = ImageDraw.Draw(base_bg)
    top_c = colors["bg_top"]
    bot_c = colors["bg_bottom"]

    for y in range(height):
        factor = y / float(height)
        r = int(top_c[0] * (1 - factor) + bot_c[0] * factor)
        g = int(top_c[1] * (1 - factor) + bot_c[1] * factor)
        b = int(top_c[2] * (1 - factor) + bot_c[2] * factor)
        draw_base.line([(0, y), (width, y)], fill=(r, g, b))

    # Pre-generate floating particles/stars
    np.random.seed(42)
    n_particles = 45
    particles_x = np.random.randint(0, width, n_particles)
    particles_y = np.random.randint(0, height, n_particles)
    particles_speed = np.random.uniform(0.6, 2.8, n_particles)
    particles_size = np.random.randint(2, 5, n_particles)

    frames = []

    # Render each frame
    for f in range(n_frames):
        t = f / float(n_frames)  # Progress 0.0 -> 1.0
        frame_img = base_bg.copy()
        draw = ImageDraw.Draw(frame_img)

        # 1. Floating particles
        for p in range(n_particles):
            # Float upwards in vertical orientation
            px = int((particles_x[p] + math.sin(f * 0.04 + p) * 15) % width)
            py = int((particles_y[p] - particles_speed[p] * f * 2.5) % height)
            glow = int(180 + 75 * math.sin(f * 0.1 + p))
            p_color = (
                (glow, glow, 255)
                if "gaming" in theme
                else (255, glow, glow)
                if "sunset" in theme
                else (glow, glow, glow)
            )
            draw.ellipse(
                [px, py, px + particles_size[p], py + particles_size[p]],
                fill=p_color,
            )

        # 2. Modern Cyber / Gamer HUD frame accents
        hud_inset = 28 if height > width else 40
        draw.rectangle(
            [hud_inset, hud_inset, width - hud_inset, height - hud_inset],
            outline=colors["primary"],
            width=2,
        )
        # Decorative corner marks
        corner_len = 32
        c_prim = colors["secondary"]
        # Top-left corner
        draw.line(
            [(hud_inset, hud_inset), (hud_inset + corner_len, hud_inset)],
            fill=c_prim,
            width=4,
        )
        draw.line(
            [(hud_inset, hud_inset), (hud_inset, hud_inset + corner_len)],
            fill=c_prim,
            width=4,
        )
        # Top-right corner
        draw.line(
            [
                (width - hud_inset, hud_inset),
                (width - hud_inset - corner_len, hud_inset),
            ],
            fill=c_prim,
            width=4,
        )
        draw.line(
            [
                (width - hud_inset, hud_inset),
                (width - hud_inset + corner_len, hud_inset),
            ],
            fill=c_prim,
            width=4,
        )
        # Bottom-left corner
        draw.line(
            [
                (hud_inset, height - hud_inset),
                (hud_inset + corner_len, height - hud_inset),
            ],
            fill=c_prim,
            width=4,
        )
        draw.line(
            [
                (hud_inset, height - hud_inset),
                (hud_inset, height - hud_inset - corner_len),
            ],
            fill=c_prim,
            width=4,
        )
        # Bottom-right corner
        draw.line(
            [
                (width - hud_inset, height - hud_inset),
                (width - hud_inset - corner_len, height - hud_inset),
            ],
            fill=c_prim,
            width=4,
        )
        draw.line(
            [
                (width - hud_inset, height - hud_inset),
                (width - hud_inset, height - hud_inset - corner_len),
            ],
            fill=c_prim,
            width=4,
        )

        # 3. Top Status Header (Mobile / Shorts Camera Interface)
        header_y = 65 if height > width else 70
        is_rec_blink = (int(f / 15) % 2) == 0
        dot_color = (255, 40, 40) if is_rec_blink else (120, 20, 20)
        draw.ellipse(
            [hud_inset + 15, header_y - 6, hud_inset + 27, header_y + 6], fill=dot_color
        )
        draw.text(
            (hud_inset + 35, header_y),
            "REC",
            fill=colors["text"],
            font=font_ui,
            anchor="lm",
        )
        draw.text(
            (width - hud_inset - 15, header_y),
            "9:16 VERTICAL",
            fill=colors["primary"],
            font=font_ui,
            anchor="rm",
        )

        # 4. Center Category / Badge Pill
        badge_text = f"• {theme.upper()} •"
        badge_y = int(height * 0.35) if height > width else int(height * 0.30)
        pill_w = 140
        pill_h = 32
        pill_box = [
            (width // 2) - pill_w // 2,
            badge_y - pill_h // 2,
            (width // 2) + pill_w // 2,
            badge_y + pill_h // 2,
        ]
        draw.rounded_rectangle(
            pill_box, radius=16, outline=colors["primary"], width=2, fill=(20, 20, 35)
        )
        draw.text(
            (width // 2, badge_y),
            badge_text,
            fill=colors["accent"],
            font=font_badge,
            anchor="mm",
        )

        # 5. Dynamic Title with floating animation & shadow
        title_y = int(
            (height * 0.45 if height > width else height * 0.42)
            + 6 * math.sin(t * math.pi * 2)
        )
        shadow_offset = 3
        draw.multiline_text(
            (width // 2 + shadow_offset, title_y + shadow_offset),
            wrapped_title,
            fill=(0, 0, 0),
            font=font_title,
            anchor="mm",
            align="center",
            spacing=8,
        )
        draw.multiline_text(
            (width // 2, title_y),
            wrapped_title,
            fill=colors["text"],
            font=font_title,
            anchor="mm",
            align="center",
            spacing=8,
        )

        # 6. Description Subtitle
        if wrapped_desc:
            desc_y = int(height * 0.56 if height > width else height * 0.58)
            draw.multiline_text(
                (width // 2, desc_y),
                wrapped_desc,
                fill=colors["subtext"],
                font=font_subtitle,
                anchor="mm",
                align="center",
                spacing=6,
            )

        # 7. Animated Bottom Progress Bar
        bar_x1 = int(width * 0.15 if height > width else width * 0.25)
        bar_x2 = int(width * 0.85 if height > width else width * 0.75)
        bar_y = height - (130 if height > width else 70)
        bar_width = (bar_x2 - bar_x1) * t
        draw.rectangle([bar_x1, bar_y, bar_x2, bar_y + 6], fill=(50, 50, 70))
        draw.rectangle(
            [bar_x1, bar_y, bar_x1 + bar_width, bar_y + 6], fill=colors["primary"]
        )

        # 8. Brand Watermark
        brand_y = height - (85 if height > width else 95)
        draw.text(
            (width // 2, brand_y),
            "GAMERHEADS • 9:16 VERTICAL",
            fill=colors["primary"],
            font=font_ui,
            anchor="mm",
        )

        frames.append(np.array(frame_img))

    # Write MP4 via imageio libx264
    iio.imwrite(
        str(target_path),
        frames,
        fps=fps,
        codec="libx264",
    )

    if target_size_mb and target_size_mb > 0:
        _pad_mp4_file(target_path, float(target_size_mb))

    size_bytes = os.path.getsize(target_path)
    size_mb = size_bytes / (1024.0 * 1024.0)
    size_kb = size_bytes / 1024.0
    size_desc = (
        f"{size_mb:.1f} MB ({size_bytes:,} bytes)"
        if size_mb >= 1.0
        else f"{size_kb:.1f} KB"
    )

    # Save as an ADK session artifact so the chat UI can display the video
    if tool_context is not None:
        try:
            with open(target_path, "rb") as f:
                video_bytes = f.read()
            part = types.Part(
                inline_data=types.Blob(
                    mime_type="video/mp4",
                    data=video_bytes,
                )
            )
            await tool_context.save_artifact(output_filename, part)
        except Exception:
            pass

    bucket_name = os.getenv("ARTIFACT_BUCKET_NAME")
    storage_info = (
        f"- Cloud Storage: gs://{bucket_name}/{output_filename}\n"
        if bucket_name
        else ""
    )
    orientation_desc = "9:16 Vertical" if height > width else "16:9 Landscape"
    return (
        f"✅ Successfully generated MP4 video!\n"
        f"- Artifact: {output_filename}\n"
        f"{storage_info}"
        f"- Resolution: {width}x{height} ({orientation_desc}, H.264 @ {fps}fps)\n"
        f"- Duration: {duration}s ({n_frames} frames)\n"
        f"- File Size: {size_desc}\n"
        f"- Theme: {theme}\n"
        f"- Title: {title}"
    )
