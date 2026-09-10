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

"""Omni Flash interaction client for Image-to-Video generation."""

from __future__ import annotations

import asyncio
import base64
import os
import tempfile
import uuid
from typing import Any

import aiohttp
from dotenv import load_dotenv

from app.media.clips import get_ffmpeg_exe

load_dotenv()

OMNI_MODEL = os.getenv("OMNI_MODEL", "gemini-omni-1.1-flash-preview")
MAX_ATTEMPTS = 3
ATTEMPT_TIMEOUT_SECONDS = 180


def strip_data_url(data: str) -> str:
    """Strips data URL header if present (e.g. 'data:image/png;base64,')."""
    if "," in data and "data:" in data[:30]:
        return data.split(",", 1)[1]
    return data


def find_video_in_interaction(interaction: dict[str, Any]) -> tuple[str | None, str | None]:
    """Inspects all known interaction API response shapes for video content.

    Returns:
        (uri, base64_data)
    """
    output_video = interaction.get("output_video") or {}
    uri = output_video.get("uri") or output_video.get("gcsUri") or output_video.get("gcs_uri")
    b64 = output_video.get("data") or output_video.get("bytesBase64Encoded") or output_video.get("bytes_base64_encoded")

    if not uri and not b64:
        out = interaction.get("output") or {}
        uri = out.get("uri") or (out.get("video") or {}).get("uri")
        b64 = (out.get("video") or {}).get("data") or (out.get("video") or {}).get("bytesBase64Encoded")

    if not uri and not b64:
        cand_videos = (interaction.get("response") or {}).get("videos") or []
        if cand_videos:
            uri = cand_videos[0].get("gcsUri") or cand_videos[0].get("uri")
            b64 = cand_videos[0].get("bytesBase64Encoded")

    # steps[].content[]
    if not uri and not b64:
        for step in interaction.get("steps") or []:
            for item in step.get("content") or []:
                if item.get("type") == "video" or str(item.get("mime_type", "")).startswith("video/"):
                    uri = item.get("uri")
                    b64 = item.get("data")
                    if uri or b64:
                        break

    # outputs[]
    if not uri and not b64:
        for item in interaction.get("outputs") or []:
            if item.get("type") == "video" or str(item.get("mime_type", "")).startswith("video/"):
                uri = item.get("uri")
                b64 = item.get("data")
                if uri or b64:
                    break

    return uri, b64


async def generate_synthetic_clip(
    start_frame_base64: str,
    duration_seconds: int,
    aspect_ratio: str,
    dest_path: str,
) -> None:
    """Generates a synthetic normalized video clip from a starting frame for testing or fallback."""
    is_vertical = aspect_ratio == "9:16"
    width = 1080 if is_vertical else 1920
    height = 1920 if is_vertical else 1080

    clean_b64 = strip_data_url(start_frame_base64)
    frame_bytes = base64.b64decode(clean_b64)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(frame_bytes)
        img_temp = f.name

    try:
        ffmpeg = get_ffmpeg_exe()
        cmd = [
            ffmpeg,
            "-loop", "1",
            "-i", img_temp,
            "-f", "lavfi",
            "-i", "anullsrc=r=44100:cl=stereo",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-tune", "stillimage",
            "-pix_fmt", "yuv420p",
            "-vf", f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height}",
            "-c:a", "aac",
            "-t", str(duration_seconds),
            dest_path,
            "-y",
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"Synthetic clip generation failed: {stderr.decode('utf-8', errors='ignore')}")
    finally:
        if os.path.exists(img_temp):
            os.remove(img_temp)


async def omni_interaction(
    prompt: str,
    start_frame_base64: str,
    duration_seconds: int,
    aspect_ratio: str,
    dest_path: str,
    mock: bool = False,
) -> dict:
    """Renders one continuous video clip using Omni Flash image-to-video.

    Args:
        prompt: Detailed micro-action, dialogue, and vocal cue directives.
        start_frame_base64: Golden Anchor avatar or previous clip's last frame.
        duration_seconds: Clip duration (strictly 3 to 10 seconds).
        aspect_ratio: '16:9' or '9:16'.
        dest_path: Target destination path for output MP4.
        mock: Force synthetic generation (used for tests or offline execution).

    Returns:
        Dict with interaction_id and dest_path.
    """
    if mock or os.getenv("MOCK_OMNI") == "1":
        await generate_synthetic_clip(start_frame_base64, duration_seconds, aspect_ratio, dest_path)
        return {"interactionId": f"mock-{uuid.uuid4().hex[:8]}", "videoPath": dest_path}

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    project = os.getenv("GOOGLE_CLOUD_PROJECT")

    # If neither key nor project is configured, gracefully fall back to synthetic clip
    if not api_key and not project:
        await generate_synthetic_clip(start_frame_base64, duration_seconds, aspect_ratio, dest_path)
        return {"interactionId": f"mock-{uuid.uuid4().hex[:8]}", "videoPath": dest_path}

    clean_img = strip_data_url(start_frame_base64)
    mime_type = "image/png"
    if start_frame_base64.startswith("data:image/jpeg"):
        mime_type = "image/jpeg"

    if api_key:
        url = f"https://generativelanguage.googleapis.com/v1beta/interactions?key={api_key}"
        headers = {"Content-Type": "application/json"}
    else:
        # Vertex AI ADC
        import google.auth
        import google.auth.transport.requests
        creds, _ = google.auth.default()
        auth_req = google.auth.transport.requests.Request()
        creds.refresh(auth_req)
        url = f"https://aiplatform.googleapis.com/v1beta1/projects/{project}/locations/global/interactions"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {creds.token}"}
        if project:
            headers["X-Goog-User-Project"] = project

    body = {
        "model": os.getenv("OMNI_MODEL", OMNI_MODEL),
        "input": [
            {"type": "text", "text": prompt},
            {"type": "image", "mime_type": mime_type, "data": clean_img},
        ],
        "response_format": {
            "type": "video",
            "aspect_ratio": "9:16" if aspect_ratio == "9:16" else "16:9",
            "delivery": "inline",
        },
        "generation_config": {"video_config": {"task": "image_to_video"}},
    }

    last_error = None
    timeout = aiohttp.ClientTimeout(total=ATTEMPT_TIMEOUT_SECONDS)

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, headers=headers, json=body) as resp:
                    if resp.status >= 500 or resp.status == 429:
                        last_error = RuntimeError(f"Omni transient error ({resp.status}): {await resp.text()}")
                        if attempt < MAX_ATTEMPTS:
                            await asyncio.sleep(attempt * 2)
                            continue
                        raise last_error
                    if not resp.ok:
                        raise RuntimeError(f"Omni request failed ({resp.status}): {await resp.text()}")
                    interaction = await resp.json()

                    uri, b64_data = find_video_in_interaction(interaction)
                    interaction_id = interaction.get("id") or str(uuid.uuid4())

                    if b64_data:
                        v_bytes = base64.b64decode(strip_data_url(b64_data))
                        with open(dest_path, "wb") as f:
                            f.write(v_bytes)
                        return {"interactionId": interaction_id, "videoPath": dest_path}

                    if uri and uri.startswith("http"):
                        async with session.get(uri) as dl_resp:
                            if dl_resp.ok:
                                with open(dest_path, "wb") as f:
                                    f.write(await dl_resp.read())
                                return {"interactionId": interaction_id, "videoPath": dest_path}

                    raise RuntimeError("Omni API response contained no video data.")

        except Exception as e:
            last_error = e
            if attempt < MAX_ATTEMPTS:
                await asyncio.sleep(attempt * 2)
                continue
            break

    raise RuntimeError(f"Omni generation failed after {MAX_ATTEMPTS} attempts: {last_error}")
