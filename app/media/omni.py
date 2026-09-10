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
import logging
import os
import tempfile
import uuid
from typing import Any

import aiohttp
from dotenv import load_dotenv

from app.media.clips import get_ffmpeg_exe

load_dotenv()

logger = logging.getLogger(__name__)

OMNI_MODEL = os.getenv("OMNI_MODEL", "gemini-omni-1.1-flash-preview")
MAX_ATTEMPTS = 3
ATTEMPT_TIMEOUT_SECONDS = 180
# The generated clip is fetched separately from the interaction that produced it,
# so it gets its own budget rather than eating into the 180s render window.
DOWNLOAD_TIMEOUT_SECONDS = 300


class OmniRequestError(RuntimeError):
    """A rejection the API will give again. Raised past the retry loop, not into it."""


def strip_data_url(data: str) -> str:
    """Strips data URL header if present (e.g. 'data:image/png;base64,')."""
    if data.startswith("data:") and "," in data:
        return data.split(",", 1)[1]
    return data


def data_url_mime(data: str, default: str = "image/jpeg") -> str:
    """The mime type named in a data URL. Bare base64 has none, so it gets the default."""
    if not data.startswith("data:") or "," not in data:
        return default
    return data[5 : data.index(",")].split(";", 1)[0] or default


def find_video_in_interaction(
    interaction: dict[str, Any],
) -> tuple[str | None, str | None]:
    """Inspects all known interaction API response shapes for video content.

    Returns:
        (uri, base64_data)
    """
    output_video = interaction.get("output_video") or {}
    uri = (
        output_video.get("uri")
        or output_video.get("gcsUri")
        or output_video.get("gcs_uri")
    )
    b64 = (
        output_video.get("data")
        or output_video.get("bytesBase64Encoded")
        or output_video.get("bytes_base64_encoded")
    )

    if not uri and not b64:
        out = interaction.get("output") or {}
        uri = out.get("uri") or (out.get("video") or {}).get("uri")
        b64 = (out.get("video") or {}).get("data") or (out.get("video") or {}).get(
            "bytesBase64Encoded"
        )

    if not uri and not b64:
        cand_videos = (interaction.get("response") or {}).get("videos") or []
        if cand_videos:
            uri = cand_videos[0].get("gcsUri") or cand_videos[0].get("uri")
            b64 = cand_videos[0].get("bytesBase64Encoded")

    # steps[].content[]
    if not uri and not b64:
        for step in interaction.get("steps") or []:
            for item in step.get("content") or []:
                if item.get("type") == "video" or str(
                    item.get("mime_type", "")
                ).startswith("video/"):
                    uri = item.get("uri")
                    b64 = item.get("data")
                    if uri or b64:
                        break

    # outputs[]
    if not uri and not b64:
        for item in interaction.get("outputs") or []:
            if item.get("type") == "video" or str(item.get("mime_type", "")).startswith(
                "video/"
            ):
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
            "-y",
            "-loop",
            "1",
            "-i",
            img_temp,
            "-f",
            "lavfi",
            "-i",
            "anullsrc=channel_layout=stereo:sample_rate=44100",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-tune",
            "stillimage",
            "-pix_fmt",
            "yuv420p",
            "-vf",
            f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height}",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-t",
            str(duration_seconds),
            dest_path,
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(
                f"Synthetic clip generation failed: {stderr.decode('utf-8', errors='ignore')}"
            )
    finally:
        if os.path.exists(img_temp):
            os.remove(img_temp)


async def post_interaction(
    url: str, headers: dict[str, str], body: dict[str, Any]
) -> dict[str, Any]:
    """POSTs one interaction, retrying only what a retry can fix.

    Three attempts, 2s then 4s backoff, 180s per attempt. 5xx and 429 are the
    transient cases and are retried; a 4xx says the request itself is wrong, so
    it is raised straight out -- two more attempts would spend six minutes
    arriving at the same answer.

    Everything downstream of this call (parsing the response, downloading the
    clip, spotting a silent rejection) deliberately sits outside the loop: those
    failures are properties of the interaction that came back, and re-POSTing
    pays for another render to reproduce them.
    """
    timeout = aiohttp.ClientTimeout(total=ATTEMPT_TIMEOUT_SECONDS)
    last_error: Exception | None = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        logger.info("[omni] attempt %d/%d", attempt, MAX_ATTEMPTS)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, headers=headers, json=body) as resp:
                    if resp.status >= 500 or resp.status == 429:
                        last_error = RuntimeError(
                            f"Interactions API transient error ({resp.status}): "
                            f"{(await resp.text())[:500]}"
                        )
                        if attempt < MAX_ATTEMPTS:
                            await asyncio.sleep(attempt * 2)
                            continue
                        raise last_error
                    if not resp.ok:
                        raise OmniRequestError(
                            f"Interactions API request failed ({resp.status}): "
                            f"{(await resp.text())[:500]}"
                        )
                    return await resp.json()
        except OmniRequestError:
            raise
        except Exception as e:
            # Anything transport-shaped -- timeouts, resets, DNS -- is worth another go.
            last_error = e
            if attempt < MAX_ATTEMPTS:
                logger.warning(
                    "[omni] attempt %d failed (%s); retrying", attempt, e
                )
                await asyncio.sleep(attempt * 2)
                continue
            raise

    raise last_error or RuntimeError("Video generation failed after retries.")


async def omni_interaction(
    prompt: str,
    start_frame_base64: str,
    duration_seconds: int,
    aspect_ratio: str,
    dest_path: str,
    mock: bool = False,
    continuity: bool = False,
) -> dict:
    """Renders one continuous video clip using Omni Flash image-to-video.

    Args:
        prompt: Detailed micro-action, dialogue, and vocal cue directives.
        start_frame_base64: Golden Anchor avatar or previous clip's last frame.
        duration_seconds: Clip duration (strictly 3 to 10 seconds).
        aspect_ratio: '16:9' or '9:16'.
        dest_path: Target destination path for output MP4.
        mock: Force synthetic generation (used for tests or offline execution).
        continuity: True when the start frame came from the previous clip. Log only.

    Returns:
        Dict with interactionId and videoPath.
    """
    if mock or os.getenv("MOCK_OMNI") == "1":
        await generate_synthetic_clip(
            start_frame_base64, duration_seconds, aspect_ratio, dest_path
        )
        return {"interactionId": f"mock-{uuid.uuid4().hex[:8]}", "videoPath": dest_path}

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    project = os.getenv("GOOGLE_CLOUD_PROJECT")

    # No credentials is a misconfiguration, not a mode. Falling back to a
    # synthetic clip here would hand back a still frame of the avatar that looks
    # like a finished render, and nothing downstream can tell the difference.
    if not api_key and not project:
        raise RuntimeError(
            "Neither GEMINI_API_KEY nor GOOGLE_CLOUD_PROJECT is set -- set "
            "GOOGLE_CLOUD_PROJECT (with ADC) for Vertex, or GEMINI_API_KEY for AI "
            "Studio. Set MOCK_OMNI=1 to render synthetic clips offline instead."
        )

    clean_img = strip_data_url(start_frame_base64)
    mime_type = data_url_mime(start_frame_base64)

    bucket = os.getenv("ARTIFACT_BUCKET_NAME", "")
    response_format: dict[str, Any] = {
        "type": "video",
        "aspect_ratio": "9:16" if aspect_ratio == "9:16" else "16:9",
    }
    if bucket:
        response_format["delivery"] = "uri"
        response_format["gcs_uri"] = f"gs://{bucket}/omni/"
    elif api_key:
        response_format["delivery"] = "uri"
    else:
        response_format["delivery"] = "inline"

    token = None
    if api_key:
        url = f"https://generativelanguage.googleapis.com/v1beta/interactions?key={api_key}"
        headers = {"Content-Type": "application/json"}
    else:
        # Vertex AI ADC. The refresh is a blocking HTTP call to the metadata
        # server, so it goes to a thread rather than stalling the event loop.
        def _access_token() -> str:
            import google.auth
            import google.auth.transport.requests

            creds, _ = google.auth.default()
            creds.refresh(google.auth.transport.requests.Request())
            return creds.token

        token = await asyncio.to_thread(_access_token)
        url = f"https://aiplatform.googleapis.com/v1beta1/projects/{project}/locations/global/interactions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        }
        if project:
            headers["X-Goog-User-Project"] = project

    body = {
        "model": OMNI_MODEL,
        "input": [
            {"type": "text", "text": prompt},
            {"type": "image", "mime_type": mime_type, "data": clean_img},
        ],
        "response_format": response_format,
        "generation_config": {"video_config": {"task": "image_to_video"}},
    }

    logger.info(
        "[omni] generating clip (%s, image_to_video, %ss, %s, continuity=%s)",
        OMNI_MODEL,
        duration_seconds,
        aspect_ratio,
        continuity,
    )

    interaction = await post_interaction(url, headers, body)

    uri, b64_data = find_video_in_interaction(interaction)
    interaction_id = interaction.get("id") or str(uuid.uuid4())

    if b64_data:
        with open(dest_path, "wb") as f:
            f.write(base64.b64decode(strip_data_url(b64_data)))
        return {"interactionId": interaction_id, "videoPath": dest_path}

    if uri and uri.startswith("gs://"):
        def _download_gs() -> None:
            from google.cloud import storage

            client = storage.Client()
            storage.Blob.from_string(uri, client=client).download_to_filename(dest_path)

        await asyncio.to_thread(_download_gs)
        return {"interactionId": interaction_id, "videoPath": dest_path}

    if uri and uri.startswith("http"):
        dl_url = uri
        dl_headers: dict[str, str] = {}
        if api_key:
            if "key=" not in dl_url:
                dl_url += f"{'&' if '?' in dl_url else '?'}key={api_key}"
        elif token:
            dl_headers["Authorization"] = f"Bearer {token}"

        timeout = aiohttp.ClientTimeout(total=DOWNLOAD_TIMEOUT_SECONDS)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(dl_url, headers=dl_headers) as dl_resp:
                # A failed download used to fall through to the "no video data"
                # branch below, which named the wrong cause: the clip rendered
                # fine, we just could not fetch it.
                if not dl_resp.ok:
                    raise RuntimeError(
                        f"Downloading the clip failed: HTTP {dl_resp.status}"
                    )
                with open(dest_path, "wb") as f:
                    f.write(await dl_resp.read())
        return {"interactionId": interaction_id, "videoPath": dest_path}

    # A refused generation is not an error response: the API answers 200 with
    # status "completed", zero token counts and no output at all, which is
    # indistinguishable from success until you go looking for the video.
    spent = int((interaction.get("usage") or {}).get("total_input_tokens") or 0)
    logger.warning(
        "[omni] no video in the response (status=%s, input_tokens=%s, keys=[%s])",
        interaction.get("status", "-"),
        spent,
        ",".join(interaction.keys()),
    )
    if spent == 0:
        raise RuntimeError(
            "The clip generator accepted the request but produced nothing and billed zero input tokens, "
            "which is how it reports a silently rejected source image -- most often one showing a "
            "recognisable copyrighted character. Regenerate the avatar as an original design and retry."
        )

    raise RuntimeError("The clip generator returned no video.")
