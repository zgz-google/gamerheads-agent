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

"""URL Ingestion Tool for downloading external files into ADK Artifact Store."""

import hashlib
import mimetypes
import os
import re
import urllib.parse

import aiohttp
from google.adk.tools import ToolContext
from google.genai import types

MAX_DOWNLOAD_SIZE_BYTES = 300 * 1024 * 1024  # 300 MB limit


def parse_google_drive_download_url(url: str) -> tuple[str, str | None]:
    """Parses Google Drive link and returns direct download URL and file ID.

    Supports patterns:
    - drive.google.com/file/d/<file_id>/view
    - drive.google.com/open?id=<file_id>
    - drive.google.com/uc?id=<file_id>
    """
    patterns = [
        r"drive\.google\.com/file/d/([a-zA-Z0-9_-]+)",
        r"drive\.google\.com/open\?id=([a-zA-Z0-9_-]+)",
        r"drive\.google\.com/uc\?(?:.*&)?id=([a-zA-Z0-9_-]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            file_id = match.group(1)
            direct_url = f"https://drive.google.com/uc?export=download&id={file_id}"
            return direct_url, file_id
    return url, None


def determine_filename_and_mime(
    url: str,
    content_type_header: str | None,
    content_disposition: str | None,
) -> tuple[str, str]:
    """Determines artifact filename and MIME type from HTTP headers and URL."""
    filename = None

    # Check Content-Disposition header
    if content_disposition:
        disp_match = re.search(
            r'filename\*?=(?:UTF-8\'\')?["\']?([^"\';]+)["\']?',
            content_disposition,
            re.IGNORECASE,
        )
        if disp_match:
            filename = urllib.parse.unquote(disp_match.group(1)).strip()

    # Fallback to URL path
    if not filename:
        parsed_path = urllib.parse.urlparse(url).path
        base_name = os.path.basename(parsed_path)
        if base_name and "." in base_name:
            filename = base_name

    # Determine mime type
    mime_type = None
    if content_type_header:
        mime_type = content_type_header.split(";")[0].strip().lower()

    if not mime_type or mime_type in ("application/octet-stream", "text/plain"):
        if filename:
            guessed, _ = mimetypes.guess_type(filename)
            if guessed:
                mime_type = guessed

    if not mime_type:
        mime_type = "video/mp4"

    # Default extension if missing
    ext = mimetypes.guess_extension(mime_type) or (
        ".mp4" if "video" in mime_type else ".png"
    )
    if ext == ".jpe":
        ext = ".jpg"

    url_hash = hashlib.md5(url.encode("utf-8")).hexdigest()[:8]
    if not filename:
        filename = f"asset_{url_hash}{ext}"
    else:
        # Sanitize filename
        safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in filename)
        if not safe_name.endswith(ext):
            safe_name = f"{safe_name}{ext}"
        filename = f"imported_{url_hash}_{safe_name}"

    return filename, mime_type


async def ingest_url_to_artifact(url: str, tool_context: ToolContext) -> str:
    """Downloads an external media file (HTTP/HTTPS or Google Drive) and saves it to the session Artifact Store.

    CRITICAL USAGE RULES:
    - ONLY call this tool if the user explicitly provided a real URL or Google Drive link in their message.
    - NEVER invent, guess, or hallucinate URLs (e.g. placeholder domains, dummy URLs, or fake Google Drive IDs).
    - If the user asks you to look at, review, or evaluate an image or video but did not provide a URL or file, DO NOT call this tool; ask the user to provide the link or upload the file.
    - After this tool successfully saves the artifact, use `load_artifacts` to inspect it or `update_spec` to register its role.

    Args:
        url: The actual external URL or Google Drive link explicitly provided by the user. Do NOT invent or pass a hallucinated URL.

    Returns:
        A status string detailing the saved artifact name, MIME type, and size in megabytes.
    """
    clean_url = url.strip()
    is_drive = "drive.google.com" in clean_url
    download_url, file_id = parse_google_drive_download_url(clean_url)

    if is_drive and not file_id:
        return (
            "Error: Invalid Google Drive link format. Could not extract file ID. "
            "Please ensure the link is in the format 'https://drive.google.com/file/d/<FILE_ID>/view'."
        )

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }

    try:
        timeout = aiohttp.ClientTimeout(total=120)
        async with aiohttp.ClientSession(headers=headers, timeout=timeout) as session:
            async with session.get(download_url, allow_redirects=True) as response:
                if response.status == 403 or response.status == 401:
                    return (
                        f"Error: Access denied (HTTP {response.status}). "
                        "If using Google Drive, please check permissions and set sharing to "
                        "'Anyone with the link can view'."
                    )
                if response.status == 404:
                    return "Error: File not found at URL (HTTP 404). Please verify the link."
                if response.status != 200:
                    return f"Error: Download failed with HTTP status {response.status}."

                # Handle Google Drive virus scan warning page for large files
                content_type = response.headers.get("Content-Type", "")
                if is_drive and "text/html" in content_type:
                    html_text = await response.text()
                    confirm_match = re.search(r"confirm=([0-9A-Za-z_-]+)", html_text)
                    if confirm_match and file_id:
                        confirm_token = confirm_match.group(1)
                        confirm_url = f"https://drive.google.com/uc?export=download&id={file_id}&confirm={confirm_token}"
                        async with session.get(
                            confirm_url, allow_redirects=True
                        ) as conf_resp:
                            if conf_resp.status == 200:
                                response = conf_resp
                            else:
                                return "Error: Could not bypass Google Drive large-file confirmation page."

                content_disposition = response.headers.get("Content-Disposition")
                filename, mime_type = determine_filename_and_mime(
                    clean_url, response.headers.get("Content-Type"), content_disposition
                )

                # Check Content-Length if available
                content_length = response.headers.get("Content-Length")
                if content_length and int(content_length) > MAX_DOWNLOAD_SIZE_BYTES:
                    limit_mb = MAX_DOWNLOAD_SIZE_BYTES / (1024 * 1024)
                    return f"Error: File size ({int(content_length) / (1024 * 1024):.1f}MB) exceeds maximum limit of {limit_mb:.0f}MB."

                chunks = []
                total_bytes = 0
                async for chunk in response.content.iter_chunked(1024 * 1024):
                    total_bytes += len(chunk)
                    if total_bytes > MAX_DOWNLOAD_SIZE_BYTES:
                        limit_mb = MAX_DOWNLOAD_SIZE_BYTES / (1024 * 1024)
                        return f"Error: File download aborted because it exceeded {limit_mb:.0f}MB."
                    chunks.append(chunk)

                file_data = b"".join(chunks)

        # Save into ADK Artifact Store
        part = types.Part(
            inline_data=types.Blob(
                mime_type=mime_type,
                data=file_data,
            )
        )
        await tool_context.save_artifact(filename, part)

        size_mb = total_bytes / (1024.0 * 1024.0)
        return (
            f"Successfully ingested URL to artifact!\n"
            f"- Artifact Name: {filename}\n"
            f"- MIME Type: {mime_type}\n"
            f"- Size: {size_mb:.2f} MB\n"
            f"You can now call `load_artifacts(artifact_names=['{filename}'])` to inspect its visual contents, "
            f"and call `update_spec` to register its role (e.g. footageUrl='{filename}' or referenceImageUrl='{filename}')."
        )

    except aiohttp.ClientConnectorError as e:
        return f"Error: Could not connect to host: {e}"
    except Exception as e:
        return f"Error downloading URL: {e!s}"
