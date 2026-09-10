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

"""Unit tests for URL ingestion and spec tools."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from google.adk.tools import ToolContext

from app.tools.ingest_tools import (
    determine_filename_and_mime,
    ingest_url_to_artifact,
    parse_google_drive_download_url,
)
from app.tools.spec_tools import (
    update_avatar_spec,
    update_global_spec,
    update_script_spec,
)


def test_parse_google_drive_download_url():
    # Test standard /file/d/ link
    url1 = "https://drive.google.com/file/d/1A2B3C4D5E6F7G8H9I0J/view?usp=sharing"
    direct1, file_id1 = parse_google_drive_download_url(url1)
    assert file_id1 == "1A2B3C4D5E6F7G8H9I0J"
    assert (
        direct1 == "https://drive.google.com/uc?export=download&id=1A2B3C4D5E6F7G8H9I0J"
    )

    # Test open?id= link
    url2 = "https://drive.google.com/open?id=my_special_file_123"
    direct2, file_id2 = parse_google_drive_download_url(url2)
    assert file_id2 == "my_special_file_123"
    assert (
        direct2 == "https://drive.google.com/uc?export=download&id=my_special_file_123"
    )

    # Test non-drive URL
    url3 = "https://example.com/videos/gameplay.mp4"
    direct3, file_id3 = parse_google_drive_download_url(url3)
    assert file_id3 is None
    assert direct3 == url3


def test_determine_filename_and_mime():
    # Content-Disposition header precedence
    fn, mime = determine_filename_and_mime(
        url="https://example.com/download?id=99",
        content_type_header="video/mp4",
        content_disposition='attachment; filename="apex_clip.mp4"',
    )
    assert "apex_clip.mp4" in fn
    assert mime == "video/mp4"

    # URL path deduction
    fn2, mime2 = determine_filename_and_mime(
        url="https://example.com/assets/streamer_cat.png",
        content_type_header="image/png",
        content_disposition=None,
    )
    assert "streamer_cat.png" in fn2
    assert mime2 == "image/png"


@pytest.mark.asyncio
async def test_ingest_url_to_artifact_success():
    fake_data = b"\x00\x00\x00\x20ftypmp42" + b"\x00" * 1024
    mock_context = MagicMock(spec=ToolContext)
    mock_context.save_artifact = AsyncMock()

    # Mock response
    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.headers = {
        "Content-Type": "video/mp4",
        "Content-Disposition": 'attachment; filename="test_video.mp4"',
    }

    async def iter_chunked(size):
        yield fake_data

    mock_response.content = MagicMock()
    mock_response.content.iter_chunked = iter_chunked

    # Mock get context manager
    mock_get_ctx = MagicMock()
    mock_get_ctx.__aenter__ = AsyncMock(return_value=mock_response)
    mock_get_ctx.__aexit__ = AsyncMock(return_value=None)

    # Mock session
    mock_session = MagicMock()
    mock_session.get.return_value = mock_get_ctx

    # Mock ClientSession context manager
    mock_client_ctx = MagicMock()
    mock_client_ctx.__aenter__ = AsyncMock(return_value=mock_session)
    mock_client_ctx.__aexit__ = AsyncMock(return_value=None)

    with patch("aiohttp.ClientSession", return_value=mock_client_ctx):
        result = await ingest_url_to_artifact(
            "https://example.com/test_video.mp4", mock_context
        )

    assert "Successfully ingested URL to artifact!" in result
    assert "test_video.mp4" in result
    mock_context.save_artifact.assert_called_once()
    saved_filename = mock_context.save_artifact.call_args[0][0]
    assert "test_video.mp4" in saved_filename


@pytest.mark.asyncio
async def test_ingest_url_to_artifact_permission_denied():
    mock_context = MagicMock(spec=ToolContext)
    mock_context.save_artifact = AsyncMock()

    mock_response = MagicMock()
    mock_response.status = 403

    mock_get_ctx = MagicMock()
    mock_get_ctx.__aenter__ = AsyncMock(return_value=mock_response)
    mock_get_ctx.__aexit__ = AsyncMock(return_value=None)

    mock_session = MagicMock()
    mock_session.get.return_value = mock_get_ctx

    mock_client_ctx = MagicMock()
    mock_client_ctx.__aenter__ = AsyncMock(return_value=mock_session)
    mock_client_ctx.__aexit__ = AsyncMock(return_value=None)

    with patch("aiohttp.ClientSession", return_value=mock_client_ctx):
        result = await ingest_url_to_artifact(
            "https://drive.google.com/file/d/secret_123/view", mock_context
        )

    assert "Access denied" in result
    mock_context.save_artifact.assert_not_called()


@pytest.mark.asyncio
async def test_domain_spec_updates():
    mock_context = MagicMock(spec=ToolContext)
    mock_context.state = {}

    # Initial spec update with footage (global)
    result = await update_global_spec(
        footageUrl="gameplay_01.mp4",
        tool_context=mock_context,
    )
    assert "Spec updated successfully" in result
    assert "Downstream Impact Detected" not in result
    assert mock_context.state["spec"]["global"]["footageUrl"] == "gameplay_01.mp4"

    # Script spec update
    result_script = await update_script_spec(
        game="Black Myth: Wukong",
        tool_context=mock_context,
    )
    assert "Spec updated successfully" in result_script
    assert mock_context.state["spec"]["script"]["game"] == "Black Myth: Wukong"
    assert "config" not in mock_context.state
    assert "script_state" not in mock_context.state
    assert "streamer_video_state" not in mock_context.state

    # Subsequent update with avatar reference image (avatar)
    result2 = await update_avatar_spec(
        referenceImageUrl="cyber_avatar.png",
        appearance="cyberpunk girl with blue hair",
        tool_context=mock_context,
    )
    assert "Spec updated successfully" in result2
    assert "Downstream Impact Detected" not in result2
    assert (
        mock_context.state["spec"]["avatar"]["referenceImageUrl"] == "cyber_avatar.png"
    )
    assert (
        mock_context.state["spec"]["avatar"]["appearance"]
        == "cyberpunk girl with blue hair"
    )
    assert "avatar_state" not in mock_context.state
    assert "config" not in mock_context.state


@pytest.mark.asyncio
async def test_domain_spec_triggers_downstream_impact():
    mock_context = MagicMock(spec=ToolContext)
    # Existing artifacts in state
    mock_context.state = {
        "spec": {
            "global": {
                "footageUrl": "old_footage.mp4",
            },
            "avatar": {
                "appearance": "cyberpunk girl",
            },
        },
        "artifacts": {
            "script": [{"id": 1, "dialogue": "Let's go!"}],
            "avatar": {"image": "avatar.png"},
            "streamer_video": {"url": "video.mp4"},
        },
    }

    # 1. Changing footageUrl should impact script and streamer_video, but NOT avatar
    result = await update_global_spec(
        footageUrl="new_footage.mp4",
        tool_context=mock_context,
    )
    assert "Spec updated successfully" in result
    assert "⚠️ Downstream Impact Detected" in result
    assert "- [script]" in result
    assert "- [streamer_video]" in result
    assert "- [avatar]" not in result

    # 2. Changing appearance should impact avatar and streamer_video, but NOT script
    result2 = await update_avatar_spec(
        appearance="retro pixel hero",
        tool_context=mock_context,
    )
    assert "Spec updated successfully" in result2
    assert "⚠️ Downstream Impact Detected" in result2
    assert "- [avatar]" in result2
    assert "- [streamer_video]" in result2
    assert "- [script]" not in result2

    # 3. Changing additionalInstructions should impact script AND streamer_video, but NOT avatar
    result3 = await update_script_spec(
        additionalInstructions="Make it hyper sarcastic and funny",
        tool_context=mock_context,
    )
    assert "Spec updated successfully" in result3
    assert "⚠️ Downstream Impact Detected" in result3
    assert "- [script]" in result3
    assert "- [streamer_video]" in result3
    assert "- [avatar]" not in result
    assert (
        mock_context.state["spec"]["script"]["additionalInstructions"]
        == "Make it hyper sarcastic and funny"
    )
    assert "additionalInstructions" not in mock_context.state["spec"]["global"]
    assert "config" not in mock_context.state


@pytest.mark.asyncio
async def test_domain_spec_adk_state_delta():
    """Verify that domain spec tools emit state_delta to ADK State object."""
    from google.adk.events import EventActions
    from google.adk.sessions.state import State

    actions = EventActions()
    session_data = {"spec": {"global": {"footageUrl": "old.mp4"}}}
    state = State(value=session_data, delta=actions.state_delta)
    mock_ctx = MagicMock(spec=ToolContext)
    mock_ctx.state = state
    mock_ctx.actions = actions

    res = await update_script_spec(game="Apex Legends", tool_context=mock_ctx)
    assert "Spec updated successfully" in res
    assert "spec" in actions.state_delta
    assert actions.state_delta["spec"]["script"]["game"] == "Apex Legends"
    assert actions.state_delta["spec"]["global"]["footageUrl"] == "old.mp4"


@pytest.mark.asyncio
async def test_update_global_spec():
    """Verify that update_global_spec updates only global spec."""
    from app.tools.spec_tools import update_global_spec

    mock_ctx = MagicMock(spec=ToolContext)
    mock_ctx.state = {"spec": {"global": {}}}
    res = await update_global_spec(
        footageUrl="clip.mp4",
        gamingDevice="Console",
        aspectRatio="9:16",
        tool_context=mock_ctx,
    )
    assert "Spec updated successfully" in res
    assert mock_ctx.state["spec"]["global"]["footageUrl"] == "clip.mp4"
    assert mock_ctx.state["spec"]["global"]["gamingDevice"] == "Console"
    assert mock_ctx.state["spec"]["global"]["aspectRatio"] == "9:16"
