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

import pytest
from google.adk.agents.invocation_context import InvocationContext
from google.adk.artifacts import GcsArtifactService, InMemoryArtifactService
from google.adk.plugins.save_files_as_artifacts_plugin import SaveFilesAsArtifactsPlugin
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types

from app.agent import app, root_agent
from app.app_utils import services


def test_app_and_agent_configuration():
    """Verify that the agent and app are configured with the artifact plugin and tool."""
    # Check plugin registered on app
    plugin_types = [type(p) for p in app.plugins]
    assert SaveFilesAsArtifactsPlugin in plugin_types
    from app.plugins.artifact_filter_plugin import DeliverableArtifactFilterPlugin

    assert DeliverableArtifactFilterPlugin in plugin_types

    # Check load_artifacts tool on root_agent
    tool_names = [
        t.name if hasattr(t, "name") else getattr(t, "__name__", "")
        for t in root_agent.tools
    ]
    assert "load_artifacts" in tool_names


def test_gcs_artifact_service_configured(monkeypatch):
    """Verify that get_artifact_service returns a GcsArtifactService instance when configured."""
    monkeypatch.setenv("ARTIFACT_BUCKET_NAME", "gamerheads3737")
    services.get_artifact_service.cache_clear()
    svc = services.get_artifact_service()
    assert isinstance(svc, GcsArtifactService)
    assert svc.bucket_name == "gamerheads3737"
    services.get_artifact_service.cache_clear()


def test_artifact_service_in_memory_fallback(monkeypatch):
    """Verify that get_artifact_service falls back to InMemoryArtifactService when no bucket is set."""
    monkeypatch.delenv("ARTIFACT_BUCKET_NAME", raising=False)
    services.get_artifact_service.cache_clear()
    svc = services.get_artifact_service()
    assert isinstance(svc, InMemoryArtifactService)
    # Restore cache for subsequent tests
    services.get_artifact_service.cache_clear()


@pytest.mark.asyncio
async def test_plugin_intercepts_inline_data():
    """Test that SaveFilesAsArtifactsPlugin intercepts inline_data and replaces with placeholder."""
    sess_svc = InMemorySessionService()
    session = await sess_svc.create_session(app_name="app", user_id="test-user")
    art_svc = InMemoryArtifactService()

    invocation_context = InvocationContext(
        invocation_id="test-inv-001",
        artifact_service=art_svc,
        session_service=sess_svc,
        session=session,
        agent=root_agent,
    )

    plugin = SaveFilesAsArtifactsPlugin()
    user_message = types.Content(
        role="user",
        parts=[
            types.Part(text="Check this document"),
            types.Part(
                inline_data=types.Blob(
                    data=b"hello world",
                    mime_type="text/plain",
                    display_name="hello.txt",
                )
            ),
        ],
    )

    result = await plugin.on_user_message_callback(
        invocation_context=invocation_context,
        user_message=user_message,
    )

    assert result is not None
    # Check placeholder in the message parts
    assert any(
        '[Uploaded Artifact: "hello.txt"]' in (p.text or "")
        for p in (result.parts or [])
    )

    # Check artifact was saved into artifact service
    saved_artifact = await art_svc.load_artifact(
        app_name="app",
        user_id="test-user",
        session_id=session.id,
        filename="hello.txt",
    )
    assert saved_artifact is not None
    assert saved_artifact.inline_data.data == b"hello world"
