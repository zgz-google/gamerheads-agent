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

"""Artifact Deliverable Filter Plugin for GamerHeads Production Pipeline.

Ensures only whitelisted deliverables with the 'output_' prefix (avatar portrait,
streamer reaction video, composite video) are synchronized to the Gemini Enterprise
frontend via event.actions.artifact_delta.

All other files (user chat uploads, URL ingests, temporary working files) remain
safely stored in ArtifactService (GCS) for agent analysis and post-production,
but are completely hidden from the Gemini Enterprise frontend UI.
"""

import logging
from typing import Optional, Sequence

from google.adk.agents.invocation_context import InvocationContext
from google.adk.events.event import Event
from google.adk.plugins.base_plugin import BasePlugin

logger = logging.getLogger(__name__)

DEFAULT_DELIVERABLE_PREFIXES: tuple[str, ...] = ("output_",)


class DeliverableArtifactFilterPlugin(BasePlugin):
    """ADK Plugin that intercepts events and strips non-deliverables from artifact_delta."""

    def __init__(
        self,
        allowed_prefixes: Sequence[str] = DEFAULT_DELIVERABLE_PREFIXES,
        name: str = "deliverable_artifact_filter_plugin",
    ):
        super().__init__(name)
        self.allowed_prefixes = tuple(allowed_prefixes)

    def is_deliverable(self, filename: str) -> bool:
        """Determines if the artifact filename is a user-facing deliverable."""
        return any(filename.startswith(prefix) for prefix in self.allowed_prefixes)

    async def on_event_callback(
        self, *, invocation_context: InvocationContext, event: Event
    ) -> Optional[Event]:
        """Intercepts all runner events before persistence and yielding.

        Filters event.actions.artifact_delta so only deliverables starting with
        an allowed prefix (default: 'output_') reach the frontend.
        """
        if not event.actions or not event.actions.artifact_delta:
            return None

        original_delta = event.actions.artifact_delta
        filtered_delta = {
            filename: version
            for filename, version in original_delta.items()
            if self.is_deliverable(filename)
        }

        if len(filtered_delta) != len(original_delta):
            hidden = set(original_delta.keys()) - set(filtered_delta.keys())
            logger.info(
                "Filtered out internal/user-uploaded artifacts from frontend sync: %s",
                hidden,
            )
            event.actions.artifact_delta = filtered_delta

        return event
