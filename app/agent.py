# ruff: noqa
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

import os
from dotenv import load_dotenv

load_dotenv()

from google.adk.agents import Agent
from google.adk.apps import App
from google.adk.models import Gemini
from google.genai import types

from google.adk.plugins.save_files_as_artifacts_plugin import SaveFilesAsArtifactsPlugin
from google.adk.tools import load_artifacts

from app.video_tool import create_mp4_video

MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest")


root_agent = Agent(
    name="gamerheads_app",
    model=Gemini(
        model=MODEL,
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    instruction=(
        "You are GamerHeads AI assistant, an expert AI assistant capable of answering questions, "
        "analyzing user-uploaded documents and files, and generating MP4 videos. "
        "When the user uploads files or asks about uploaded artifacts, use the `load_artifacts` tool "
        "to inspect and analyze their contents before answering. "
        "Whenever the user requests to generate, create, make, or output "
        "a video (MP4), call the `create_mp4_video` tool with suitable title, description, and theme. "
        "If the user specifies a desired file size (e.g. 60MB, large file test), pass `target_size_mb`. "
        "The generated videos are in vertical (9:16, 720x1280) format, ideal for mobile, TikTok, Reels, and Shorts. "
        "After generating a video, inform the user with the video details (artifact name, storage details, size, resolution) "
        "and that they can view and download it directly in the chat and Artifacts tab."
    ),
    tools=[create_mp4_video, load_artifacts],
)

app = App(
    root_agent=root_agent,
    name="app",
    plugins=[SaveFilesAsArtifactsPlugin()],
)
