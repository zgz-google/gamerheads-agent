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

"""Media utilities for video frame extraction, normalization, concatenation, and compositing."""

from app.media.clips import (
    compress_video,
    extract_last_frame,
    get_ffmpeg_exe,
    has_audio_track,
    normalize_clip,
)
from app.media.composite import composite_streamer_over_gameplay
from app.media.omni import omni_interaction
from app.media.stitch import concat_clips
from app.media.subtitles import build_ass_from_segments

__all__ = [
    "build_ass_from_segments",
    "composite_streamer_over_gameplay",
    "compress_video",
    "concat_clips",
    "extract_last_frame",
    "get_ffmpeg_exe",
    "has_audio_track",
    "normalize_clip",
    "omni_interaction",
]
