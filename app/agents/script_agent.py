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

"""ScriptAgent: Specialist agent for gaming commentary scripts and timed shot lists."""

from __future__ import annotations

import json
import os
from typing import Any

import aiohttp
from dotenv import load_dotenv
from google import genai
from google.adk.agents import Agent
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.models import Gemini
from google.adk.tools import ToolContext
from google.genai import types
from pydantic import BaseModel, Field

from app.pipeline import evaluate_stage_status, record_artifact
from app.tools.spec_tools import update_script_spec

load_dotenv()

MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest")

# ============================================================================
# 1. Data Models & Clamping / Timeline Utilities
# ============================================================================


class Segment(BaseModel):
    """A single shot segment in the streamer reaction video."""

    id: int = Field(description="Segment sequence index starting from 1")
    startTime: str = Field(description="Display start timestamp, e.g. '00:00'")
    endTime: str = Field(description="Display end timestamp, e.g. '00:05'")
    duration: int = Field(description="Segment duration in seconds, strictly 3 to 10")
    prompt: str = Field(
        description="Streamer's physical micro-expression and action; 100% pure human action, no screen elements"
    )
    dialogue: str = Field(
        description="Spoken commentary with natural gamer cadence and expressive vocal tags like [Laughing]"
    )


def clamp_segments(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Clamps every segment duration into the 3-10s window that Omni renders."""
    clamped = []
    for index, segment in enumerate(raw):
        dur = segment.get("duration", 6)
        try:
            dur_int = round(float(dur))
        except (ValueError, TypeError):
            dur_int = 6
        clamped_dur = max(3, min(10, dur_int))

        clamped.append(
            {
                "id": segment.get("id", index + 1),
                "startTime": segment.get("startTime") or "00:00",
                "endTime": segment.get("endTime") or "00:00",
                "duration": clamped_dur,
                "prompt": segment.get("prompt") or "",
                "dialogue": segment.get("dialogue") or "",
            }
        )
    return clamped


def compute_timeline(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Calculates accumulated start and end seconds for each shot."""
    cursor = 0
    result = []
    for seg in segments:
        start = cursor
        cursor += seg["duration"]
        start_min, start_sec = divmod(start, 60)
        end_min, end_sec = divmod(cursor, 60)
        result.append(
            {
                **seg,
                "start_seconds": start,
                "end_seconds": cursor,
                "startTime": f"{start_min:02d}:{start_sec:02d}",
                "endTime": f"{end_min:02d}:{end_sec:02d}",
                "timeline_str": f"{start_min:02d}:{start_sec:02d} - {end_min:02d}:{end_sec:02d}",
            }
        )
    return result


# ============================================================================
# 2. Prompt Engineering & System Instructions
# ============================================================================

SCRIPT_SYSTEM_INSTRUCTION = """You are an expert gaming livestreamer director and scriptwriter.
CRITICAL DIALOGUE DURATION & WORD COUNT RULE:
Every segment's 'dialogue' MUST contain enough spoken words to match its 'duration' in seconds at a natural speaking rate.
- 3s segment: 4 to 5 words.
- 4s segment: 5 to 7 words.
- 5s segment: 7 to 10 words.
- 6s segment: 10 to 13 words.
- 7s segment: 13 to 16 words.
- 8s segment: 16 to 19 words.
- 9s segment: 19 to 22 words.
- 10s segment: 22 to 25 words.
NEVER produce 1-word or 2-word dialogue (e.g. "Nice!") for long multi-second clips. Always write full, engaging livestreamer commentary sentences.
"""

BASE_PERSONA = """
You are a top-tier, high-energy gaming livestreamer (Streamer).
You speak naturally, use gamer slang appropriately (but not cringey), and know how to retain viewers.
Your vibe is professional yet hype. You are NOT a generic AI assistant.

CRITICAL PRONOUN RULE: Always use gender-neutral pronouns ('they' or 'them') when referring to the streamer in all descriptions and prompts. Do not use 'he', 'she', 'him', 'his', or 'her'.
"""

STREAMER_RULES = """
CRITICAL DURATION & TIMELINE RULES:
1. **TOTAL DURATION**: The sum of all segment durations MUST EXACTLY EQUATE to the length of the uploaded gameplay video.
2. **SEGMENTATION**: Break the script into consecutive, natural scene beats of **3 to 10 seconds** each.
3. **STRICT SPOKEN WORD COUNT MATCHING SEGMENT DURATION**:
   - Streamer dialogue must be realistically paced so that the streamer speaks naturally across the full duration of the shot without cutting off or being silent.
   - You MUST adhere to these exact word targets based on each segment's duration:
     * 3-second segment: 4 to 5 spoken words.
     * 4-second segment: 5 to 7 spoken words.
     * 5-second segment: 7 to 10 spoken words.
     * 6-second segment: 10 to 13 spoken words.
     * 7-second segment: 13 to 16 spoken words.
     * 8-second segment: 16 to 19 spoken words.
     * 9-second segment: 19 to 22 spoken words.
     * 10-second segment: 22 to 25 spoken words.
   - CRITICAL NEGATIVE CONSTRAINT: DO NOT output 1-word or 2-word dialogue (e.g. "Nice!", "Let's go") for long 6-10 second clips! If a clip is 9 seconds long, the streamer MUST speak 19 to 22 words of full, continuous commentary sentences.
   - EXCLUDE VOCAL FX BRACKETS FROM WORD COUNT: Bracketed direction tags like "[Laughing]", "[Sharp gasp]", "[ASMR whisper]", "[Shouting excitedly]" are voice synthesis directives and are excluded from the spoken word count.
4. **TIMESTAMPS**: Calculate cumulative timestamps for each segment (e.g. "00:00", "00:06", "00:13").

VISUAL DESCRIPTION RULES (STREAMER ACTIONS & MICRO-EXPRESSIONS):
1. **STREAMER ACTION**: Must be EXTREMELY DETAILED (Micro-Expression Level).
   - Describe specific facial features: "Eyes wide open," "Jaw dropped," "Bit lip," "Eyebrows furrowed," "Grins broadly."
   - Describe body language: "Leans forward aggressively," "Throws head back in laughter," "Covers mouth in shock."

2. **PURE HUMAN ACTION (NO GAME/SCREEN ELEMENTS)**:
   - The streamer action description must be 100% about the human.
   - **NEVER** mention what is on the screen (e.g. DO NOT say "Reacts to explosion", "Looking at the dragon").
   - Instead use physical descriptions: "Reacts with shock", "Staring intensely ahead", "Wincing in pain".

DIALOGUE & AUDIO RULES:
1. **VOCAL FX (VFX)**: You MUST prefix dialogue with expressive vocal cues in brackets when appropriate:
   - Examples: "[Laughing] No way, they actually pulled that off!", "[Sharp gasp] Look at that health bar!", "[Shouting excitedly] Let's go!", "[ASMR whisper] Watch this sneak attack..."
   - This directs the AI model's native voice synthesis engine.

FORMATTING RULES:
1. Refer to the character as 'Streamer'. Use gender-neutral pronouns ('they'/'them') when referring to the streamer. Do not use 'he' or 'she'.

NEGATIVE CONSTRAINTS:
1. DO NOT describe the streamer turning the phone/screen towards camera.
2. DO NOT mention background music or non-vocal SFX.
3. Camera perspective remains completely static (locked tripod shot).
"""


def script_device_instruction(device: str) -> str:
    """Returns platform-specific body language constraints."""
    if device == "Mobile (Vertical)":
        return "EVERY 'prompt' MUST START WITH: \"Streamer holds phone VERTICALLY (Portrait) with both hands.\" followed by the action. Thumbs tapping/swiping."
    if device == "Mobile (Horizontal)":
        return "EVERY 'prompt' MUST START WITH: \"Streamer holds phone HORIZONTALLY (Landscape) with both hands.\" followed by the action. Thumbs tapping."
    if device == "PC":
        return "Ensure descriptions involve keyboard/mouse interaction on a desk."
    if device == "Console":
        return "Ensure descriptions involve holding a standard Gamepad/Controller."
    if device == "Hands-free (No device)":
        return "Ensure descriptions do NOT involve any interactions with devices (no phones, no controllers, no keyboards). Streamer is completely hands-free."
    return "Ensure descriptions involve keyboard/mouse interaction on a desk."


def build_script_prompt(
    title: str = "",
    cta: str = "",
    device: str = "PC",
    additional_instructions: str = "",
    search_grounding: bool = False,
    game_url: str = "",
    researched_facts: str = "",
    footage_url: str = "",
) -> str:
    device = device or "PC"
    title = title.strip()
    cta = cta.strip()
    notes = additional_instructions.strip()
    device_instruction = script_device_instruction(device)
    grounding_target = f'"{title}"' if title else "the game"

    if search_grounding and game_url:
        facts_extra = (
            f'\nResearched Authentic Facts:\n"""\n{researched_facts.strip()}\n"""\n'
            if researched_facts
            else ""
        )
        grounding_instruction = (
            f'3. **GOOGLE SEARCH GROUNDING ACTIVE**: You have Google Search grounding enabled for the official Game URL: "{game_url}". '
            f"Use the Google Search tool to look up details, features, launch dates, unique mechanics, platforms, pricing, or target audience for {grounding_target}. "
            f"You have the context and liberty to incorporate these researched facts to promote the game and sound like an authentic fan/expert, aligning your promotion naturally with the gameplay visuals.{facts_extra}"
        )
    else:
        grounding_instruction = "3. **NO SEARCH GROUNDING**: Do NOT use Google Search grounding. Restrict the streamer's commentary strictly to what is directly visible in the gameplay footage. Do not make up features or facts about the game that are not visible."

    video_instruction = (
        "4. **VIDEO SYNCHRONIZATION**: You have been provided with the gameplay video file. You MUST analyze the video to identify key events, actions, milestones, combat status, victories, or failures occurring at each timestamp. Your commentary [Streamer Dialogue] and physical reactions [Streamer Action] MUST synchronize directly and logically with these specific gameplay visuals in the video."
        if footage_url
        else ""
    )

    title_instruction = (
        f'5. **GAME TITLE INTRO**: The streamer should naturally mention the Game Title ("{title}") in the first shot/segment of the script to introduce the game at timestamp 00:00.'
        if title
        else '5. **GAME INTRO**: No game title was provided. The streamer must NOT invent a game name. Open the script by welcoming viewers to "this game" or by describing what is visible in the footage.'
    )

    cta_instruction = (
        f'6. **CALL TO ACTION (CTA) PLACEMENT**: The streamer **MUST** naturally deliver the Call to Action ("{cta}") in the final shot/segment of the script as the closing remark.'
        if cta
        else "6. **CALL TO ACTION (CTA) PLACEMENT**: Close the final shot with a short, natural call to action inviting viewers to play along. Do not invent links, prices, offers, or specific promotions."
    )

    user_instruction_rule = (
        f'7. **USER INSTRUCTIONS ADHERENCE**: You must strictly follow and incorporate the specific instructions regarding the streamer\'s tone, messaging, gaming style, persona, or any specific features/actions mentioned in the User Instructions: "{notes}".'
        if notes
        else ""
    )

    game_context = (
        f'- Game: "{title}"'
        if title
        else "- Game: not provided -- do not invent a name; describe what is visible in the footage"
    )
    cta_context = (
        f'- CTA: "{cta}"'
        if cta
        else "- CTA: not provided -- close with a short, natural call to action (no invented links, prices, or offers)"
    )

    return f"""
{BASE_PERSONA}

TASK: Create a synchronized gameplay commentary script.

PROJECT CONTEXT:
{game_context}
{cta_context}
- User Instructions (Style/Tone/Messaging/Streamer Persona/Additional notes): "{notes}"
- **Gaming Device (Selected by User)**: "{device}"

CRITICAL INSTRUCTION:
1. **DEVICE AUTHENTICITY**: The user has explicitly selected **{device}** as the platform.
   - {device_instruction}
2. Apply expressive vocal effects in brackets ([Laughing], [Shouting], [ASMR whisper], [Gasping]) to [Streamer Dialogue].
{grounding_instruction}
{video_instruction}
{title_instruction}
{cta_instruction}
{user_instruction_rule}

{STREAMER_RULES}
"""


# ============================================================================
# 3. Specialist Tools for ScriptAgent
# ============================================================================


async def research_game(
    tool_context: ToolContext,
    query: str | None = None,
) -> str:
    """Researches authentic game mechanics, lore, features, and terminology using Google Search.

    Call this tool when:
    - The user wants to explore what makes the game special (Exploratory / Research intent).
    - Google Search grounding is enabled (`searchGrounding=True`) and authentic facts need to be gathered before drafting the commentary script.

    Args:
        query: Optional specific search query or focus area (e.g. 'combat mechanics', 'lore details').
        tool_context: The ADK tool context.

    Returns:
        A summary of researched gameplay features, mechanics, and terminology saved to session state.
    """
    state = tool_context.state
    spec = state.get("spec", {})
    script_spec = spec.get("script", {})

    game = (script_spec.get("game") or "").strip()
    game_url = (script_spec.get("gameUrl") or "").strip()

    if not game and not query:
        return (
            "Cannot research game: No game title has been provided. "
            "Please specify the game name first or record it via 'update_script_spec'."
        )

    search_target = (
        f"{game} {query}".strip()
        if query and game
        else (
            query.strip()
            if query
            else f"{game} gameplay mechanics features unique terminology {game_url}".strip()
        )
    )

    client = genai.Client()
    prompt = (
        f"You are a gaming researcher. Conduct research on the game: '{game or query}'. "
        f"Target query / focus: {search_target}. "
        + (
            f"Official website or store URL: {game_url}. Prioritize verified information from this URL. "
            if game_url
            else ""
        )
        + "Summarize 3-5 authentic gameplay features, unique combat/play mechanics, notable updates, "
        "and authentic player community slang/terminology that a streamer would naturally mention in high-energy commentary. "
        "Keep it concise, factual, and punchy."
    )

    grounding_urls: list[str] = []
    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
            ),
        )
        researched_facts = response.text or ""
        if response.candidates and response.candidates[0].grounding_metadata:
            for chunk in (
                response.candidates[0].grounding_metadata.grounding_chunks or []
            ):
                if chunk.web and chunk.web.uri:
                    grounding_urls.append(chunk.web.uri)
    except Exception as e:
        return f"Error conducting game research via Google Search: {e}"

    if not researched_facts.strip():
        return "Google Search did not return any usable research facts. Please refine the query or game title."

    # Persist researched facts as an artifact deliverable in session state
    record_artifact(
        state,
        "research",
        {
            "game": game or query,
            "facts": researched_facts.strip(),
            "sources": grounding_urls,
        },
    )

    sources_str = (
        "\n\nSources:\n" + "\n".join(f"- {u}" for u in grounding_urls)
        if grounding_urls
        else ""
    )
    return (
        f"Successfully researched authentic gameplay facts for '{game or query}':\n\n"
        f"{researched_facts.strip()}"
        f"{sources_str}\n\n"
        "These authentic facts are now saved in session state and will be incorporated into commentary script generation."
    )


async def watch_gameplay_and_generate_script(tool_context: ToolContext) -> str:
    """Watches the gameplay footage registered in session state and writes the timed shot list.

    This tool reads the gameplay video (artifact or URL), sends it to the multimodal model,
    and produces a sequence of 3-to-10 second segments with exact duration, micro-expression
    actions, and synchronized dialogue.

    Call this tool when:
    - Initial script creation is needed after gameplay footage is ingested.
    - The user wants a complete creative rewrite of the script (e.g. changing style or persona).

    Returns:
        A formatted shot list detailing the timing, dialogue, and on-screen actions.
    """
    state = tool_context.state
    spec = state.get("spec", {})
    global_spec = spec.get("global", {})
    script_spec = spec.get("script", {})
    artifacts = state.get("artifacts", {})

    footage_target = global_spec.get("footageUrl")

    if not footage_target:
        return (
            "Cannot generate script: No gameplay footage has been provided yet. "
            "Writing a synchronized commentary script requires the gameplay clip to measure "
            "exact total duration, pace the scene beats, and match reactions with on-screen milestones. "
            "Please inform the Director (Coordinator) that the gameplay footage clip is required before the script can be drafted, "
            "so the Director can ask the user to upload or provide the link."
        )

    # 1. Retrieve the video bytes
    video_bytes: bytes | None = None
    mime_type = "video/mp4"

    # Try loading as session artifact first
    try:
        part = await tool_context.load_artifact(footage_target)
        if part and part.inline_data and part.inline_data.data:
            video_bytes = part.inline_data.data
            if part.inline_data.mime_type:
                mime_type = part.inline_data.mime_type
    except Exception:
        pass

    # If not in artifact store or loading failed, try fetching if it's a URL
    if not video_bytes and (
        footage_target.startswith("http://") or footage_target.startswith("https://")
    ):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(footage_target) as resp:
                    if resp.status == 200:
                        video_bytes = await resp.read()
                        content_type = resp.headers.get("Content-Type")
                        if content_type:
                            mime_type = content_type.split(";")[0].strip()
        except Exception as e:
            return f"Error downloading footage from {footage_target}: {e}"

    if not video_bytes:
        return (
            f"Error: Could not retrieve video data for '{footage_target}'. "
            "Please ensure the file was ingested as an artifact or is a valid URL."
        )

    # 2. Build prompt and check research prerequisites
    title = script_spec.get("game", "")
    cta = script_spec.get("cta", "")
    device = global_spec.get("gamingDevice", "PC")
    additional_notes = script_spec.get("additionalInstructions", "")
    search_grounding = script_spec.get("searchGrounding", False)
    game_url = script_spec.get("gameUrl", "")

    researched_facts = artifacts.get("research", {}).get("facts", "")

    if search_grounding and not researched_facts:
        target_name = title or "this game"
        return (
            f"Cannot generate script yet: Google Search grounding is enabled for '{target_name}', "
            "but authentic research facts have not been gathered yet. "
            "Please call the 'research_game' tool first to retrieve verified gameplay mechanics, "
            "lore, and features before generating the commentary script."
        )

    user_prompt = build_script_prompt(
        title=title,
        cta=cta,
        device=device,
        additional_instructions=additional_notes,
        search_grounding=search_grounding,
        game_url=game_url,
        researched_facts=researched_facts,
        footage_url=footage_target,
    )

    # 3. Call Gemini with video part and structured schema
    client = genai.Client()
    contents = [
        types.Part.from_bytes(data=video_bytes, mime_type=mime_type),
        user_prompt,
    ]

    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SCRIPT_SYSTEM_INSTRUCTION,
                response_mime_type="application/json",
                response_schema=list[Segment],
            ),
        )
        raw_text = response.text or "[]"
        raw_segments = json.loads(raw_text)
    except Exception as err:
        return f"Error executing script model: {err}"

    if not isinstance(raw_segments, list) or len(raw_segments) == 0:
        return "Error: Script model returned an empty segment list."

    clamped = clamp_segments(raw_segments)
    final_segments = compute_timeline(clamped)

    # 4. Save deliverables to state["artifacts"]["script"]
    record_artifact(
        state,
        "script",
        {
            "segments": final_segments,
            "total_duration": final_segments[-1]["end_seconds"]
            if final_segments
            else 0,
            "device": device,
        },
    )

    # 5. Format return summary & detect downstream impact
    total_dur = final_segments[-1]["end_seconds"] if final_segments else 0
    lines_summary = []
    for s in final_segments:
        lines_summary.append(
            f'Line {s["id"]} [{s["startTime"]} - {s["endTime"]} ({s["duration"]}s)]: "{s["dialogue"]}"'
        )

    return (
        f"Successfully generated script with {len(final_segments)} segments (Total: {total_dur}s):\n\n"
        + "\n".join(lines_summary)
    )


class LineEditItem(BaseModel):
    line: int = Field(description="Which line to change, numbered from 1")
    dialogue: str | None = Field(
        default=None,
        description=(
            "The new spoken commentary line, or None to keep unchanged. "
            "Must include expressive vocal tags in brackets (e.g. [Laughing], [Shouting]). "
            "Spoken word count must strictly match segment duration: "
            "3s: 4-5 words | 4s: 5-7 words | 5s: 7-10 words | 6s: 10-13 words | "
            "7s: 13-16 words | 8s: 16-19 words | 9s: 19-22 words | 10s: 22-25 words. "
            "Never output 1-2 word lines for multi-second clips."
        ),
    )
    on_screen: str | None = Field(
        default=None,
        description=(
            "The new visual streamer action, or None to keep unchanged. "
            "Must ALWAYS use gender-neutral pronouns ('they' / 'them'), never 'he' or 'she'. "
            "Describe ONLY pure human micro-expressions, posture, and hands; "
            "NEVER mention on-screen game elements, UI, dragons, or enemies."
        ),
    )


async def edit_script_lines(
    edits: list[LineEditItem],
    tool_context: ToolContext,
) -> str:
    """Updates specific commentary lines or on-screen actions in an existing script.

    Only updates the specified lines and preserves all other segments, timings, and durations.
    Automatically marks downstream deliverables (streamer_video, composite) OUT_OF_SYNC.

    Args:
        edits: List of LineEditItem specifying line number and new dialogue/action.
        tool_context: ADK ToolContext holding session state.

    Returns:
        Summary of modified lines with updated script overview.
    """
    state = tool_context.state
    artifacts = state.get("artifacts", {})
    script_artifact = artifacts.get("script")

    if not script_artifact or "segments" not in script_artifact:
        return "Error: There is no script to edit yet. Call watch_gameplay_and_generate_script first."

    segments = list(script_artifact.get("segments", []))
    seg_map = {s["id"]: i for i, s in enumerate(segments)}

    modified = []
    for edit in edits:
        if edit.line not in seg_map:
            continue
        idx = seg_map[edit.line]
        if edit.dialogue is not None:
            segments[idx]["dialogue"] = edit.dialogue
        if edit.on_screen is not None:
            segments[idx]["prompt"] = edit.on_screen
        modified.append(edit.line)

    # Write back
    if script_artifact:
        script_artifact["segments"] = segments
        record_artifact(state, "script", script_artifact)

    lines_summary = []
    for s in segments:
        marker = " (MODIFIED)" if s["id"] in modified else ""
        lines_summary.append(
            f'Line {s["id"]}{marker} [{s.get("startTime", "00:00")} - {s.get("endTime", "00:00")} ({s.get("duration", 6)}s)]: "{s.get("dialogue", "")}"'
        )

    return f"Successfully updated line(s) {modified}:\n\n" + "\n".join(lines_summary)


# ============================================================================
# 4. ScriptAgent Definition
# ============================================================================

SCRIPT_AGENT_INSTRUCTION = """You are the expert Gaming Scriptwriter & Cinematographer (ScriptAgent) in the GamerHeads studio.
Your purpose is creating and maintaining high-energy, synchronized gameplay commentary shot lists that perfectly match the creator's vision and production settings.

【CORE PRODUCTION PRINCIPLES】

1. QUERY TRIAGE & SPEC-FIRST (Check update_script_spec options BEFORE generating):
   - Whenever user input or instructions arrive, FIRST check if the creator is asking to modify, configure, or provide anything.
   - Check all available parameters in `update_script_spec`:
     * `game`: Game name/title
     * `gameUrl`: Official link or store page
     * `searchGrounding`: Enable/disable Google Search grounding (True/False)
     * `cta`: Call to action (e.g. subscribe, like, wishlist)
     * `additionalInstructions`: ANY creative requirement, tone, style, pacing, persona, commentary rules, story angle, or instructions
   - If the request contains or modifies ANY of these spec options:
     -> You MUST call `update_script_spec` FIRST to update the spec and persist it into session state.
     -> Only AFTER updating the spec, proceed to generation (`watch_gameplay_and_generate_script`).
   - Exception for Surgical Tweaks: If the creator is only asking to fine-tune specific existing lines (e.g. "把第2句改一下"), use `edit_script_lines` to modify those lines without changing the overall spec.

2. FOOTAGE ANCHOR (No video, no script):
   - Commentary is strictly synchronized to on-screen visual events. If gameplay footage (`footageUrl`) is missing, never fabricate shot lists out of thin air. Record provided info via `update_script_spec` and inform the Director that video footage is the prerequisite.

3. WORKFLOW SYNCHRONIZATION (Keep deliverables fresh & aligned):
   Your commentary script must always stay in sync with project settings:
   - Initial Draft: Ensure any spec parameters are recorded via `update_script_spec`, then call `watch_gameplay_and_generate_script`.
   - Explicit Rewrite / Settings Changed: When modifying settings via `update_script_spec`, if the tool reports that the script is Out of Sync (or status shows `OUT_OF_SYNC`), you MUST continue and regenerate the script via `watch_gameplay_and_generate_script` to realign with the new settings.

4. SURGICAL LINE EDITING (Protect established timings):
   When a script already exists and specific lines or wording need fine-tuning (e.g. "把第2句改一下" or autonomous polish), use `edit_script_lines` to preserve the rest of the shot list and timing. Never regenerate the whole video script for line-specific tweaks.

5. GROUNDING-FIRST (Research before drafting):
   Authentic commentary requires real gameplay facts. When Google Search grounding is enabled, ensure authentic game features and mechanics exist (`artifacts.research`). If missing, call `research_game` before generating the script.

6. PROACTIVE CREATIVE CONSULTING (Suggest enhancements beyond bare footage):
   While footage is the only hard prerequisite, a truly viral script thrives on context:
   - When the game title (`game`) is unspecified: draft the script based on visual cues, but proactively suggest that the Director check for the game name to unlock game-specific terminology.
   - When grounding is off or tone is unspecified: proactively suggest offering research on game mechanics or proposing distinct streamer personas (e.g. funny/trolling vs. esports tryhard).

7. SPECIALIST DELIVERY (Concise upstream reporting):
   You are an internal specialist reporting to the Director (main agent). Once all required production actions are complete and your job is finished, concisely synthesize the deliverable:
   - Provide ONLY the line timings (timestamps and duration) and spoken dialogue (`dialogue`) for each segment.
   - NEVER include physical actions, body movements, gestures, micro-expressions, or camera prompts (`prompt` / `on_screen`) in your report to the Director. All visual actions and framing prompts are already preserved in session state artifacts (`artifacts.script`) for downstream video generation.
   - Include any single high-value creative enhancement recommendation for the Director to pose to the creator.
"""


def script_agent_instruction(context: ReadonlyContext) -> str:
    state = context.state
    spec = state.get("spec", {})
    global_spec = spec.get("global", {})
    script_spec = spec.get("script", {})
    artifacts = state.get("artifacts", {})

    footage_url = global_spec.get("footageUrl", "")
    gaming_device = global_spec.get("gamingDevice", "PC")
    aspect_ratio = global_spec.get("aspectRatio", "16:9")
    game = script_spec.get("game", "")
    game_url = script_spec.get("gameUrl", "")
    search_grounding = script_spec.get("searchGrounding", False)
    cta = script_spec.get("cta", "")
    additional_instructions = script_spec.get("additionalInstructions", "")
    has_script = "script" in artifacts

    status_lines = [
        "【CURRENT PRODUCTION SPEC & SESSION STATE】",
        f"- Registered Gameplay Footage (footageUrl): {footage_url or 'None (Not provided yet)'}",
        f"- Registered Game Name: {game or 'None (Not specified yet)'}",
        f"- Gaming Platform: {gaming_device}",
        f"- Video Aspect Ratio: {aspect_ratio}",
        f"- Google Search Grounding: {'Enabled' if search_grounding else 'Disabled'}",
    ]
    if game_url:
        status_lines.append(f"- Official Game URL: {game_url}")
    if cta:
        status_lines.append(f"- Call to Action (CTA): {cta}")
    if additional_instructions:
        status_lines.append(
            f"- Creative Instructions / Tone: {additional_instructions}"
        )
    if has_script:
        script_artifact = artifacts.get("script", {})
        segments = script_artifact.get("segments", [])
        total_duration = script_artifact.get(
            "total_duration",
            segments[-1].get("end_seconds", 0) if segments else 0,
        )

        sc_eval = evaluate_stage_status("script", state)
        sync_tag = (
            f" [⚠️ OUT OF SYNC: {sc_eval['summary']}]"
            if sc_eval["status"] == "OUT_OF_SYNC"
            else " [Ready]"
        )
        status_lines.append(
            f"- Existing Script Deliverable: Present in session ({len(segments)} segments, {total_duration}s total duration){sync_tag}"
        )
        if segments:
            status_lines.append("  * Current Commentary Shot List:")
            for s in segments:
                status_lines.append(
                    f"    - Line {s.get('id')} [{s.get('startTime', '00:00')} - {s.get('endTime', '00:00')} ({s.get('duration', 6)}s)]:"
                )
                status_lines.append(f"      * Action: {s.get('prompt', '')}")
                status_lines.append(f'      * Dialogue: "{s.get("dialogue", "")}"')
    else:
        status_lines.append("- Existing Script Deliverable: None drafted yet")

    status_block = "\n".join(status_lines)
    return f"{SCRIPT_AGENT_INSTRUCTION}\n\n{status_block}"


script_agent = Agent(
    name="script_agent",
    model=Gemini(
        model=MODEL,
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    instruction=script_agent_instruction,
    tools=[
        update_script_spec,
        research_game,
        watch_gameplay_and_generate_script,
        edit_script_lines,
    ],
)
