# GamerHeads User Guide

## Overview
**GamerHeads Gameplay Reaction Director** is an intelligent AIGC creative director agent that transforms raw gameplay footage into polished, broadcast-grade streamer reaction videos. By treating the Agent like an experienced creative director in a professional production studio, users can produce end-to-end reaction videos in minutes—complete with custom virtual streamer personas, timed commentary scripts, lip-synced reaction synthesis, and customizable picture-in-picture (PIP) / split-screen post-production. 

This enables game creators, esports teams, and studios to produce high-engagement gaming reaction content at scale without needing expensive camera setups, motion capture equipment, or tedious manual video editing.

---

## What it does
GamerHeads covers the complete end-to-end lifecycle of gameplay reaction video production:

1. **Footage & Concept Ingestion**:
   - Ingests raw gameplay clips via direct video URLs, Google Drive links, or direct uploads.
   - Automatically detects video orientation (16:9 widescreen or 9:16 vertical shorts) and gaming platforms (PC, PlayStation, Xbox, Nintendo Switch, Steam Deck, or Mobile).

2. **Intelligent Commentary Scripting**:
   - Analyzes gameplay pacing, key action moments, clutch plays, and boss fights.
   - Integrates Google Search Grounding to fetch game lore, tactical callouts, and trending community memes into a timed, shot-by-shot commentary script.

3. **Streamer Likeness Design (The Golden Anchor Avatar)**:
   - Configures streamer appearance, apparel, expression, streaming room setting (cyberpunk neon, cozy retro bedroom, modern esports studio), and handheld gaming devices.
   - Generates a high-fidelity master portrait ("Golden Anchor") via Nano Banana to anchor visual identity and consistency.

4. **Lip-Synced Reaction Video Synthesis**:
   - Conditions multi-segment video generation on the Golden Anchor portrait and dialogue lines.
   - Preserves visual continuity across consecutive clips while synthesizing natural facial expressions, head movement, and lip-sync speech.

5. **Sub-Second Post-Production & Video Compositing**:
   - Rapidly composites the reaction video over the gameplay footage using FFmpeg (~3 seconds).
   - Supports customizable layouts (PIP in any corner, top/bottom split, side-by-side), independent multi-track audio balancing (gameplay volume vs. streamer voice), and styled on-screen subtitles.

---

## Who it's for
- **Gaming Content Creators & Streamers**: Rapidly creating reaction videos, highlight commentaries, and viral TikTok / YouTube Shorts without filming on camera or complex editing.
- **Game Studios & Publishers**: Generating automated, localized promotional reaction trailers, gameplay feature reveals, and social teasers across multiple languages and regions.
- **Esports & Gaming Communities**: Producing tournament recap reactions, tactical play-by-play breakdowns, and meme-filled highlight clips.
- **Virtual Streamer (VTuber) Producers & Agencies**: Building and scaling unique virtual creator personas with consistent aesthetics across extensive gaming libraries.
- **Social Media Marketing Agencies**: Batch-producing high-engagement 9:16 vertical short-form reaction videos tailored for TikTok, Instagram Reels, and YouTube Shorts.

---

## How to use

### Scenario 0: End-to-End Gameplay Reaction (Real Case)
> *Provide gameplay footage with creative preferences, and let the Director manage script creation, avatar generation, reaction synthesis, and final compositing step by step.*

**Prompt:**
```text
帮我用这段艾尔登法环 Boss 战的游戏视频做一条主播反应视频：
https://drive.google.com/file/d/1vCklWi4SfRt2k6wl3TEuXBmLzNfY9spP/view?usp=drive_link

主播设定为一个活泼幽默、穿着连帽衫的硬核女玩家，房间要有赛博朋克霓虹灯电竞风，解说语气要充满激情，突出打 Boss 时的紧张和最后残血反杀的兴奋！
```

---

### Scenario 1: Handheld Console Streamer with Thematic Vibe
> *Create a reaction video with a specific gaming device (e.g., Nintendo Switch / Steam Deck) and tailored studio aesthetic.*

**Prompt:**
```text
Create a high-energy reaction video for this Monster Hunter Wilds gameplay clip:
Criteria / Gameplay Footage: [Google Drive URL]

Streamer Persona:
- A charismatic streamer wearing cat-ear headphones and an oversized hoodie, holding a Steam Deck.
- Studio Setting: A cozy, rainy-night Tokyo loft with warm ambient lighting and anime figurines in the background.
- Commentary Vibe: Hype, funny, and reacting with shock at the monster's ultimate attack.
```

---

### Scenario 2: 9:16 Vertical Mobile Game Shorts & Layout Customization
> *Transform mobile gameplay into vertical short-form content with custom PIP placement and audio balancing.*

**Prompt:**
```text
Here is my mobile battle royale gameplay clip: [Google Drive URL]

I want a 9:16 vertical short for TikTok:
1. Streamer Persona: Energetic mobile gaming creator holding a smartphone.
2. Layout: Picture-in-picture in the bottom-right corner.
3. Audio Balance: Set gameplay volume to 35% and streamer commentary volume to 100% so the speech is crystal clear.
4. Subtitles: Add bold, high-contrast on-screen subtitles.
```

---

### Scenario 3: Interactive Revision & Rapid Post-Production Rework
> *Fine-tune commentary lines or adjust video layout without re-rendering expensive video assets.*

**Prompt:**
```text
The commentary script looks great, but let's change line 3 to: "No way they survived that combo with 1 HP left!"

Also, for the final composite, please move the streamer overlay from bottom-right to top-left so it doesn't block the skill cooldown icons.
```

---

### Scenario 4: Game Lore Exploration & Deep-Dive Commentary
> *Explore lore, boss mechanics, or trending community memes first before committing to the script.*

**Prompt:**
```text
先帮我查查《黑神话：悟空》中这个 Boss 的背景故事和社区玩家常见的梗，结合这些细节帮我策划一段富有深度的硬核解说脚本。

游戏视频链接: [Google Drive URL]
```

---

## Under the hood
- **Architecture**: Diamond DAG (Directed Acyclic Graph) with strict separation between input specifications (`state.spec`) and generated deliverables (`state.artifacts`), featuring real-time state tracking and automatic downstream impact analysis (`detect_impact`).
- **AI Models & Media Engines**:
  - **Gemini 3.8 Flash**: Multi-modal gameplay video comprehension, pacing analysis, and search-grounded commentary scriptwriting.
  - **Nano Banana**: High-fidelity Golden Anchor streamer portrait generation conditioned on streamer appearance, device, and room environment.
  - **Omni Video Generation**: Multi-segment image-to-video generation maintaining frame-to-frame visual continuity and lip-sync alignment.
  - **FFmpeg Engine**: Sub-second deterministic audio balancing, PIP / split-screen layout compositing, and ASS subtitle burning.
- **Framework & Deployment**:
  - Built with **Google ADK (Agent Development Kit)** and managed via `agents-cli`.
  - Production deployment to **Google Cloud Vertex AI Agent Runtime (Reasoning Engine)** with automated registration to **Gemini Enterprise**.

See repository and documentation for more details:
- [GamerHeads Agent](https://github.com/zgz-google/gamerheads-agent)
