# OPC AI Video Studio — Architecture

## One-Person Company Multi-Agent System

**Concept:** One Human + Multiple AI Agents = One-Person Video Studio
**Based on:** Prof. Yong Zhang's Multi-Agent Collaboration & OPC lecture, Shenzhen University

---

## 1. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Web Frontend (Browser)                      │
│            upload · progress(SSE) · review · gallery · export   │
└──────────────────────────────┬──────────────────────────────────┘
                               │ HTTP / SSE
┌──────────────────────────────▼──────────────────────────────────┐
│                   FastAPI Layer (api/server.py)                  │
│                                                                 │
│  POST /upload    → Accept video file, save to input/             │
│  POST /run       → Start pipeline (video_id + config)           │
│  GET  /status    → Current pipeline state                        │
│  GET  /sse       → Real-time progress (Server-Sent Events)       │
│  GET  /clips     → List generated clips + metadata               │
│  POST /approve   → Human approves clips at review gate           │
│  GET  /export    → Download manifest / output package            │
│  GET  /output/*  → Static file serving (clips, thumbnails, etc) │
└──────────────────────────────┬──────────────────────────────────┘
                               │ calls
┌──────────────────────────────▼──────────────────────────────────┐
│            ManagerAgent (agents/manager_agent.py)                 │
│            ── Supervisory (Hierarchical) Topology ──             │
│                                                                 │
│  • Receives request, decides agent order (from config.yaml)     │
│  • Routes work to each agent via agent.run(pipeline_data)       │
│  • Retry once on failure, then escalate/skip                    │
│  • Pauses after Clip stage for human review                     │
│  • Streams progress events (stage_start, clip_found, etc.)      │
│  • Aggregates final output + manifest.json                      │
└──────────────────────────────┬──────────────────────────────────┘
                               │ delegates to
┌──────────────────────────────▼──────────────────────────────────┐
│                   AI Agent Team (8 Specialists)                  │
│                                                                 │
│  ┌──────────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │  Content     │→ │  Clip    │  │ Subtitle │  │  Audio   │   │
│  │  Analyzer    │  │  Expert  │  │  Expert  │  │  Expert  │   │
│  └──────────────┘  └────┬─────┘  └──────────┘  └──────────┘   │
│                         │                                       │
│                    [PAUSE]                                      │
│                   Human Review                                  │
│                   (approve/drop/edit clips)                     │
│                         │                                       │
│                         ▼                                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │Thumbnail │  │  Social  │  │  Format  │  │ Quality  │       │
│  │Generator │  │  Media   │  │  Adapter │  │ Reviewer │       │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. WorkBuddy Expert Mapping

The studio's agent team maps to WorkBuddy expert roles, demonstrating alignment
with the lecture's "AI digital employees" concept and the project brief's
recommendation to use WorkBuddy as the multi-agent platform.

| Studio Agent | WorkBuddy Expert Role | Responsibility | Lecture Concept |
|-------------|----------------------|----------------|-----------------|
| **ManagerAgent** | **Orchestrator / Project Manager** | Coordinates the team, routes work, handles failures, aggregates output | Coordinator in Supervisory Topology |
| **ContentAnalyzerAgent** | **Content Analyst Expert** | Transcribes video, detects topics/sentiment/chapters/key moments | Intelligence Analyst |
| **ClipAgent** | **Video Editor Expert** | Finds highlights, cuts 15-60s clips | Scriptwriter/Editor |
| **SubtitleAgent** | **Localization Expert** | Whisper STT + multi-language subtitle translation | Localization Specialist |
| **AudioAgent** | **Voice/TTS Expert** | Translates audio, generates AI voiceover (Edge TTS) | Voiceover Artist |
| **ThumbnailAgent** | **Visual Designer Expert** | Extracts key frames, adds text overlays | Creative Director |
| **SocialMediaAgent** | **Social Media Strategist Expert** | Writes platform-specific captions + hashtags | Marketing Specialist |
| **FormatAgent** | **Platform Adapter Expert** | Resizes for 9:16/16:9/1:1, deduplicates renders | Distribution Specialist |
| **QualityAgent** | **QA Reviewer Expert** | Checks audio/subtitle/sync quality, generates report | QA Specialist |

---

## 3. Task Flow — Sequential Pipeline with Review Gate

```
    ┌──────────────────────────────────────────────────────┐
    │                  INPUT: 10+ min video                 │
    └──────────────────────────┬───────────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │  1. Content Analyzer │  Transcribe, topics,
                    │     (Pre-analysis)   │  sentiment, chapters
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  2. Clip Expert      │  Find highlights,
                    │     (Cut clips)      │  cut 15-60s segments
                    └──────────┬──────────┘
                               │
                    ╔══════════▼══════════╗
                    ║  ⏸ HUMAN REVIEW     ║  Approve / drop / edit
                    ║  (Review Gate)       ║  clips before proceeding
                    ╚══════════┬══════════╝
                               │
              ┌────────────────┼────────────────┐
              │                │                │
     ┌────────▼───────┐ ┌─────▼──────┐ ┌───────▼──────┐
     │ 3. Subtitle    │ │ 4. Audio   │ │ 5. Thumbnail│
     │    Expert      │ │    Expert  │ │   Generator │
     │ (Whisper +     │ │ (Edge TTS  │ │ (Key frames │
     │  translate)    │ │  voiceover)│ │  + overlays)│
     └────────┬───────┘ └─────┬──────┘ └───────┬──────┘
              │                │                │
              └────────────────┼────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │  6. Social Media     │  Platform-specific
                    │     Copywriter       │  captions + hashtags
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  7. Format Adapter  │  9:16 / 16:9 / 1:1
                    │     (Resize)         │  (dedup identical)
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  8. Quality Reviewer │  Audio levels,
                    │     (QA check)       │  sync, silence gaps
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  OUTPUT: clips +     │
                    │  subtitles + audio + │
                    │  thumbnails + social │
                    │  + manifest.json     │
                    └─────────────────────┘
```

---

## 4. Multi-Agent Design Principles (from Lecture)

This architecture implements the key principles from Prof. Yong Zhang's lecture:

### Role Specialization
Each agent has a **unique skill** (prompt, tools, skills, knowledge) — no single agent does everything.

### Supervisory (Hierarchical) Topology
The ManagerAgent is the central coordinator that assigns tasks and aggregates results.
> "Its centralized structure is easier to implement, debug, and visualize."

### Team Shared Space
The `pipeline_data` dict is shared between agents — everyone works from the same truth. Each agent reads what it needs and writes its results back.

### Human-in-the-Loop
The review gate after the Clip stage ensures **human oversight** — the "one person" directs the AI team. This is the single biggest system design upgrade showing the human is the boss.

### System Safeguards
- **Runtime Control:** Per-stage timeout (5 min), max retries (1), resource caps
- **Permission Isolation:** Each agent only accesses its own tools and data
- **Human-in-the-Loop:** Approval required before final render

### Task Execution Pattern
**Sequential pipeline** with a pause point — appropriate for tasks with strict dependencies (each stage needs the previous stage's output).

---

## 5. Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| **Backend** | FastAPI (Python) | Async, built-in docs, wraps existing pipeline |
| **Real-time** | Server-Sent Events (SSE) | Simpler than WebSockets, one-way streaming |
| **Orchestration** | ManagerAgent (Python class) | Supervisory topology, retry logic, review gate |
| **Video Processing** | ffmpeg (via imageio-ffmpeg) | Bundled, no system install |
| **Speech-to-Text** | faster-whisper | CPU, int8 quantization, CN-network resilient |
| **Translation** | deep-translator | Google → MyMemory fallback (no API key) |
| **Text-to-Speech** | Edge TTS | Free, works in mainland China |
| **Frontend** | Vanilla HTML/CSS/JS | No build step, fast to develop |

---

## 6. CN Network Resilience

The studio automatically handles services that are blocked on mainland China networks:

| Service | Used For | CN Status | Fallback |
|---------|----------|-----------|----------|
| huggingface.co | Whisper model download | Blocked | Auto-routes via hf-mirror.com |
| translate.google.com | Subtitle translation | Blocked | Falls back to MyMemory API |
| gTTS (Google) | Text-to-speech | Blocked | Uses Edge TTS by default |
