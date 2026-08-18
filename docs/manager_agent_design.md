# ManagerAgent Design Document

## OPC AI Video Studio — Supervisory Orchestrator

**Role:** System Architect & Orchestrator (Member 1)
**Topology:** Supervisory (Hierarchical) — as recommended in Prof. Yong Zhang's lecture:
> "We will start with the Supervisory model. Its centralized structure is easier to implement, debug, and visualize."

---

## 1. Purpose

The ManagerAgent is the **central coordinator** of the AI Video Studio pipeline. It replaces the simple for-loop in `main.py` with a robust orchestrator that:

1. **Routes work** to specialized agents (the "digital employees")
2. **Handles failures** with retry logic (retry once, then escalate)
3. **Pauses for human review** after the Clip stage (human-in-the-loop)
4. **Streams progress** in real-time via events (for SSE / web UI)
5. **Aggregates output** into a final manifest

---

## 2. Class Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    BaseAgent (ABC)                       │
│  - name, role, config, logger                           │
│  + announce(msg), log(msg, level)                       │
│  + run(input_data) → wraps execute() with timing        │
│  + execute(input_data) [abstract]                      │
│  + ensure_dir(path)                                     │
└───────────────────────┬─────────────────────────────────┘
                        │ inherits
┌───────────────────────▼─────────────────────────────────┐
│                  ManagerAgent                            │
│                                                         │
│  - pipeline: List[str]         # stage order from config│
│  - max_retries: int = 1        # retry once on failure   │
│  - stage_timeout: int = 300    # 5 min per stage         │
│  - progress_callback            # event emitter          │
│  - review_gate: asyncio.Event  # pause/resume signal     │
│  - approved_clips: List[dict]  # set before resume       │
│                                                         │
│  + execute(input_data) → dict  # full pipeline run       │
│  - _run_stage_with_retry(stage, pipeline_data, results)  │
│  - _instantiate_agent(stage) → BaseAgent                 │
│  - _update_pipeline_data(stage, result, pipeline_data)  │
│  - _pause_for_review(pipeline_data)                      │
│  - _emit(event_type, data)                              │
│  - _build_manifest(pipeline_data, results) → dict        │
│  - _is_core_stage(stage) → bool                         │
└─────────────────────────────────────────────────────────┘
```

---

## 3. Stage-to-Agent Mapping

The ManagerAgent maps pipeline stage names (from `config.yaml`) to agent classes:

```python
STAGE_AGENTS = {
    "content_analysis": ("agents.content_analyzer_agent", "ContentAnalyzerAgent"),
    "clip":             ("agents.clip_agent",             "ClipAgent"),
    "subtitle":         ("agents.subtitle_agent",         "SubtitleAgent"),
    "audio":            ("agents.audio_agent",             "AudioAgent"),
    "thumbnail":        ("agents.thumbnail_agent",        "ThumbnailAgent"),
    "social_media":     ("agents.social_media_agent",     "SocialMediaAgent"),
    "format":           ("agents.format_agent",           "FormatAgent"),
    "quality":          ("agents.quality_agent",          "QualityAgent"),
}
```

**Core stages** (failure stops the pipeline): `content_analysis`, `clip`, `subtitle`
**Non-core stages** (failure is logged, pipeline continues): `audio`, `thumbnail`, `social_media`, `format`, `quality`

---

## 4. Retry Policy

```
For each stage:
  1. Try to execute the stage
  2. If it fails:
     a. If core stage: retry once with same params
        - If retry fails: STOP pipeline, emit pipeline_error
     b. If non-core stage: retry once with same params
        - If retry fails: LOG error, mark stage as "failed", CONTINUE to next stage
  3. Emit stage_complete or stage_error event
```

**Max retries:** 1 (so max 2 attempts per stage)
**Stage timeout:** 300 seconds (5 min) — if exceeded, abort stage

---

## 5. Review Gate (Human-in-the-Loop)

### When It Triggers
After the **Clip stage** completes successfully, before the **Subtitle stage** begins.

### Flow
1. ManagerAgent runs `content_analysis` → `clip` stages normally
2. After clips are generated, ManagerAgent calls `_pause_for_review(pipeline_data)`
3. This emits a `review_ready` event containing the clip list
4. ManagerAgent sets status to "awaiting_review" and blocks on `review_gate.wait()`
5. The FastAPI layer (`pipeline_runner.py`) holds the background task
6. The web UI (Member 4) shows clips with approve/drop/edit buttons
7. Human calls `POST /approve/{run_id}` with approved clip indices + optional edits
8. `pipeline_runner.py` sets `approved_clips` on ManagerAgent and sets the asyncio.Event
9. ManagerAgent resumes: replaces `pipeline_data["clips"]` with approved clips
10. Pipeline continues: `subtitle` → `audio` → `thumbnail` → `social_media` → `format` → `quality`

### Timeout
If no approval comes within 5 minutes, auto-approve all clips (prevents deadlock during demo).

### Thread/Asyncio Boundary
- ManagerAgent runs in a **background thread** (agents are synchronous, blocking)
- `review_gate` is a `threading.Event` (not asyncio.Event) since it's waited on in the thread
- The FastAPI `/approve` endpoint sets the event from the async context
- Progress events are pushed to an `asyncio.Queue` via `loop.call_soon_threadsafe()`

---

## 6. Progress Event Protocol

Events are dicts with `event_type` and `data` keys:

| Event Type | Data | When |
|-----------|------|------|
| `stage_start` | `{stage, timestamp}` | Before each agent runs |
| `stage_complete` | `{stage, duration, summary}` | After each agent succeeds |
| `stage_error` | `{stage, error, will_retry}` | When an agent fails |
| `clip_found` | `{index, start, end, title, reason, score}` | After ClipAgent produces clips |
| `review_ready` | `{clips: [...], message}` | When pipeline pauses for review |
| `pipeline_complete` | `{manifest_path, total_time, clips_count}` | After all stages finish |
| `pipeline_error` | `{error, stage}` | On unrecoverable failure |

---

## 7. Pipeline Data Flow (Team Shared Space)

The `pipeline_data` dict is the shared context passed between agents (the "Team Shared Space" from the lecture). Each agent reads from it and writes results back:

```
Initial:
  {video_path, video_name}

After content_analysis:
  + content_analysis: {topics, sentiment, chapters, key_moments, summary, segments}
  + full_transcript: [...]
  + detected_language: "en"
  + video_info: {duration, width, height, fps, has_audio}

After clip:
  + clips: [{path, filename, start, end, duration, title, reason, ...}]

--- REVIEW GATE (human approves/edits clips) ---

After subtitle:
  clips[i] += subtitles: {lang: {formats, segments}}

After audio:
  clips[i] += audio_translations: {lang: {video_path, voiceover_path}}

After thumbnail:
  clips[i] += thumbnails: [{path, filename, timestamp}]

After social_media:
  clips[i] += social_media: {platforms, keywords, copy_file}

After format:
  clips[i] += platform_versions: {platform: {path, width, height}}

After quality:
  + quality_report: {total_clips, issues, summary}
```

---

## 8. Manifest Output

The final manifest (saved to `output/manifest.json`) contains:
- Project name, video name, video path
- Processing time
- Config used (languages, max_clips, whisper_model)
- Content analysis summary (language, topics, chapters, key moments, summary)
- Per-clip details: filename, path, duration, start/end, title, subtitles, audio_translations, thumbnails, social_media, platform_versions

---

## 9. Configuration (config.yaml addition)

```yaml
manager_agent:
  max_retries: 1              # Retry once on failure
  stage_timeout: 300          # 5 minutes per stage
  review_gate_enabled: true   # Pause after clip stage for human review
  review_gate_timeout: 300    # Auto-approve after 5 min if no response
  core_stages:               # These stages stop the pipeline on failure
    - "content_analysis"
    - "clip"
    - "subtitle"
```
