# Web Frontend ↔ Backend API Contract

> Maintained by **M4 (Frontend)** — implement in `api/server.py` (M1 owns that file).
> The frontend (`web/`) is a **thin wrapper**: it only does `fetch` + `EventSource` + render.
> If `/api/config` is unreachable, the frontend automatically switches to **Demo Mode**
> (simulated run) so the UI is always presentable.

Base URL: `http://localhost:8000` · All responses JSON (`application/json`).
Media files are served directly: `/output/...` maps to the pipeline's `output/` folder.

---

## 1. `GET /api/config` — studio options & defaults

The single source of truth for every option the UI renders. Return values **from `config.yaml`**:

```json
{
  "project": "OPC AI Video Studio",
  "languages": [ {"code": "en", "name": "English"}, {"code": "zh", "name": "Chinese"}, "...20+" ],
  "formats": [
    {"platform": "TikTok", "width": 1080, "height": 1920, "aspect_ratio": "9:16"},
    {"platform": "YouTube Shorts", "width": 1080, "height": 1920, "aspect_ratio": "9:16"},
    {"platform": "Instagram Reels", "width": 1080, "height": 1920, "aspect_ratio": "9:16"},
    {"platform": "YouTube", "width": 1920, "height": 1080, "aspect_ratio": "16:9"},
    {"platform": "Instagram Feed", "width": 1080, "height": 1080, "aspect_ratio": "1:1"}
  ],
  "whisper_models": ["tiny", "base", "small", "medium", "large-v3"],
  "detection_strategies": ["combined", "transcript", "audio_energy", "silence"],
  "tts_engines": ["edge", "gtts"],
  "tones": ["energetic", "casual", "professional", "educational"],
  "stages": [
    {"id": "content_analysis", "name": "Content Analyzer", "role": "Video Content Intelligence Analyst"},
    {"id": "clip", "name": "Clip Expert", "role": "Video Highlight Cutter"},
    {"id": "subtitle", "name": "Subtitle Expert", "role": "Multi-Language Subtitle Generator"},
    {"id": "audio", "name": "Audio Expert", "role": "Audio Translation & Voiceover"},
    {"id": "thumbnail", "name": "Thumbnail Generator", "role": "Thumbnail Creator"},
    {"id": "social_media", "name": "Social Media Copywriter", "role": "Content Strategist"},
    {"id": "format", "name": "Format Adapter", "role": "Multi-Platform Formatter"},
    {"id": "quality", "name": "Quality Reviewer", "role": "QA Specialist"}
  ],
  "defaults": {
    "subtitle_languages": ["en", "zh"],
    "subtitle_mode": "separate",          // "separate" | "burn" | "both"
    "subtitle_formats": ["srt", "vtt"],
    "audio_languages": ["zh"],
    "tts_engine": "edge",
    "formats": ["TikTok", "YouTube Shorts", "Instagram Reels", "YouTube", "Instagram Feed"],
    "stages": ["content_analysis", "clip", "subtitle", "audio", "thumbnail", "social_media", "format", "quality"],
    "whisper_model": "base",
    "max_clips": 5,
    "min_clip_duration": 15,
    "max_clip_duration": 60,
    "detection_strategy": "combined",
    "gemini_required": true,
    "padding_before": 1.5,
    "padding_after": 1.0,
    "snap_to_silence": true,
    "source_language": null,
    "original_audio_volume": 0.15,
    "voiceover_volume": 1.0,
    "tts_speed": 1.0,
    "thumbnail_enabled": true,
    "social_tone": "energetic",
    "quality_checks": {"av_sync": true, "subtitle_sync": true, "silence_gaps": true, "min_audio_level": -30}
  }
}
```

## 2. `GET /api/videos` — videos available in `input/`

```json
{ "videos": [ {"name": "Kayaking.mp4", "path": "input/Kayaking.mp4", "size_mb": 2964, "modified": "2026-08-18T13:41:00"} ] }
```

## 3. `POST /api/upload` — upload a video (multipart, field `file`)

```json
{ "name": "MyTrip.mp4", "path": "input/MyTrip.mp4" }
```

## 4. `POST /api/run` — start a pipeline run

Body (all fields optional; omitted → `defaults` from config):

```json
{
  "video": "input/Kayaking.mp4",
  "subtitle_languages": ["en", "zh"],
  "subtitle_mode": "separate",          // "separate" | "burn" | "both"
  "subtitle_formats": ["srt", "vtt"],
  "audio_languages": ["zh"],
  "tts_engine": "edge",
  "formats": ["TikTok", "YouTube Shorts", "Instagram Reels"],
  "stages": ["content_analysis", "clip", "subtitle", "audio", "thumbnail", "social_media", "format", "quality"],
  "whisper_model": "base",
  "max_clips": 5,
  "min_clip_duration": 15,
  "max_clip_duration": 60,
  "detection_strategy": "combined",
  "gemini_required": true,
  "padding_before": 1.5,
  "padding_after": 1.0,
  "snap_to_silence": true,
  "source_language": null,
  "original_audio_volume": 0.15,
  "voiceover_volume": 1.0,
  "tts_speed": 1.0,
  "thumbnail_enabled": true,
  "social_tone": "energetic",
  "quality_checks": {"av_sync": true, "subtitle_sync": true, "silence_gaps": true, "min_audio_level": -30}
}
```

Response: `{ "run_id": "20260818-1455-abc123" }` (server starts the pipeline in a background thread).

## 5. `GET /api/runs/{run_id}/events` — Server-Sent Events (live progress)

`text/event-stream`. Event names + `data` payloads:

| event | data |
|---|---|
| `run_started` | `{run_id, video, stages: ["content_analysis", ...], start_time}` |
| `stage_started` | `{stage, index, total, name, role}` |
| `stage_log` | `{stage, message}` (optional fine-grained lines) |
| `stage_completed` | `{stage, ok: true, message}` |
| `stage_failed` | `{stage, ok: false, error}` |
| `review_ready` | `{count}` (pipeline pauses immediately after Clip Expert) |
| `review_completed` | `{approved, rejected}` (downstream processing resumes) |
| `run_completed` | `{run_id, elapsed, manifest, clips, quality}` |
| `run_failed` | `{run_id, error}` |
| `ping` | `{}` every 15 s (keep-alive) |

Frontend falls back to polling `GET /api/runs/{run_id}` every 2 s if SSE fails.
`gemini_required` defaults to `true`. The run fails if the Gemini key/SDK is missing,
a candidate proxy cannot be created or uploaded, the API cannot be reached,
the request times out, a transcript/video response is invalid, or every visual
candidate is rejected. The reason is delivered through `run_failed` and the
status snapshot's `error` field.

## 6. `GET /api/runs/{run_id}` — status snapshot (poll fallback)

```json
{ "run_id": "...", "status": "running|awaiting_review|completed|failed", "elapsed": 42.0,
  "error": "",
  "stages": [ {"stage": "clip", "status": "active|done|failed|pending", "elapsed": 3.2} ],
  "clips_count": 5 }
```

## 7. Review gate

- `GET /api/runs/{run_id}/clips` → review data:

```json
{ "clips": [ {"index": 1, "filename": "Kayaking_clip_01.mp4", "path": "output/clips/...mp4",
   "duration": 16.35, "start": 12.4, "end": 28.7, "title": "...", "reason": "...",
   "hook_score": 72, "subtitle_langs": ["en","zh"], "thumbnails": ["output/thumbnails/...png"],
   "caption": "...", "hashtags": ["#fyp","#kayaking"], "decision": "approved|dropped|pending" } ] }
```

- `POST /api/runs/{run_id}/clips/{index}/decision` body `{"decision": "approved"|"dropped"}` → `{ok: true}`
- `POST /api/runs/{run_id}/review` body `{"approved_indices": [1,3]}` resumes the pipeline with only those clips.

The review gate occurs immediately after Clip Expert. Subtitle, audio,
thumbnail, social-media, format, and quality agents do not run until this
endpoint is called. An empty list ends the run without downstream processing.
The final decisions are saved as `review_decisions.json` in that run's folder.

## 8. Results

- `GET /api/runs/{run_id}/manifest` → that run's `manifest.json` content
- `GET /api/runs/{run_id}/quality` → that run's `quality_report.json` content
- `GET /api/runs/{run_id}/downloads` → list of ready-to-download artifacts

Every processing attempt is isolated under
`output/YYYY-MM-DD/HH-MM-SS_video-name_run-id/`. Reprocessing a video therefore
does not overwrite an earlier manifest, report, clip, or caption.

```json
{ "clips": [{"filename": "...", "url": "/output/2026-08-19/14-32-08_video_a3f1/clips/...mp4"}],
  "subtitles": [{"filename": "...", "url": "/output/2026-08-19/14-32-08_video_a3f1/subtitles/...srt", "lang": "zh"}],
  "formatted": [{"platform": "TikTok", "url": "/output/2026-08-19/14-32-08_video_a3f1/formatted/..._1080x1920.mp4"}],
  "burned": [{"lang": "en", "url": "/output/2026-08-19/14-32-08_video_a3f1/burned/...mp4"}],
  "thumbnails": ["/output/2026-08-19/14-32-08_video_a3f1/thumbnails/...png"],
  "social": ["/output/2026-08-19/14-32-08_video_a3f1/social/...json"] }
```

## Notes for M1

- Serve `web/` at `/` and `output/` at `/output` (StaticFiles). SSE via `StreamingResponse`.
- Runs registry: keep `{run_id: RunState}` in memory (a dict + lock) — the demo is single-user.
- Add `on_stage(stage, status, message)` callback support to `run_studio()` (already added in `main.py`) so the server can emit `stage_*` events.
- `subtitle_mode: "burn"` → after the pipeline, burn the chosen subtitle language into copies under `output/burned/` (ffmpeg `subtitles` filter, font fallback chain `msyh.ttc → simhei.ttf → arial.ttf`). Best-effort: failures surface as `stage_log` warnings, never crash the run.
- Review decisions live in `runs/{run_id}_decisions.json`; the gallery applies them (dropped clips hidden).
