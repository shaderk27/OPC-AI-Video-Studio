# OPC AI Video Studio

**One Human + Multiple AI Agents = One-Person Video Studio**

A multi-agent platform that turns long videos into short, engaging social media clips automatically. Based on the project concept by Prof. Yong Zhang, Shenzhen University.

## What It Does

Give it a 10+ minute raw video (travel vlog, lecture, match footage, etc.) and the studio will:

1. **Analyze content** — transcribe the full video, detect topics, sentiment, chapters, and key moments
2. **Find highlights** and cut them into 15-60s clips (using topic relevance, emotion, audio energy, hook scoring, and optional Gemini editorial review)
3. **Pause for clip approval** so rejected candidates do not receive further processing
4. **Generate subtitles** in multiple languages
5. **Translate audio** into multiple languages with AI voiceover
6. **Create thumbnails** for each clip
7. **Write social media copy** for TikTok, YouTube, Instagram, X, Douyin, and Xiaohongshu
8. **Format clips** for each platform (vertical, horizontal, square)
9. **Review quality** and generate a report

## The AI Agent Team

| Agent | Role | What It Does |
|-------|------|-------------|
| **Content Analyzer** | Video Content Intelligence Analyst | Pre-analyzes full video: transcribes, detects topics, sentiment, chapters, key moments, generates summary. Enriches all downstream agents. |
| **Clip Expert** | Video Highlight Cutter | Uses topics, emotion, audio energy, hook scoring, optional Gemini transcript review, and completion padding to select complete highlights |
| **Subtitle Expert** | Multi-Language Subtitle Generator | Uses Whisper for speech-to-text, translates subtitles into any language |
| **Audio Expert** | Audio Translation Specialist | Translates speech and generates AI voiceover in target languages, merges with video |
| **Thumbnail Generator** | Thumbnail Creator | Extracts key frames and adds text overlays for eye-catching thumbnails |
| **Social Media Copywriter** | Content Strategist | Writes platform-specific captions and hashtags, including distinct Douyin and Xiaohongshu styles |
| **Format Adapter** | Multi-Platform Formatter | Resizes clips for TikTok (9:16), YouTube (16:9), Instagram (1:1). Platforms with the same dimensions share one render (TikTok/Shorts/Reels → single 1080x1920 file) |
| **Quality Reviewer** | QA Specialist | Checks audio levels, subtitle sync, silence gaps, generates quality report |

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs the media-processing dependencies, the optional Google Gen AI client, and FastAPI/uvicorn for the web control panel.

To enable Gemini-assisted clip ranking, copy the environment template and put
your Gemini API key in the local `.env` file. `.env` is ignored by Git, while
`.env.example` is safe to commit. Do not place the key in `config.yaml`.

```powershell
Copy-Item .env.example .env
# Edit .env and replace the placeholder:
# GEMINI_API_KEY=your-api-key
```

The application loads the project-level `.env` automatically. A value already
set in the operating-system environment takes priority over the file.

The default model is `gemini-3.5-flash-lite`. If the key is absent, the request
times out, or Gemini returns unusable output, the Clip Expert automatically
keeps its original heuristic ranking.

### 1b. Web Control Panel (optional, recommended for demos)

```bash
python scripts/start_web.py       # opens http://127.0.0.1:8000
```

The web UI wraps the whole pipeline — no CLI needed:

- **Setup** — pick or upload a video, choose subtitle languages (separate subtitle files, burned directly into the video, or both), audio voiceover languages, and output formats (grouped by aspect ratio — TikTok/Shorts/Reels share one 1080×1920 render). Advanced options cover Whisper model size, clip count & durations, detection strategy, padding/snap-to-silence, TTS engine/speed/volumes, caption tone, thumbnails, quality checks, and per-agent toggles.
- **Live progress** — real-time stage rail + log console (Server-Sent Events, with polling fallback).
- **Review gate** — immediately after Clip Expert, explicitly approve or drop every candidate. Only approved clips continue to subtitles, audio, thumbnails, social copy, formatting, and quality review.
- **Gallery** — download clips, subtitles, translated audio, formatted platform versions, thumbnails and social copy.
- **Demo Mode** — if the backend isn't running, the UI plays a simulated run so you can preview the workflow anytime.

API contract between the frontend and the FastAPI backend: `web/API.md`.

### 2. Configure Your Studio

Edit `config.yaml` to set:
- Target languages for subtitles and audio
- Number of clips to generate
- Clip duration range (15-60s default)
- Social media platforms
- Whisper model size (tiny/base/small/medium/large-v3)

### 3. Run the Studio

```bash
# Basic usage
python main.py input/my_video.mp4

# With specific languages and clip count
python main.py input/travel_vlog.mp4 --languages en,zh,ja --max-clips 5

# Skip certain stages
python main.py input/lecture.mp4 --skip thumbnail,quality

# Store the dated run folder under a different archive root
python main.py input/travel_vlog.mp4 --output demo-output
```

### 4. Find Your Output

Every processing attempt receives its own folder. The default convention is
`output/YYYY-MM-DD/HH-MM-SS_video-name_run-id/`, so processing the same video
again does not overwrite earlier clips, reports, or captions. `--output`
changes the archive root while preserving the dated run-folder structure.

```
output/
└── 2026-08-19/
    └── 14-32-08_shenzhen-travel_a3f1b2c4/
        ├── content_analysis.json  # Topics, sentiment, chapters, and key moments
        ├── manifest.json          # Approved outputs and their paths
        ├── quality_report.json    # Quality review results
        ├── review_decisions.json  # Approved and rejected clip candidates
        ├── clips/                 # Raw highlight candidates (MP4)
        ├── subtitles/             # SRT, VTT, and TXT files per language
        ├── audio/                 # Translated audio and voiceover files
        ├── thumbnails/            # Thumbnail images (PNG)
        ├── social/                # Platform captions and hashtags (JSON)
        ├── formatted/             # Platform-specific video dimensions
        └── burned/                # Videos with embedded subtitles
```

Rejected raw candidates remain in that run's `clips/` folder for inspection,
but they are excluded from `manifest.json` and do not receive downstream
processing.

## Configuration

### Target Languages

Supported language codes (ISO 639-1):

| Code | Language | Code | Language |
|------|----------|------|----------|
| en | English | ja | Japanese |
| zh | Chinese | ko | Korean |
| es | Spanish | ar | Arabic |
| fr | French | hi | Hindi |
| de | German | pt | Portuguese |
| ru | Russian | it | Italian |
| tr | Turkish | vi | Vietnamese |
| th | Thai | id | Indonesian |

Edit `config.yaml`:

```yaml
subtitle_agent:
  target_languages:
    - "en"
    - "zh"
    - "es"

audio_agent:
  target_languages:
    - "en"
    - "zh"
```

### Highlight Detection

The Clip Expert uses multiple strategies to find interesting moments:

- **Transcript analysis**: Searches for keywords, detects questions, exclamations, content density
- **Content topics**: Scores candidates against Content Analyzer topics and promotes diverse relevant moments
- **Emotion + audio energy**: Rewards moments where strong sentiment and audible energy occur together
- **Hook score**: Ranks opening strength, emotion, audio energy, specificity, and coherence
- **Gemini editorial review**: Optionally reviews timestamped transcript excerpts for hook, flow, value, and standalone completeness, then blends that judgement with the measurable score
- **Silence detection**: Snaps clip boundaries to silence for clean cuts
- **Combined** (default): Uses all strategies together

Only transcript excerpts, candidate timestamps, topics, and numeric scores are
sent to Gemini in this first integration; video files and frames are not
uploaded. Configure or disable it in `config.yaml`:

```yaml
clip_agent:
  gemini_review:
    enabled: true
    required: false
    model: "gemini-3.5-flash-lite"
    candidate_limit: 12
    blend_weight: 0.5
```

In the web control panel, enable **Require Gemini API** under Advanced options
to make the review mandatory for that run. Missing credentials, SDK errors,
timeouts, unreachable API responses, and invalid model output then fail the
Clip stage and display the reason. Leave it off to retain heuristic fallback.

### Whisper Model Sizes

| Model | Size | Speed | Accuracy | RAM |
|-------|------|-------|----------|-----|
| tiny | 75MB | Fastest | Basic | ~1GB |
| base | 145MB | Fast | Good | ~1GB |
| small | 484MB | Medium | Better | ~2GB |
| medium | 1.5GB | Slow | High | ~5GB |
| large-v3 | 3GB | Slowest | Best | ~10GB |

## Network Notes (Mainland China)

Some online services used by the studio are blocked or unreachable on CN
networks. The studio handles this automatically:

| Service | Used for | CN status | How the studio handles it |
|---------|----------|-----------|---------------------------|
| huggingface.co | Whisper model download | Blocked | Auto-routes through **hf-mirror.com**, caches under `models/whisper/` |
| translate.google.com | Subtitle translation | Blocked | Falls back to **MyMemory** (mymemory.translated.net) |
| gTTS (Google) | Text-to-speech | Blocked | Uses **Edge TTS** by default (`tts_engine: "edge"`) |
| Gemini API | Optional clip editorial review | May be unavailable | Keeps the local heuristic ranking; set `gemini_review.enabled: false` to disable calls |

- **Pre-download the Whisper model** (optional but recommended on first run):

  ```bash
  python scripts/setup_whisper_model.py
  ```

- To force a different HuggingFace endpoint, set the environment variable:

  ```bash
  set HF_ENDPOINT=https://hf-mirror.com     # Windows CMD
  $env:HF_ENDPOINT="https://hf-mirror.com"  # PowerShell
  ```

## Architecture

```
                    ┌─────────────────┐
                    │   Input Video   │
                    │  (10+ minutes)  │
                    └────────┬────────┘
                             │
                    ┌────────▼────────────┐
                    │ Content Analyzer    │
                    │ (Topics, Sentiment, │
                    │  Chapters, Summary) │
                    └────────┬────────────┘
                             │
                    ┌────────▼────────┐
                    │  Clip Expert    │
                    │ (Find & Cut)    │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ Human Approval │
                    │ (Keep or Drop) │
                    └────────┬────────┘
                             │
                 ┌───────────┼───────────┐
                 │           │           │
        ┌────────▼───┐ ┌────▼────┐ ┌───▼────────┐
        │ Subtitle   │ │ Audio   │ │ Thumbnail  │
        │ Expert     │ │ Expert  │ │ Generator  │
        └────────┬───┘ └────┬────┘ └───┬────────┘
                 │          │          │
        ┌────────▼──────────▼──────────▼────────┐
        │       Social Media Copywriter          │
        └────────────────────┬──────────────────┘
                             │
                    ┌────────▼────────┐
                    │ Format Adapter  │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ Quality Reviewer│
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Final Output   │
                    │ (clips + subs + │
                    │ audio + social) │
                    └─────────────────┘
```

## Enhancement Agents

Beyond the three core experts (Clip, Subtitle, Audio), the studio includes five enhancement agents:

1. **Content Analyzer** - Pre-analyzes full video for topics, sentiment, chapters, key moments, and summary. Runs before Clip Expert to enable smarter highlight selection and avoid duplicate transcription.
2. **Thumbnail Generator** - Extracts visually striking frames and adds text overlays
3. **Social Media Copywriter** - Writes platform-specific captions with hashtags
4. **Format Adapter** - Converts clips to 9:16, 16:9, and 1:1 aspect ratios. Identical target resolutions are rendered once and shared across platforms, so TikTok/Shorts/Reels don't triple-encode the same 1080x1920 video
5. **Quality Reviewer** - Validates audio levels, subtitle sync, and silence gaps

## Tech Stack

- **Video Processing**: ffmpeg (bundled via imageio-ffmpeg)
- **Speech-to-Text**: faster-whisper (Whisper model, CPU)
- **Translation**: Google Translator with MyMemory fallback (via deep-translator, no API key needed)
- **Text-to-Speech**: Edge TTS (Microsoft, free) with gTTS fallback
- **Video Editing**: moviepy
- **Image Processing**: Pillow

## Project Context

This project implements the concept from the "One-Person AI Video Studio Project" by Prof. Yong Zhang, Assistant Professor at the School of Artificial Intelligence, Shenzhen University. The project was part of the 2026 Summer International Camp.

Core concept: **One Human + Multiple AI Agents = One-Person Work Studio** — empowering creators with AI to handle the entire video production pipeline independently.

## License

Educational project. Use freely for learning and development.
