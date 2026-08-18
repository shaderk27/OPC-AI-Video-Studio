# OPC AI Video Studio

**One Human + Multiple AI Agents = One-Person Video Studio**

A multi-agent platform that turns long videos into short, engaging social media clips automatically. Based on the project concept by Prof. Yong Zhang, Shenzhen University.

## What It Does

Give it a 10+ minute raw video (travel vlog, lecture, match footage, etc.) and the studio will:

1. **Analyze content** — transcribe the full video, detect topics, sentiment, chapters, and key moments
2. **Find highlights** and cut them into 15-60s clips (using the content analysis for smarter selection)
3. **Generate subtitles** in multiple languages
4. **Translate audio** into multiple languages with AI voiceover
5. **Create thumbnails** for each clip
6. **Write social media copy** for TikTok, YouTube, Instagram, X
7. **Format clips** for each platform (vertical, horizontal, square)
8. **Review quality** and generate a report

## The AI Agent Team

| Agent | Role | What It Does |
|-------|------|-------------|
| **Content Analyzer** | Video Content Intelligence Analyst | Pre-analyzes full video: transcribes, detects topics, sentiment, chapters, key moments, generates summary. Enriches all downstream agents. |
| **Clip Expert** | Video Highlight Cutter | Uses content analysis + audio energy to find interesting moments, cuts them into short clips |
| **Subtitle Expert** | Multi-Language Subtitle Generator | Uses Whisper for speech-to-text, translates subtitles into any language |
| **Audio Expert** | Audio Translation Specialist | Translates speech and generates AI voiceover in target languages, merges with video |
| **Thumbnail Generator** | Thumbnail Creator | Extracts key frames and adds text overlays for eye-catching thumbnails |
| **Social Media Copywriter** | Content Strategist | Writes platform-specific captions and hashtags |
| **Format Adapter** | Multi-Platform Formatter | Resizes clips for TikTok (9:16), YouTube (16:9), Instagram (1:1). Platforms with the same dimensions share one render (TikTok/Shorts/Reels → single 1080x1920 file) |
| **Quality Reviewer** | QA Specialist | Checks audio levels, subtitle sync, silence gaps, generates quality report |

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs: moviepy, imageio-ffmpeg (bundled ffmpeg, no system install needed), faster-whisper, deep-translator, gTTS, pyyaml, rich, pysrt, Pillow.

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
```

### 4. Find Your Output

All generated content goes to the `output/` directory:

```
output/
├── content_analysis.json  # Full content analysis (topics, sentiment, chapters, key moments)
├── clips/              # Short highlight clips (MP4)
├── subtitles/          # SRT, VTT, TXT files per language
├── audio/              # Translated audio clips (MP4 + MP3 voiceover)
├── thumbnails/         # Thumbnail images (PNG)
├── social/             # Social media copy (JSON)
├── formatted/          # Formatted clips (named by resolution, e.g. ..._1080x1920.mp4)
├── manifest.json       # Master index of all generated content
└── quality_report.json # Quality review report
```

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
- **Audio energy**: Finds high-energy moments (cheers, laughter, emphasis)
- **Silence detection**: Snaps clip boundaries to silence for clean cuts
- **Combined** (default): Uses all strategies together

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
