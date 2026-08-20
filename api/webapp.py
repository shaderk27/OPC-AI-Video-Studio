"""
OPC AI Video Studio — Web Control Panel Bridge (M4)
====================================================
APIRouter that serves the browser UI (web/). This is the /api/* contract
documented in web/API.md: config, videos, upload, run, SSE live progress,
review-gate clip decisions, manifest/quality/downloads, and the optional
subtitle burn-in pass.

It lives beside the pipeline_runner/ManagerAgent architecture (server.py)
and drives main.run_studio() in a background thread with an on_stage
callback that feeds the SSE event stream.

Included into the FastAPI app in api/server.py:
    app.include_router(webapp.router)
"""

import json
import os
import queue
import re
import shutil
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from utils.run_paths import prepare_run_output

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config.yaml"
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = Path(os.environ.get("OPC_OUTPUT_DIR", str(BASE_DIR / "output")))
WEB_DIR = BASE_DIR / "web"

router = APIRouter()


# ---------------------------------------------------------------------------
# Run registry (single-user demo: one active run at a time)
# ---------------------------------------------------------------------------
RUNS: Dict[str, Dict[str, Any]] = {}
LOCK = threading.Lock()
ACTIVE_RUN: Dict[str, bool] = {}


def _new_run() -> Dict[str, Any]:
    return {
        "id": time.strftime("%Y%m%d-%H%M%S") + "-" + os.urandom(2).hex(),
        "status": "pending",          # pending | running | awaiting_review | completed | failed
        "queue": queue.Queue(),
        "stages": {},                 # stage -> {status, elapsed}
        "start_time": None,
        "elapsed": 0.0,
        "video": "",
        "manifest": None,
        "quality": None,
        "decisions": {},              # index -> {decision, caption}
        "review_clips": [],
        "approved_indices": None,
        "output_dir": "",
        "review_event": threading.Event(),
        "error": "",
        "log": [],
    }


def _emit(run: Dict[str, Any], event: str, data: Any):
    run["queue"].put({"event": event, "data": data})
    run["log"].append((event, data))


def _save_review_decisions(run: Dict[str, Any]) -> Optional[Path]:
    """Persist the post-Clip approval audit inside this run's folder."""
    output_dir = run.get("output_dir")
    if not output_dir:
        return None

    clips = []
    for index, clip in enumerate(run.get("review_clips", []), 1):
        review = run.get("decisions", {}).get(str(index), {})
        clips.append({
            "index": index,
            "filename": clip.get("filename", ""),
            "title": clip.get("title", ""),
            "start": clip.get("start", 0),
            "end": clip.get("end", 0),
            "decision": review.get("decision", "pending"),
            "caption": review.get("caption", ""),
            "hashtags": review.get("hashtags", []),
        })

    approved_count = sum(
        item["decision"] == "approved" for item in clips
    )
    payload = {
        "run_id": run.get("id", ""),
        "video": run.get("video", ""),
        "submitted_at": datetime.now().astimezone().isoformat(),
        "approved_count": approved_count,
        "rejected_count": len(clips) - approved_count,
        "clips": clips,
    }
    decision_path = Path(output_dir) / "review_decisions.json"
    with open(decision_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
    return decision_path


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------
LANGUAGES = [
    {"code": "en", "name": "English"},
    {"code": "zh", "name": "Chinese"},
    {"code": "es", "name": "Spanish"},
    {"code": "fr", "name": "French"},
    {"code": "de", "name": "German"},
    {"code": "ja", "name": "Japanese"},
    {"code": "ko", "name": "Korean"},
    {"code": "ar", "name": "Arabic"},
    {"code": "hi", "name": "Hindi"},
    {"code": "pt", "name": "Portuguese"},
    {"code": "ru", "name": "Russian"},
    {"code": "it", "name": "Italian"},
    {"code": "tr", "name": "Turkish"},
    {"code": "vi", "name": "Vietnamese"},
    {"code": "th", "name": "Thai"},
    {"code": "id", "name": "Indonesian"},
]

STAGES = [
    {"id": "content_analysis", "name": "Content Analyzer", "role": "Video Content Intelligence Analyst"},
    {"id": "clip", "name": "Clip Expert", "role": "Video Highlight Cutter"},
    {"id": "subtitle", "name": "Subtitle Expert", "role": "Multi-Language Subtitle Generator"},
    {"id": "audio", "name": "Audio Expert", "role": "Audio Translation & Voiceover"},
    {"id": "thumbnail", "name": "Thumbnail Generator", "role": "Thumbnail Creator"},
    {"id": "social_media", "name": "Social Media Copywriter", "role": "Content Strategist"},
    {"id": "format", "name": "Format Adapter", "role": "Multi-Platform Formatter"},
    {"id": "quality", "name": "Quality Reviewer", "role": "QA Specialist"},
]


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _defaults(cfg: dict) -> dict:
    sub = cfg.get("subtitle_agent", {})
    aud = cfg.get("audio_agent", {})
    clp = cfg.get("clip_agent", {})
    gemini = clp.get("gemini_review", {})
    fmt = cfg.get("format_agent", {})
    qa = cfg.get("quality_agent", {})
    return {
        "subtitle_languages": sub.get("target_languages", ["en"]),
        "subtitle_mode": "separate",
        "subtitle_formats": sub.get("formats", ["srt", "vtt"]),
        "audio_languages": aud.get("target_languages", ["zh"]),
        "tts_engine": aud.get("tts_engine", "edge"),
        "formats": [f["platform"] for f in fmt.get("formats", [])],
        "stages": [s["id"] for s in STAGES],
        "whisper_model": sub.get("whisper_model", "small"),
        "max_clips": clp.get("max_clips", 5),
        "min_clip_duration": clp.get("min_clip_duration", 15),
        "max_clip_duration": clp.get("max_clip_duration", 60),
        "detection_strategy": clp.get("detection_strategy", "combined"),
        "padding_before": clp.get("padding_before", 1.5),
        "padding_after": clp.get("padding_after", 1.0),
        "snap_to_silence": clp.get("snap_to_silence", True),
        "gemini_required": gemini.get("required", True),
        "source_language": sub.get("source_language"),
        "original_audio_volume": aud.get("original_audio_volume", 0.15),
        "voiceover_volume": aud.get("voiceover_volume", 1.0),
        "tts_speed": aud.get("tts_speed", 1.0),
        "thumbnail_enabled": True,
        "social_tone": cfg.get("social_media_agent", {}).get("tone", "energetic"),
        "quality_checks": {
            "av_sync": qa.get("check_av_sync", True),
            "subtitle_sync": qa.get("check_subtitle_sync", True),
            "silence_gaps": qa.get("check_silence_gaps", True),
            "min_audio_level": qa.get("min_audio_level", -30),
        },
    }


@router.get("/api/config")
def api_config():
    cfg = load_config()
    formats = []
    for f in cfg.get("format_agent", {}).get("formats", []):
        if f.get("enabled", True):
            formats.append(f)
    return {
        "project": cfg.get("project", {}).get("name", "OPC AI Video Studio"),
        "languages": LANGUAGES,
        "formats": formats,
        "whisper_models": ["tiny", "base", "small", "medium", "large-v3"],
        "detection_strategies": ["combined", "transcript", "audio_energy", "silence"],
        "tts_engines": ["edge", "gtts"],
        "tones": ["energetic", "casual", "professional", "educational"],
        "stages": STAGES,
        "defaults": _defaults(cfg),
    }


# ---------------------------------------------------------------------------
# Videos / upload
# ---------------------------------------------------------------------------
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".ts"}


@router.get("/api/videos")
def api_videos():
    videos = []
    if INPUT_DIR.exists():
        for p in sorted(INPUT_DIR.iterdir()):
            if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
                videos.append({
                    "name": p.name,
                    "path": str(p.relative_to(BASE_DIR)).replace("\\", "/"),
                    "size_mb": round(p.stat().st_size / (1024 * 1024), 1),
                    "modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(p.stat().st_mtime)),
                })
    return {"videos": videos}


@router.post("/api/upload")
async def api_upload(file: UploadFile = File(...)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in VIDEO_EXTS:
        raise HTTPException(400, f"Unsupported video type '{ext}'. Allowed: {sorted(VIDEO_EXTS)}")
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    dest = INPUT_DIR / file.filename
    with open(dest, "wb") as out:
        shutil.copyfileobj(file.file, out)
    return {"name": file.filename, "path": f"input/{file.filename}"}


# ---------------------------------------------------------------------------
# Run request
# ---------------------------------------------------------------------------
class WebRunRequest(BaseModel):
    video: str
    subtitle_languages: Optional[List[str]] = None
    subtitle_mode: str = "separate"           # separate | burn | both
    subtitle_formats: Optional[List[str]] = None
    audio_languages: Optional[List[str]] = None
    tts_engine: Optional[str] = None
    formats: Optional[List[str]] = None
    stages: Optional[List[str]] = None
    whisper_model: Optional[str] = None
    max_clips: Optional[int] = None
    min_clip_duration: Optional[int] = None
    max_clip_duration: Optional[int] = None
    detection_strategy: Optional[str] = None
    padding_before: Optional[float] = None
    padding_after: Optional[float] = None
    snap_to_silence: Optional[bool] = None
    gemini_required: Optional[bool] = None
    source_language: Optional[str] = None
    original_audio_volume: Optional[float] = None
    voiceover_volume: Optional[float] = None
    tts_speed: Optional[float] = None
    thumbnail_enabled: Optional[bool] = None
    social_tone: Optional[str] = None
    quality_checks: Optional[dict] = None


def _apply_overrides(cfg: dict, req: WebRunRequest, run: dict):
    d = _defaults(cfg)

    def pick(v, default):
        return v if v is not None else default

    sub = cfg.setdefault("subtitle_agent", {})
    sub["target_languages"] = pick(req.subtitle_languages, d["subtitle_languages"])
    sub["formats"] = pick(req.subtitle_formats, d["subtitle_formats"])
    sub["whisper_model"] = pick(req.whisper_model, d["whisper_model"])
    sub["source_language"] = pick(req.source_language, d["source_language"])

    aud = cfg.setdefault("audio_agent", {})
    aud["target_languages"] = pick(req.audio_languages, d["audio_languages"])
    aud["tts_engine"] = pick(req.tts_engine, d["tts_engine"])
    aud["original_audio_volume"] = pick(req.original_audio_volume, d["original_audio_volume"])
    aud["voiceover_volume"] = pick(req.voiceover_volume, d["voiceover_volume"])
    aud["tts_speed"] = pick(req.tts_speed, d["tts_speed"])

    clp = cfg.setdefault("clip_agent", {})
    clp["max_clips"] = pick(req.max_clips, d["max_clips"])
    clp["min_clip_duration"] = pick(req.min_clip_duration, d["min_clip_duration"])
    clp["max_clip_duration"] = pick(req.max_clip_duration, d["max_clip_duration"])
    clp["detection_strategy"] = pick(req.detection_strategy, d["detection_strategy"])
    clp["padding_before"] = pick(req.padding_before, d["padding_before"])
    clp["padding_after"] = pick(req.padding_after, d["padding_after"])
    clp["snap_to_silence"] = pick(req.snap_to_silence, d["snap_to_silence"])
    gemini = clp.setdefault("gemini_review", {})
    gemini["required"] = pick(req.gemini_required, d["gemini_required"])
    content_gemini = cfg.setdefault("content_analyzer", {}).setdefault(
        "gemini_analysis", {}
    )
    content_gemini["required"] = gemini["required"]
    if gemini["required"]:
        gemini["enabled"] = True
        content_gemini["enabled"] = True

    fmt = cfg.setdefault("format_agent", {})
    selected = set(pick(req.formats, d["formats"]))
    fmt["formats"] = [f for f in fmt.get("formats", []) if f.get("platform") in selected]

    cfg.setdefault("social_media_agent", {})["tone"] = pick(req.social_tone, d["social_tone"])

    qa = cfg.setdefault("quality_agent", {})
    qc = pick(req.quality_checks, d["quality_checks"]) or {}
    qa["check_av_sync"] = qc.get("av_sync", True)
    qa["check_subtitle_sync"] = qc.get("subtitle_sync", True)
    qa["check_silence_gaps"] = qc.get("silence_gaps", True)
    qa["min_audio_level"] = qc.get("min_audio_level", -30)

    # stage selection -> skip list
    all_stages = [s["id"] for s in STAGES]
    active = [s for s in all_stages if s in set(pick(req.stages, d["stages"]))]
    if gemini["required"]:
        active = [
            s for s in all_stages
            if s in set(active) | {"content_analysis", "clip"}
        ]
    if pick(req.thumbnail_enabled, True) is False and "thumbnail" in active:
        active.remove("thumbnail")
    skip = [s for s in all_stages if s not in active]

    # output dir override (for tests)
    if os.environ.get("OPC_OUTPUT_DIR"):
        out = str(OUTPUT_DIR)
        paths = cfg.setdefault("paths", {})
        paths.update({
            "output_dir": out,
            "clips_dir": os.path.join(out, "clips"),
            "subtitles_dir": os.path.join(out, "subtitles"),
            "audio_dir": os.path.join(out, "audio"),
            "thumbnails_dir": os.path.join(out, "thumbnails"),
            "social_dir": os.path.join(out, "social"),
        })

    run["active_stages"] = active
    return skip


# ---------------------------------------------------------------------------
# Background pipeline worker
# ---------------------------------------------------------------------------
def _run_worker(run_id: str, req: WebRunRequest):
    run = RUNS[run_id]
    run["status"] = "running"
    run["start_time"] = time.time()
    start = run["start_time"]
    cfg = load_config()

    try:
        video_path = req.video
        if not os.path.isabs(video_path):
            video_path = str(BASE_DIR / video_path)
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {req.video}")
        run["video"] = video_path

        skip = _apply_overrides(cfg, req, run)
        cfg, run_dir = prepare_run_output(
            cfg,
            video_path,
            run_id,
            datetime.fromtimestamp(start).astimezone(),
        )
        run["output_dir"] = str(run_dir)
        run["stages"] = {s: {"status": "pending", "elapsed": 0.0} for s in run["active_stages"]}

        _emit(run, "run_started", {
            "run_id": run_id,
            "video": os.path.basename(video_path),
            "stages": run["active_stages"],
        })

        def on_stage(stage, status, message=None):
            st = run["stages"].get(stage, {"status": "pending", "elapsed": 0.0})
            if status == "started":
                st.update(status="active", elapsed=0.0)
                _emit(run, "stage_started", {
                    "stage": stage, "index": run["active_stages"].index(stage) + 1,
                    "total": len(run["active_stages"]),
                    "name": next((s["name"] for s in STAGES if s["id"] == stage), stage),
                    "role": next((s["role"] for s in STAGES if s["id"] == stage), ""),
                })
            elif status == "completed":
                st.update(status="done", elapsed=round(time.time() - start, 1))
                _emit(run, "stage_completed", {"stage": stage, "ok": True, "message": message or ""})
            else:
                st.update(status="failed", elapsed=round(time.time() - start, 1))
                _emit(run, "stage_failed", {"stage": stage, "ok": False, "error": message or "failed"})

        def on_review(clips):
            run["review_clips"] = clips
            run["approved_indices"] = None
            run["decisions"] = {
                str(index): {"decision": "pending"}
                for index in range(1, len(clips) + 1)
            }
            run["review_event"].clear()
            run["status"] = "awaiting_review"
            _emit(run, "review_ready", {
                "count": len(clips),
                "message": "Approve clips before downstream processing",
            })
            run["review_event"].wait()

            approved_indices = run.get("approved_indices") or []
            approved = [
                clips[index - 1] for index in approved_indices
                if 1 <= index <= len(clips)
            ]
            run["status"] = "running"
            _emit(run, "review_completed", {
                "approved": len(approved),
                "rejected": len(clips) - len(approved),
            })
            return approved

        from main import run_studio
        run_studio(
            video_path=video_path,
            config=cfg,
            skip_stages=skip,
            on_stage=on_stage,
            on_review=on_review,
        )

        # load manifest + quality FIRST (the burn pass reads the manifest)
        out_dir = cfg.get("paths", {}).get("output_dir", "output")
        manifest_path = os.path.join(out_dir, "manifest.json")
        quality_path = os.path.join(out_dir, "quality_report.json")
        if os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as f:
                run["manifest"] = json.load(f)
        if os.path.exists(quality_path):
            with open(quality_path, "r", encoding="utf-8") as f:
                run["quality"] = json.load(f)

        # burn-in pass (embed subtitles into video) if requested
        if req.subtitle_mode in ("burn", "both"):
            _burn_subtitles(run, cfg, req.subtitle_languages or ["en"])

        run["status"] = "completed"
        run["elapsed"] = round(time.time() - start, 1)
        _emit(run, "run_completed", {
            "run_id": run_id,
            "elapsed": run["elapsed"],
            "clips": len(run.get("manifest", {}).get("clips", [])),
        })
    except Exception as e:
        run["status"] = "failed"
        run["elapsed"] = round(time.time() - start, 1)
        run["error"] = str(e)
        _emit(run, "run_failed", {"run_id": run_id, "error": run["error"]})
    finally:
        ACTIVE_RUN["busy"] = False


def _burn_subtitles(run: dict, cfg: dict, langs: list):
    """Best-effort: burn each selected subtitle language into copies under output/burned/.

    Uses cwd-relative paths inside the filter graph (absolute Windows paths with
    '\\:' escapes are not reliably parsed by the subtitles filter), so we run
    ffmpeg with cwd=output_dir and copy a CJK-capable font into output/fonts/.
    """
    import imageio_ffmpeg
    from utils.video_utils import get_video_info

    out_dir = Path(cfg.get("paths", {}).get("output_dir", "output"))
    manifest = run.get("manifest") or {}
    if not manifest.get("clips"):
        return

    ff = imageio_ffmpeg.get_ffmpeg_exe()
    burned_dir = out_dir / "burned"
    fonts_dir = out_dir / "fonts"
    burned_dir.mkdir(parents=True, exist_ok=True)
    fonts_dir.mkdir(parents=True, exist_ok=True)

    font_candidates = [
        r"C:/Windows/Fonts/msyh.ttc",
        r"C:/Windows/Fonts/simhei.ttf",
        r"C:/Windows/Fonts/simsun.ttc",
        r"C:/Windows/Fonts/arial.ttf",
        "/System/Library/Fonts/PingFang.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    fontfile = next((p for p in font_candidates if os.path.exists(p)), None)
    font_name = None
    if fontfile:
        try:
            font_name = "studio_font" + os.path.splitext(fontfile)[1].lower()
            shutil.copy(fontfile, fonts_dir / font_name)
        except Exception as e:
            _emit(run, "stage_log", {"stage": "format", "message": f"  burn: font copy failed ({e})"})
            font_name = None

    style = "FontName=Microsoft YaHei,FontSize=18,Alignment=2,MarginV=28"
    style_esc = style.replace(",", "\\,")   # escape commas for the filter graph

    for clip in manifest.get("clips", []):
        clip_path = clip.get("path", "")
        if not clip_path or not os.path.exists(clip_path):
            continue
        stem = os.path.splitext(os.path.basename(clip_path))[0]

        for lang in langs:
            sub_path = (clip.get("subtitles") or {}).get(lang, {}).get("formats", {}).get("srt", "")
            if isinstance(sub_path, dict):       # tolerate {path, filename} shape
                sub_path = sub_path.get("path", "")
            if not sub_path or not os.path.exists(sub_path):
                _emit(run, "stage_log", {"stage": "format", "message": f"  burn: no {lang} srt for {clip.get('filename')}"})
                continue

            out_path = os.path.abspath(str(burned_dir / f"{stem}_burn_{lang}.mp4"))
            try:
                # ffmpeg runs with cwd=output_dir: inputs/outputs must be absolute,
                # but the srt path inside the filter graph must stay cwd-relative.
                clip_abs = os.path.abspath(clip_path)
                sub_abs = os.path.abspath(sub_path)
                rel = os.path.relpath(sub_abs, os.path.abspath(out_dir)).replace("\\", "/")
                if rel.startswith(".."):          # srt outside output dir -> copy next to the clip
                    rel = os.path.basename(sub_path)
                    shutil.copy(sub_path, os.path.join(out_dir, rel))
                vf = f"subtitles=filename={rel}"
                if font_name:
                    vf += f":fontsdir=fonts"
                vf += f":force_style={style_esc}"
                proc = subprocess.run(
                    [ff, "-y", "-i", clip_abs, "-vf", vf,
                     "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                     "-c:a", "copy", out_path],
                    capture_output=True, text=True, cwd=str(out_dir),
                )
                if os.path.exists(out_path):
                    info = get_video_info(out_path)
                    _emit(run, "stage_log", {"stage": "format",
                                             "message": f"  burn OK: {os.path.basename(out_path)} ({info['width']}x{info['height']})"})
                else:
                    _emit(run, "stage_log", {"stage": "format",
                                             "message": f"  burn FAILED (ffmpeg exit {proc.returncode}): {clip.get('filename')}"})
            except Exception as e:
                _emit(run, "stage_log", {"stage": "format", "message": f"  burn error: {e}"})


# ---------------------------------------------------------------------------
# Run endpoints
# ---------------------------------------------------------------------------
@router.post("/api/run")
def api_run(req: WebRunRequest):
    if ACTIVE_RUN.get("busy"):
        raise HTTPException(409, "A run is already in progress. Wait for it to finish or refresh.")
    cfg = load_config()
    effective_gemini_required = (
        req.gemini_required
        if req.gemini_required is not None
        else _defaults(cfg)["gemini_required"]
    )
    if effective_gemini_required:
        if req.stages is not None:
            missing = {
                "content_analysis", "clip"
            } - set(req.stages)
            if missing:
                raise HTTPException(
                    400,
                    "Require Gemini API also requires Content Analyzer and "
                    "Clip Expert stages.",
                )
        gemini = cfg.setdefault("clip_agent", {}).setdefault("gemini_review", {})
        gemini.update({"enabled": True, "required": True})
        content_gemini = cfg.setdefault("content_analyzer", {}).setdefault(
            "gemini_analysis", {}
        )
        content_gemini.update({"enabled": True, "required": True})
        from utils.gemini_clip_reviewer import GeminiClipReviewer
        from utils.gemini_content_analyzer import GeminiContentAnalyzer
        setup_error = (
            GeminiClipReviewer(cfg).configuration_error() or
            GeminiContentAnalyzer(cfg).configuration_error()
        )
        if setup_error:
            raise HTTPException(
                400, f"Gemini API is required but not configured: {setup_error}"
            )
    run = _new_run()
    run["video"] = req.video
    RUNS[run["id"]] = run
    ACTIVE_RUN["busy"] = True
    threading.Thread(target=_run_worker, args=(run["id"], req), daemon=True).start()
    return {"run_id": run["id"]}


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.get("/api/runs/{run_id}/events")
def api_run_events(run_id: str):
    run = RUNS.get(run_id)
    if not run:
        raise HTTPException(404, "Run not found")

    def gen():
        yield _sse("run_started", {
            "run_id": run_id,
            "video": os.path.basename(run["video"]) if run["video"] else "",
            "stages": run.get("active_stages", []),
        })
        for stage, st in run.get("stages", {}).items():
            if st["status"] != "pending":
                yield _sse("stage_" + ("completed" if st["status"] == "done" else "failed"),
                           {"stage": stage, "ok": st["status"] == "done"})
        while True:
            try:
                evt = run["queue"].get(timeout=15)
            except queue.Empty:
                if run["status"] in ("completed", "failed"):
                    break
                yield _sse("ping", {})
                continue
            yield _sse(evt["event"], evt["data"])
            if evt["event"] in ("run_completed", "run_failed"):
                break

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.get("/api/runs/{run_id}")
def api_run_status(run_id: str):
    run = RUNS.get(run_id)
    if not run:
        raise HTTPException(404, "Run not found")
    return {
        "run_id": run_id,
        "status": run["status"],
        "elapsed": run["elapsed"],
        "video": os.path.basename(run["video"]) if run["video"] else "",
        "stages": run.get("stages", {}),
        "clips_count": len(
            run.get("review_clips", [])
            if run.get("status") == "awaiting_review"
            else (run.get("manifest") or {}).get("clips", [])
        ),
        "error": run.get("error", ""),
    }


def _social_copy_for(clip: dict) -> dict:
    """Load the social copy JSON for a clip (captions/hashtags)."""
    sc = clip.get("social_media")
    if not sc:
        return {}
    path = sc.get("copy_file") if isinstance(sc, dict) else None
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


@router.get("/api/runs/{run_id}/clips")
def api_run_clips(run_id: str):
    run = RUNS.get(run_id)
    if not run:
        raise HTTPException(404, "Run not found")
    source_clips = (
        run.get("review_clips", [])
        if run.get("status") == "awaiting_review"
        else (run.get("manifest") or {}).get("clips", [])
    )
    if not source_clips and run.get("status") not in ("awaiting_review", "completed"):
        raise HTTPException(409, "Clips are not ready for review")
    items = []
    for i, clip in enumerate(source_clips, start=1):
        social = _social_copy_for(clip)
        first_platform = (social.get("TikTok") or social.get("YouTube Shorts") or
                          social.get("Instagram Reels") or next(iter(social.values()), {}) or {})
        decision = run["decisions"].get(str(i), {})
        items.append({
            "index": i,
            "filename": clip.get("filename", ""),
            "path": clip.get("path", ""),
            "url": "/" + clip.get("path", "").replace("\\", "/").lstrip("/"),
            "duration": clip.get("duration", 0),
            "start": clip.get("start", 0),
            "end": clip.get("end", 0),
            "title": clip.get("title", ""),
            "reason": clip.get("reason", ""),
            "hook_score": clip.get("hook_score", 0),
            "hook_score_breakdown": clip.get("hook_score_breakdown", {}),
            "topic_relevance": clip.get("topic_relevance", 0),
            "matched_topic": clip.get("matched_topic", ""),
            "is_topic_relevant": clip.get("is_topic_relevant", False),
            "selection_score": clip.get("selection_score", 0),
            "gemini_video_reviewed": clip.get("gemini_video_reviewed", False),
            "gemini_video_score": clip.get("gemini_video_score"),
            "gemini_event_type": clip.get("gemini_event_type", ""),
            "gemini_video_reason": clip.get("gemini_video_reason", ""),
            "subtitle_langs": list((clip.get("subtitles") or {}).keys()),
            "thumbnails": [t for t in (clip.get("thumbnails") or [])],
            "caption": decision.get("caption") or first_platform.get("caption", ""),
            "hashtags": decision.get("hashtags") or first_platform.get("hashtags", []),
            "decision": decision.get("decision", "pending"),
        })
    return {"clips": items}


@router.post("/api/runs/{run_id}/review")
def api_continue_after_review(run_id: str, body: dict):
    """Release the post-Clip gate with an explicit approved clip list."""
    run = RUNS.get(run_id)
    if not run:
        raise HTTPException(404, "Run not found")
    if run.get("status") != "awaiting_review":
        raise HTTPException(409, "Run is not awaiting clip review")

    clip_count = len(run.get("review_clips", []))
    raw_indices = body.get("approved_indices", [])
    if not isinstance(raw_indices, list) or any(
            not isinstance(index, int) for index in raw_indices):
        raise HTTPException(400, "approved_indices must be a list of integers")
    approved_indices = sorted(set(raw_indices))
    if any(index < 1 or index > clip_count for index in approved_indices):
        raise HTTPException(400, "approved clip index is out of range")

    run["approved_indices"] = approved_indices
    for index in range(1, clip_count + 1):
        run["decisions"].setdefault(str(index), {})["decision"] = (
            "approved" if index in approved_indices else "dropped"
        )
    _save_review_decisions(run)
    run["review_event"].set()
    return {
        "ok": True,
        "approved_count": len(approved_indices),
        "rejected_count": clip_count - len(approved_indices),
    }


@router.post("/api/runs/{run_id}/clips/{index}/decision")
def api_clip_decision(run_id: str, index: str, body: dict):
    run = RUNS.get(run_id)
    if not run:
        raise HTTPException(404, "Run not found")
    decision = body.get("decision")
    if decision not in ("approved", "dropped", "pending"):
        raise HTTPException(400, "decision must be approved|dropped|pending")
    run["decisions"].setdefault(index, {})["decision"] = decision
    return {"ok": True}


@router.post("/api/runs/{run_id}/clips/{index}/caption")
def api_clip_caption(run_id: str, index: str, body: dict):
    run = RUNS.get(run_id)
    if not run:
        raise HTTPException(404, "Run not found")
    run["decisions"].setdefault(index, {})["caption"] = body.get("caption", "")
    run["decisions"].setdefault(index, {})["hashtags"] = body.get("hashtags", [])
    return {"ok": True}


@router.get("/api/runs/{run_id}/manifest")
def api_run_manifest(run_id: str):
    run = RUNS.get(run_id)
    if not run:
        raise HTTPException(404, "Run not found")
    return run.get("manifest") or {}


@router.get("/api/runs/{run_id}/quality")
def api_run_quality(run_id: str):
    run = RUNS.get(run_id)
    if not run:
        raise HTTPException(404, "Run not found")
    return run.get("quality") or {}


@router.get("/api/runs/{run_id}/downloads")
def api_run_downloads(run_id: str):
    run = RUNS.get(run_id)
    if not run or not run["manifest"]:
        raise HTTPException(404, "Run not finished or no manifest yet")
    out_dir = Path(run.get("output_dir") or OUTPUT_DIR)
    def url(p):  # -> absolute URL path for <video src> / downloads
        path = Path(p)
        try:
            relative = path.resolve().relative_to(OUTPUT_DIR.resolve())
            return "/output/" + relative.as_posix()
        except ValueError:
            return "/" + str(path).replace("\\", "/").lstrip("/")

    clips, subtitles, formatted, burned, audio, thumbs, social = [], [], [], [], [], [], []

    # map platform -> aspect_ratio from config (manifest only stores paths)
    ratio_of = {}
    try:
        for f in load_config().get("formats", []):
            ratio_of[f.get("platform", "")] = f.get("aspect_ratio", "")
    except Exception:
        pass

    def _aspect_from_path(p):
        m = re.search(r"_(\d{2,5})x(\d{2,5})\.(mp4|mov|webm)$", os.path.basename(p))
        return f"{m.group(1)}:{m.group(2)}" if m else ""

    for clip in run["manifest"].get("clips", []):
        p = clip.get("path", "")
        if p and os.path.exists(p):
            clips.append({"filename": os.path.basename(p), "url": url(p),
                          "duration": clip.get("duration", 0), "title": clip.get("title", "")})
        for lang, ld in (clip.get("subtitles") or {}).items():
            for ext, sp in (ld.get("formats") or {}).items():
                sp_path = sp.get("path", "") if isinstance(sp, dict) else str(sp)
                if sp_path and os.path.exists(sp_path):
                    subtitles.append({"filename": os.path.basename(sp_path), "url": url(sp_path),
                                      "lang": lang, "ext": ext})
        for lang, ld in (clip.get("audio_translations") or {}).items():
            vp = ld.get("video_path", "") if isinstance(ld, dict) else ""
            if vp and os.path.exists(vp):
                audio.append({"filename": os.path.basename(vp), "url": url(vp), "lang": lang})
        for plat, pv in (clip.get("platform_versions") or {}).items():
            pp = pv.get("path", "") if isinstance(pv, dict) else str(pv)
            if pp and os.path.exists(pp):
                formatted.append({"platform": plat, "filename": os.path.basename(pp), "url": url(pp),
                                  "aspect_ratio": (pv.get("aspect_ratio", "") if isinstance(pv, dict) else "")
                                  or ratio_of.get(plat, "") or _aspect_from_path(pp)})
        for t in (clip.get("thumbnails") or []):
            tp = t.get("path", "") if isinstance(t, dict) else str(t)
            if tp and os.path.exists(tp):
                thumbs.append(url(tp))
    burned_dir = out_dir / "burned"
    if burned_dir.exists():
        for p in sorted(burned_dir.glob("*.mp4")):
            burned.append({"filename": p.name, "url": url(p)})
    social_dir = out_dir / "social"
    if social_dir.exists():
        for p in sorted(social_dir.glob("*.json")):
            social.append(url(p))
    return {"clips": clips, "subtitles": subtitles, "audio": audio, "formatted": formatted,
            "burned": burned, "thumbnails": thumbs, "social": social}
