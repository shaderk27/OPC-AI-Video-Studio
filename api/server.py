"""
OPC AI Video Studio — FastAPI Server
    allow_methods=["*"],
    allow_headers=["*"],
)

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = PROJECT_ROOT / "input"
OUTPUT_DIR = PROJECT_ROOT / "output"
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Serve generated output files (clips, thumbnails, subtitles, etc.)
app.mount("/output", StaticFiles(directory=str(OUTPUT_DIR)), name="output")

# Track uploaded videos: {video_id: {filename, path, size}}
_uploads: Dict[str, dict] = {}


# ─────────────────────────────────────────────────────────────────────────────
# Health check
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/")
async def health():
    return {
        "service": "OPC AI Video Studio",
        "status": "running",
        "version": "1.0.0",
        "endpoints": [
            "POST /upload", "POST /run", "GET /status/{id}",
            "GET /sse/{id}", "GET /clips/{id}", "POST /approve/{id}",
            "GET /export/{id}", "GET /output/{path}",
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Upload video
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/upload", response_model=UploadResponse)
async def upload_video(file: UploadFile = File(...)):
    """Upload a video file to be processed by the studio."""
    if not file.filename:
        raise HTTPException(status_code=422, detail="No filename provided")

    # Basic validation — accept common video formats
    allowed_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed_exts:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type '{ext}'. Allowed: {allowed_exts}",
        )

    video_id = str(uuid.uuid4())[:8]
    save_path = INPUT_DIR / f"{video_id}_{file.filename}"

    # Write the uploaded file
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    size = os.path.getsize(save_path)

    _uploads[video_id] = {
        "filename": file.filename,
        "path": str(save_path),
        "size": size,
    }

    return UploadResponse(
        video_id=video_id,
        filename=file.filename,
        path=str(save_path),
        size_bytes=size,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Run pipeline
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/run", response_model=RunResponse)
async def run_pipeline(request: RunRequest):
    """Start the video processing pipeline for an uploaded video."""
    # Check video exists
    if request.video_id not in _uploads:
        raise HTTPException(status_code=404, detail="Video not found. Upload first.")

    upload = _uploads[request.video_id]
    video_path = upload["path"]
    video_name = Path(upload["filename"]).stem

    # Load config and apply overrides
    config = load_config()

    if request.languages:
        config.setdefault("subtitle_agent", {})["target_languages"] = request.languages
        config.setdefault("audio_agent", {})["target_languages"] = request.languages

    if request.max_clips:
        config.setdefault("clip_agent", {})["max_clips"] = request.max_clips

    if request.review_gate_enabled is not None:
        config.setdefault("manager_agent", {})["review_gate_enabled"] = request.review_gate_enabled

    # Start the run
    run_id = runner.start_run(
        video_path=video_path,
        video_name=video_name,
        config=config,
        skip_stages=request.skip_stages,
    )

    return RunResponse(
        run_id=run_id,
        video_id=request.video_id,
        status="running",
        message="Pipeline started",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Status
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/status/{run_id}", response_model=RunStatus)
async def get_status(run_id: str):
    """Get the current status of a pipeline run."""
    if not runner.run_exists(run_id):
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

    status = runner.get_status(run_id)
    if not status:
        raise HTTPException(status_code=500, detail="Failed to get status")

    return RunStatus(**status)


# ─────────────────────────────────────────────────────────────────────────────
# SSE — real-time progress stream
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/sse/{run_id}")
async def stream_progress(run_id: str):
    """Stream pipeline progress events via Server-Sent Events (SSE)."""
    if not runner.run_exists(run_id):
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

    async def event_generator():
        """Generate SSE events from the pipeline runner."""
        last_index = 0

        while True:
            # Get new events since last check
            events = runner.get_events(run_id, after_index=last_index)

            for event in events:
                last_index += 1
                event_type = event.get("event_type", "update")
                data = event.get("data", {})
                timestamp = event.get("timestamp", "")

                # Format as SSE
                sse_data = json.dumps({"data": data, "timestamp": timestamp})
                yield f"event: {event_type}\ndata: {sse_data}\n\n"

            # Check if pipeline is done
            status = runner.get_status(run_id)
            if status and status["status"] in ("complete", "error"):
                # Send one final event
                final_data = json.dumps({
                    "status": status["status"],
                    "run_id": run_id,
                    "elapsed": status["elapsed_seconds"],
                })
                yield f"event: done\ndata: {final_data}\n\n"
                break

            # Brief pause before polling again (1 second)
            await asyncio.sleep(1)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Clips
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/clips/{run_id}", response_model=ClipsResponse)
async def get_clips(run_id: str):
    """List all generated clips for a run."""
    if not runner.run_exists(run_id):
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

    clips = runner.get_clips(run_id)
    if clips is None:
        clips = []

    # Build clip info with relative paths for the /output static mount
    clip_list = []
    for clip in clips:
        clip_path = clip.get("path", "")
        # Convert absolute path to URL path for static serving
        if clip_path and os.path.isabs(clip_path):
            try:
                rel = os.path.relpath(clip_path, str(OUTPUT_DIR))
                clip_path = f"/output/{rel.replace(os.sep, '/')}"
            except ValueError:
                pass

        clip_info = {
            "filename": clip.get("filename", ""),
            "path": clip_path,
            "duration": clip.get("duration", 0),
            "start": clip.get("start", 0),
            "end": clip.get("end", 0),
            "title": clip.get("title", ""),
            "reason": clip.get("reason", ""),
            "detected_language": clip.get("detected_language"),
            "transcript_preview": clip.get("transcript_preview"),
        }

        # Add thumbnails if available
        if clip.get("thumbnails"):
            clip_info["thumbnails"] = []
            for thumb in clip["thumbnails"]:
                thumb_path = thumb.get("path", "")
                if thumb_path and os.path.isabs(thumb_path):
                    try:
                        rel = os.path.relpath(thumb_path, str(OUTPUT_DIR))
                        thumb_path = f"/output/{rel.replace(os.sep, '/')}"
                    except ValueError:
                        pass
                clip_info["thumbnails"].append({
                    "path": thumb_path,
                    "filename": thumb.get("filename", ""),
                    "timestamp": thumb.get("timestamp", 0),
                })

        # Add subtitles if available
        if clip.get("subtitles"):
            clip_info["subtitles"] = {}
            for lang, lang_data in clip["subtitles"].items():
                clip_info["subtitles"][lang] = {
                    "formats": {},
                    "segment_count": len(lang_data.get("segments", [])),
                }
                for fmt, fmt_data in lang_data.get("formats", {}).items():
                    fmt_path = fmt_data.get("path", "")
                    if fmt_path and os.path.isabs(fmt_path):
                        try:
                            rel = os.path.relpath(fmt_path, str(OUTPUT_DIR))
                            fmt_path = f"/output/{rel.replace(os.sep, '/')}"
                        except ValueError:
                            pass
                    clip_info["subtitles"][lang]["formats"][fmt] = {
                        "path": fmt_path,
                        "filename": fmt_data.get("filename", ""),
                    }

        # Add audio translations if available
        if clip.get("audio_translations"):
            clip_info["audio_translations"] = {}
            for lang, lang_data in clip["audio_translations"].items():
                clip_info["audio_translations"][lang] = {
                    "video_path": lang_data.get("video_path", ""),
                    "voiceover_path": lang_data.get("voiceover_path", ""),
                }

        # Add social media if available
        if clip.get("social_media"):
            clip_info["social_media"] = clip["social_media"]

        # Add platform versions if available
        if clip.get("platform_versions"):
            clip_info["platform_versions"] = {}
            for platform, pv_data in clip["platform_versions"].items():
                pv_path = pv_data.get("path", "")
                if pv_path and os.path.isabs(pv_path):
                    try:
                        rel = os.path.relpath(pv_path, str(OUTPUT_DIR))
                        pv_path = f"/output/{rel.replace(os.sep, '/')}"
                    except ValueError:
                        pass
                clip_info["platform_versions"][platform] = {
                    "path": pv_path,
                    "width": pv_data.get("width", 0),
                    "height": pv_data.get("height", 0),
                }

        clip_list.append(clip_info)

    return ClipsResponse(run_id=run_id, clips=clip_list, total=len(clip_list))


# ─────────────────────────────────────────────────────────────────────────────
# Approve — review gate
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/approve/{run_id}", response_model=ApproveResponse)
async def approve_clips(run_id: str, request: ApproveRequest):
    """Approve clips at the review gate and resume the pipeline."""
    if not runner.run_exists(run_id):
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

    result = runner.approve(
        run_id=run_id,
        approved_indices=request.approved_indices,
        edits=[e.model_dump() for e in request.edits] if request.edits else None,
        approve_all=request.approve_all,
    )

    if not result:
        raise HTTPException(status_code=500, detail="Failed to approve clips")

    if "error" in result:
        raise HTTPException(status_code=409, detail=result["error"])

    return ApproveResponse(
        run_id=run_id,
        status=result.get("status", "resumed"),
        approved_count=result.get("approved_count", 0),
        message=result.get("message", "Pipeline resumed"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Export
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/export/{run_id}", response_model=ExportResponse)
async def export_run(run_id: str):
    """Get export info for a completed run (manifest path + output directory)."""
    if not runner.run_exists(run_id):
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

    manifest = runner.get_manifest(run_id)
    if not manifest:
        status = runner.get_status(run_id)
        raise HTTPException(
            status_code=409,
            detail=f"Run not complete (status: {status['status'] if status else 'unknown'})",
        )

    return ExportResponse(
        run_id=run_id,
        manifest_path=manifest.get("manifest_path", ""),
        clips_count=manifest.get("clips", []).__len__(),
        output_dir=str(OUTPUT_DIR),
        download_url=f"/output/manifest.json",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Architecture info (for the frontend / presentation)
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/architecture")
async def get_architecture():
    """Return the agent team structure and WorkBuddy expert mapping."""
    from agents.manager_agent import STAGE_AGENTS, WORKBUDDY_EXPERT_MAP

    agents = []
    for stage, (module_path, class_name) in STAGE_AGENTS.items():
        agents.append({
            "stage": stage,
            "class": class_name,
            "expert": WORKBUDDY_EXPERT_MAP.get(stage, stage),
        })

    return {
        "topology": "Supervisory (Hierarchical)",
        "coordinator": "ManagerAgent",
        "agents": agents,
        "pipeline": [
            "content_analysis", "clip",
            "[HUMAN REVIEW GATE]",
            "subtitle", "audio", "thumbnail",
            "social_media", "format", "quality",
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Run on import (for uvicorn)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=True)
