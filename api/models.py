"""
OPC AI Video Studio — Pydantic Models
Request/response schemas for the FastAPI backend.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Upload
# ─────────────────────────────────────────────────────────────────────────────

class UploadResponse(BaseModel):
    """Response after uploading a video file."""
    video_id: str = Field(..., description="Unique ID for the uploaded video")
    filename: str = Field(..., description="Original filename")
    path: str = Field(..., description="Saved file path")
    size_bytes: int = Field(..., description="File size in bytes")


# ─────────────────────────────────────────────────────────────────────────────
# Run Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class RunRequest(BaseModel):
    """Request to start the video processing pipeline."""
    video_id: str = Field(..., description="Video ID from upload")
    languages: Optional[List[str]] = Field(
        None, description="Target languages (e.g., ['en', 'zh'])"
    )
    subtitle_languages: Optional[List[str]] = Field(
        None, description="Subtitle languages; overrides the legacy languages field"
    )
    audio_languages: Optional[List[str]] = Field(
        None, description="Dubbing languages; overrides the legacy languages field"
    )
    max_clips: Optional[int] = Field(
        None, ge=1, le=20, description="Target clips to generate"
    )
    skip_stages: Optional[List[str]] = Field(
        None, description="Stages to skip (e.g., ['thumbnail', 'quality'])"
    )
    review_gate_enabled: bool = Field(
        True, description="Pause after clip stage for human review"
    )
    localization_review_enabled: bool = Field(
        True, description="Pause for combined Subtitle/Audio script approval"
    )
    gemini_required: bool = Field(
        True,
        description="Fail the run if Gemini content analysis or clip review cannot be completed",
    )


class RunResponse(BaseModel):
    """Response after starting a pipeline run."""
    run_id: str = Field(..., description="Unique ID for this pipeline run")
    video_id: str = Field(..., description="Video being processed")
    status: str = Field("running", description="Initial status")
    message: str = Field("Pipeline started", description="Human-readable message")


# ─────────────────────────────────────────────────────────────────────────────
# Status
# ─────────────────────────────────────────────────────────────────────────────

class StageStatus(BaseModel):
    """Status of a single pipeline stage."""
    name: str
    status: str  # "pending", "running", "complete", "failed", "skipped"
    duration: Optional[float] = None
    error: Optional[str] = None


class RunStatus(BaseModel):
    """Full status of a pipeline run."""
    run_id: str
    video_id: str
    status: str  # running, awaiting_review, awaiting_localization_review, complete, error
    current_stage: Optional[str] = None
    stages: List[StageStatus] = []
    clips_count: int = 0
    errors: List[str] = []
    elapsed_seconds: Optional[float] = None
    created_at: str
    message: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Clips
# ─────────────────────────────────────────────────────────────────────────────

class ThumbnailInfo(BaseModel):
    path: str
    filename: str
    timestamp: float
    overlay_path: Optional[str] = None


class SubtitleInfo(BaseModel):
    formats: Dict[str, Dict[str, str]] = {}
    segment_count: int = 0


class ClipInfo(BaseModel):
    """Metadata for a generated clip."""
    filename: str
    path: str
    duration: float
    start: float
    end: float
    title: str
    reason: str
    detected_language: Optional[str] = None
    transcript_preview: Optional[str] = None
    gemini_video_reviewed: bool = False
    gemini_video_score: Optional[float] = None
    gemini_event_type: Optional[str] = None
    gemini_video_reason: Optional[str] = None
    subtitles: Dict[str, SubtitleInfo] = {}
    audio_translations: Dict[str, Dict[str, str]] = {}
    thumbnails: List[ThumbnailInfo] = []
    social_media: Optional[Dict[str, Any]] = None
    platform_versions: Dict[str, Dict[str, Any]] = {}
    quality_passed: Optional[bool] = None


class ClipsResponse(BaseModel):
    """Response listing all clips for a run."""
    run_id: str
    clips: List[ClipInfo]
    total: int


# ─────────────────────────────────────────────────────────────────────────────
# Approve (Review Gate)
# ─────────────────────────────────────────────────────────────────────────────

class ClipEdit(BaseModel):
    """An edit to a clip before approval."""
    index: int = Field(..., ge=0, description="Clip index in the list")
    title: Optional[str] = None
    start: Optional[float] = None
    end: Optional[float] = None


class ApproveRequest(BaseModel):
    """Human approval of clips at the review gate."""
    approved_indices: List[int] = Field(
        ..., description="Indices of clips to approve (0-based)"
    )
    edits: Optional[List[ClipEdit]] = Field(
        None, description="Optional edits to clips"
    )
    approve_all: bool = Field(
        False, description="If true, approve all clips (ignores approved_indices)"
    )


class ApproveResponse(BaseModel):
    """Response after approving clips."""
    run_id: str
    status: str = "resumed"
    approved_count: int
    message: str = "Pipeline resumed"


class LocalizationReviewRequest(BaseModel):
    items: List[Dict[str, Any]]


class LocalizationReviewResponse(BaseModel):
    run_id: str
    status: str = "resumed"
    approved_count: int


# ─────────────────────────────────────────────────────────────────────────────
# Progress Events (SSE)
# ─────────────────────────────────────────────────────────────────────────────

class ProgressEvent(BaseModel):
    """A progress event streamed via SSE."""
    event_type: str  # stage_start, stage_complete, stage_error, clip_found,
                     # review_ready, pipeline_complete, pipeline_error
    data: Dict[str, Any]
    timestamp: str


# ─────────────────────────────────────────────────────────────────────────────
# Export
# ─────────────────────────────────────────────────────────────────────────────

class ExportResponse(BaseModel):
    """Response with export info."""
    run_id: str
    manifest_path: str
    clips_count: int
    output_dir: str
    download_url: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Error
# ─────────────────────────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    """Standard error response."""
    detail: str
    error_code: Optional[str] = None
