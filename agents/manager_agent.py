"""
Manager Agent (Orchestrator)
============================
Role: Pipeline Orchestrator & Coordinator

Implements the Supervisory (Hierarchical) topology from Prof. Yong Zhang's
multi-agent lecture: a central coordinator that assigns tasks to specialized
agents, handles failures with retry logic, pauses for human review, streams
progress events, and aggregates the final output.

This is the "boss" of the AI Video Studio — the human's representative that
directs the team of digital employees.

Design doc: docs/manager_agent_design.md
"""

import importlib
import json
import os
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from agents.base_agent import BaseAgent


# ─────────────────────────────────────────────────────────────────────────────
# Stage-to-Agent mapping (the "team roster")
# Maps pipeline stage names to (module_path, class_name) for dynamic import.
# ─────────────────────────────────────────────────────────────────────────────
STAGE_AGENTS: Dict[str, tuple] = {
    "content_analysis": ("agents.content_analyzer_agent", "ContentAnalyzerAgent"),
    "clip":             ("agents.clip_agent",             "ClipAgent"),
    "subtitle":         ("agents.subtitle_agent",         "SubtitleAgent"),
    "audio":            ("agents.audio_agent",             "AudioAgent"),
    "thumbnail":        ("agents.thumbnail_agent",        "ThumbnailAgent"),
    "social_media":     ("agents.social_media_agent",     "SocialMediaAgent"),
    "format":           ("agents.format_agent",           "FormatAgent"),
    "quality":          ("agents.quality_agent",          "QualityAgent"),
}

# Default pipeline order (overridden by config.yaml)
DEFAULT_PIPELINE = [
    "content_analysis", "clip", "subtitle", "audio",
    "thumbnail", "social_media", "format", "quality",
]

# Core stages — failure here stops the entire pipeline
DEFAULT_CORE_STAGES = {"content_analysis", "clip", "subtitle"}

# WorkBuddy expert mapping (conceptual, for architecture diagram)
WORKBUDDY_EXPERT_MAP = {
    "manager":            "Orchestrator / Project Manager",
    "content_analysis":   "Content Analyst Expert",
    "clip":               "Video Editor Expert",
    "subtitle":           "Localization Expert",
    "audio":              "Voice/TTS Expert",
    "thumbnail":          "Visual Designer Expert",
    "social_media":       "Social Media Strategist Expert",
    "format":             "Platform Adapter Expert",
    "quality":            "QA Reviewer Expert",
}


class ManagerAgent(BaseAgent):
    """
    Supervisory orchestrator for the AI Video Studio pipeline.

    Responsibilities (from the lecture's Supervisory topology):
      1. Receive the request and decide which agents to run (from config)
      2. Route work to each agent, passing the shared pipeline_data
      3. Handle failures: retry once, then escalate/skip
      4. Pause after Clip stage for human review (human-in-the-loop)
      5. Stream progress events for real-time monitoring
      6. Aggregate final output into manifest.json
    """

    def __init__(
        self,
        config: Dict[str, Any],
        progress_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        review_gate: Optional[threading.Event] = None,
    ):
        super().__init__(
            name="Manager",
            role="Pipeline Orchestrator & Coordinator",
            config=config,
        )

        # Pipeline configuration
        mgr_config = config.get("manager_agent", {})
        self.pipeline: List[str] = config.get("orchestrator", {}).get(
            "pipeline", DEFAULT_PIPELINE
        )
        self.max_retries: int = mgr_config.get("max_retries", 1)
        self.stage_timeout: int = mgr_config.get("stage_timeout", 300)
        self.review_gate_enabled: bool = mgr_config.get("review_gate_enabled", True)
        self.review_gate_timeout: int = mgr_config.get("review_gate_timeout", 300)
        self.core_stages: set = set(
            mgr_config.get("core_stages", list(DEFAULT_CORE_STAGES))
        )

        # Progress streaming
        self.progress_callback = progress_callback

        # Review gate (threading.Event — waited on in the pipeline thread,
        # set from the async FastAPI context via PipelineRunner)
        self.review_gate = review_gate or threading.Event()
        self.approved_clips: Optional[List[Dict]] = None
        self._awaiting_review = False

        # Run tracking
        self.run_id: str = str(uuid.uuid4())[:8]
        self.stages_completed: List[str] = []
        self.stages_failed: List[str] = []
        self.stage_durations: Dict[str, float] = {}
        self.errors: List[str] = []

    # ─────────────────────────────────────────────────────────────────────
    # Main execution — runs the full pipeline
    # ─────────────────────────────────────────────────────────────────────
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the full AI Video Studio pipeline.

        input_data: {"video_path": str, "video_name": str, "skip_stages": [...]}
        returns: {"manifest": {...}, "pipeline_data": {...}, "results": {...}}
        """
        start_time = time.time()
        video_path = input_data["video_path"]
        skip_stages = set(input_data.get("skip_stages", []))

        # Initialize the shared pipeline data (Team Shared Space)
        pipeline_data: Dict[str, Any] = {
            "video_path": os.path.abspath(video_path),
            "video_name": input_data.get("video_name", Path(video_path).stem),
        }
        results: Dict[str, Any] = {}

        self.announce("Pipeline started — supervising the AI team")
        self._emit("pipeline_start", {
            "run_id": self.run_id,
            "video_name": pipeline_data["video_name"],
            "pipeline": [s for s in self.pipeline if s not in skip_stages],
        })

        # Verify input video exists
        if not os.path.exists(video_path):
            error_msg = f"Video file not found: {video_path}"
            self._emit("pipeline_error", {"error": error_msg, "stage": "init"})
            return {"error": error_msg, "manifest": None}

        # Run each stage in order
        for stage in self.pipeline:
            if stage in skip_stages:
                self.log(f"Skipping stage: {stage}")
                self._emit("stage_skipped", {"stage": stage})
                continue

            # Special handling: pause for review after clip stage
            if stage == "clip":
                self._run_stage_with_retry(stage, pipeline_data, results)

                # Check if we should pause for human review
                if self.review_gate_enabled and pipeline_data.get("clips"):
                    self._pause_for_review(pipeline_data)
                    if not pipeline_data.get("clips"):
                        self._emit("pipeline_stopped", {
                            "reason": "No clips were approved",
                            "stage": "review",
                        })
                        break
                continue

            # All other stages: run with retry
            self._run_stage_with_retry(stage, pipeline_data, results)

        # Build final manifest
        elapsed = time.time() - start_time
        manifest = self._build_manifest(pipeline_data, results, elapsed)

        self._emit("pipeline_complete", {
            "manifest_path": manifest.get("manifest_path", ""),
            "total_time": round(elapsed, 1),
            "clips_count": len(pipeline_data.get("clips", [])),
        })

        self.announce(
            f"Pipeline complete in {elapsed:.1f}s — "
            f"{len(self.stages_completed)} stages succeeded, "
            f"{len(self.stages_failed)} failed"
        )

        return {
            "manifest": manifest,
            "pipeline_data": pipeline_data,
            "results": results,
            "elapsed": elapsed,
        }

    # ─────────────────────────────────────────────────────────────────────
    # Stage execution with retry logic
    # ─────────────────────────────────────────────────────────────────────
    def _run_stage_with_retry(
        self,
        stage: str,
        pipeline_data: Dict[str, Any],
        results: Dict[str, Any],
    ) -> bool:
        """
        Run a single pipeline stage with retry logic.
        Returns True if the stage succeeded, False if it failed.
        """
        attempts = 0
        max_attempts = self.max_retries + 1

        while attempts < max_attempts:
            attempts += 1
            stage_start = time.time()
            try:
                self._emit("stage_start", {
                    "stage": stage,
                    "attempt": attempts,
                    "expert": WORKBUDDY_EXPERT_MAP.get(stage, stage),
                })

                # Instantiate and run the agent
                agent = self._instantiate_agent(stage)
                if agent is None:
                    raise RuntimeError(f"Unknown stage: {stage}")

                result = agent.run(pipeline_data)

                # Update the shared pipeline data with results
                self._update_pipeline_data(stage, result, pipeline_data)
                results[stage] = result

                duration = time.time() - stage_start
                self.stage_durations[stage] = duration
                self.stages_completed.append(stage)

                self._emit("stage_complete", {
                    "stage": stage,
                    "duration": round(duration, 1),
                    "attempt": attempts,
                    "summary": self._stage_summary(stage, result),
                })

                # Emit clip_found events after the clip stage
                if stage == "clip" and result.get("clips"):
                    for i, clip in enumerate(result["clips"]):
                        self._emit("clip_found", {
                            "index": i,
                            "start": clip.get("start", 0),
                            "end": clip.get("end", 0),
                            "duration": clip.get("duration", 0),
                            "title": clip.get("title", f"Clip {i+1}"),
                            "reason": clip.get("reason", ""),
                        })

                return True

            except Exception as e:
                duration = time.time() - stage_start
                error_msg = f"Stage '{stage}' failed (attempt {attempts}): {e}"
                self.log(error_msg, "error")

                if attempts < max_attempts:
                    self._emit("stage_error", {
                        "stage": stage,
                        "error": str(e),
                        "will_retry": True,
                        "attempt": attempts,
                    })
                    self.log(f"Retrying stage '{stage}'...")
                    time.sleep(1)  # Brief pause before retry
                else:
                    # Final failure
                    self._emit("stage_error", {
                        "stage": stage,
                        "error": str(e),
                        "will_retry": False,
                        "attempt": attempts,
                    })
                    self.stages_failed.append(stage)
                    self.errors.append(error_msg)

                    gemini_required = bool(
                        self.config.get("clip_agent", {})
                        .get("gemini_review", {})
                        .get("required", False)
                    )
                    content_gemini_required = bool(
                        self.config.get("content_analyzer", {})
                        .get("gemini_analysis", {})
                        .get("required", False)
                    )
                    if ((stage == "clip" and gemini_required) or
                            (stage == "content_analysis" and content_gemini_required)):
                        raise RuntimeError(error_msg) from e

                    # Core stages stop the pipeline
                    if stage in self.core_stages:
                        self.announce(
                            f"Core stage '{stage}' failed. Stopping pipeline."
                        )
                        self._emit("pipeline_error", {
                            "error": error_msg,
                            "stage": stage,
                        })
                        return False
                    else:
                        self.announce(
                            f"Non-core stage '{stage}' failed. Continuing..."
                        )
                        return False

        return False

    # ─────────────────────────────────────────────────────────────────────
    # Agent instantiation (dynamic import)
    # ─────────────────────────────────────────────────────────────────────
    def _instantiate_agent(self, stage: str) -> Optional[BaseAgent]:
        """Dynamically import and instantiate the agent for a stage."""
        if stage not in STAGE_AGENTS:
            self.log(f"Unknown stage: {stage}", "error")
            return None

        module_path, class_name = STAGE_AGENTS[stage]
        module = importlib.import_module(module_path)
        agent_class = getattr(module, class_name)
        return agent_class(self.config)

    # ─────────────────────────────────────────────────────────────────────
    # Pipeline data update (Team Shared Space)
    # ─────────────────────────────────────────────────────────────────────
    def _update_pipeline_data(
        self,
        stage: str,
        result: Dict[str, Any],
        pipeline_data: Dict[str, Any],
    ):
        """Update the shared pipeline_data with results from a stage."""
        if stage == "content_analysis":
            pipeline_data["content_analysis"] = result.get("content_analysis", {})
            pipeline_data["full_transcript"] = result.get("full_transcript", [])
            pipeline_data["detected_language"] = result.get("detected_language", "")
            pipeline_data["video_info"] = result.get("video_info", {})

        elif stage == "clip":
            pipeline_data["clips"] = result.get("clips", [])
            pipeline_data["video_info"] = result.get(
                "video_info", pipeline_data.get("video_info", {})
            )

        elif stage in ("subtitle", "audio", "thumbnail", "social_media", "format"):
            # These agents update clips in-place
            if "clips" in result:
                pipeline_data["clips"] = result["clips"]

        elif stage == "quality":
            if "clips" in result:
                pipeline_data["clips"] = result["clips"]
            pipeline_data["quality_report"] = result.get("quality_report", {})

    # ─────────────────────────────────────────────────────────────────────
    # Review gate — pause for human approval
    # ─────────────────────────────────────────────────────────────────────
    def _pause_for_review(self, pipeline_data: Dict[str, Any]):
        """
        Pause the pipeline after the Clip stage for human review.
        Emits a review_ready event, then blocks until /approve is called.
        """
        clips = pipeline_data.get("clips", [])
        self._awaiting_review = True

        clip_summaries = []
        for i, clip in enumerate(clips):
            clip_summaries.append({
                "index": i,
                "filename": clip.get("filename", f"clip_{i+1}.mp4"),
                "start": clip.get("start", 0),
                "end": clip.get("end", 0),
                "duration": clip.get("duration", 0),
                "title": clip.get("title", f"Clip {i+1}"),
                "reason": clip.get("reason", ""),
                "transcript_preview": clip.get("transcript_preview", ""),
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
            })

        self._emit("review_ready", {
            "clips": clip_summaries,
            "message": "Review and approve clips before final processing",
            "clips_count": len(clips),
        })

        self.announce(
            f"Review gate: {len(clips)} clips ready for approval. "
            f"Waiting for human review..."
        )

        # Wait for approval (with timeout to prevent deadlock)
        self.review_gate.clear()
        if self.review_gate_timeout > 0:
            approved = self.review_gate.wait(timeout=self.review_gate_timeout)
        else:
            self.review_gate.wait()
            approved = True

        if not approved:
            # Timeout — auto-approve all clips
            self.log("Review gate timeout — auto-approving all clips")
            self._emit("review_timeout", {
                "message": "Review timeout — auto-approved all clips",
            })

        self._awaiting_review = False

        # Apply approved clips if provided
        if self.approved_clips is not None:
            pipeline_data["clips"] = self.approved_clips
            self.announce(
                f"Review complete — {len(self.approved_clips)} clips approved"
            )
        else:
            self.announce("Review complete — all clips approved")

    def set_approved_clips(self, clips: List[Dict[str, Any]]):
        """Set the approved clips and release the review gate.
        Called by PipelineRunner when /approve is received.
        """
        self.approved_clips = clips
        self.review_gate.set()

    def is_awaiting_review(self) -> bool:
        """Check if the pipeline is paused at the review gate."""
        return self._awaiting_review

    # ─────────────────────────────────────────────────────────────────────
    # Progress event emission
    # ─────────────────────────────────────────────────────────────────────
    def _emit(self, event_type: str, data: Dict[str, Any]):
        """Send a progress event to the callback (for SSE streaming)."""
        event = {
            "event_type": event_type,
            "data": data,
            "timestamp": datetime.now().isoformat(),
        }
        if self.progress_callback:
            try:
                self.progress_callback(event)
            except Exception as e:
                self.log(f"Progress callback error: {e}", "warning")

    # ─────────────────────────────────────────────────────────────────────
    # Stage summary (for progress events)
    # ─────────────────────────────────────────────────────────────────────
    def _stage_summary(self, stage: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a brief summary of a stage's result."""
        if stage == "content_analysis":
            ca = result.get("content_analysis", {})
            return {
                "topics": len(ca.get("topics", [])),
                "chapters": len(ca.get("chapters", [])),
                "key_moments": len(ca.get("key_moments", [])),
                "language": ca.get("detected_language", ""),
            }
        elif stage == "clip":
            return {"clips": len(result.get("clips", []))}
        elif stage == "subtitle":
            return {"clips_with_subtitles": len(result.get("clips", []))}
        elif stage == "audio":
            clips = result.get("clips", [])
            total_translations = sum(
                len(c.get("audio_translations", {})) for c in clips
            )
            return {"audio_translations": total_translations}
        elif stage == "thumbnail":
            clips = result.get("clips", [])
            total_thumbs = sum(len(c.get("thumbnails", [])) for c in clips)
            return {"thumbnails": total_thumbs}
        elif stage == "social_media":
            return {"clips_with_copy": len(result.get("clips", []))}
        elif stage == "format":
            clips = result.get("clips", [])
            total_versions = sum(
                len(c.get("platform_versions", {})) for c in clips
            )
            return {"platform_versions": total_versions}
        elif stage == "quality":
            qr = result.get("quality_report", {}).get("summary", {})
            return qr
        return {}

    # ─────────────────────────────────────────────────────────────────────
    # Manifest builder
    # ─────────────────────────────────────────────────────────────────────
    def _build_manifest(
        self,
        pipeline_data: Dict[str, Any],
        results: Dict[str, Any],
        elapsed: float,
    ) -> Dict[str, Any]:
        """Build the final manifest with all generated content paths."""
        config = self.config
        manifest = {
            "project": "OPC AI Video Studio",
            "run_id": self.run_id,
            "output_dir": config.get("paths", {}).get("output_dir", "output"),
            "video": pipeline_data.get("video_name", ""),
            "video_path": pipeline_data.get("video_path", ""),
            "processing_time_seconds": round(elapsed, 1),
            "stages_completed": self.stages_completed,
            "stages_failed": self.stages_failed,
            "errors": self.errors,
            "config": {
                "target_languages": config.get("subtitle_agent", {}).get(
                    "target_languages", []
                ),
                "max_clips": config.get("clip_agent", {}).get("max_clips", 5),
                "whisper_model": config.get("subtitle_agent", {}).get(
                    "whisper_model", "small"
                ),
            },
            "content_analysis": {},
            "clips": [],
        }

        # Content analysis summary
        ca = pipeline_data.get("content_analysis")
        if ca:
            manifest["content_analysis"] = {
                "detected_language": ca.get("detected_language", ""),
                "total_segments": ca.get("total_segments", 0),
                "total_words": ca.get("total_words", 0),
                "topics": [t["term"] for t in ca.get("topics", [])],
                "chapters": len(ca.get("chapters", [])),
                "key_moments": len(ca.get("key_moments", [])),
                "summary": ca.get("summary", ""),
                "overall_sentiment": ca.get("sentiment", {}).get("overall", ""),
                "content_type": ca.get("content_type", "unknown"),
                "entities": ca.get("entities", []),
                "corrected_terms": ca.get("corrected_terms", []),
                "gemini_analyzed": ca.get("gemini_analyzed", False),
                "gemini_model": ca.get("gemini_model"),
                "gemini_chunks_analyzed": ca.get("gemini_chunks_analyzed", 0),
            }

        # Per-clip details
        for clip in pipeline_data.get("clips", []):
            clip_manifest = {
                "filename": clip.get("filename", ""),
                "path": clip.get("path", ""),
                "duration": clip.get("duration", 0),
                "start": clip.get("start", 0),
                "end": clip.get("end", 0),
                "title": clip.get("title", ""),
                "reason": clip.get("reason", ""),
                "detected_language": clip.get("detected_language", ""),
                "hook_score": clip.get("hook_score", 0),
                "hook_score_breakdown": clip.get("hook_score_breakdown", {}),
                "topic_relevance": clip.get("topic_relevance", 0),
                "matched_topic": clip.get("matched_topic", ""),
                "is_topic_relevant": clip.get("is_topic_relevant", False),
                "selection_score": clip.get("selection_score", 0),
                "gemini_video_reviewed": clip.get("gemini_video_reviewed", False),
                "gemini_video_accepted": clip.get("gemini_video_accepted"),
                "gemini_video_score": clip.get("gemini_video_score"),
                "gemini_video_score_breakdown": clip.get(
                    "gemini_video_score_breakdown", {}
                ),
                "gemini_event_type": clip.get("gemini_event_type", ""),
                "gemini_video_reason": clip.get("gemini_video_reason", ""),
                "pre_gemini_video_start": clip.get("pre_gemini_video_start"),
                "pre_gemini_video_end": clip.get("pre_gemini_video_end"),
            }

            if clip.get("subtitles"):
                clip_manifest["subtitles"] = {}
                for lang, lang_data in clip["subtitles"].items():
                    clip_manifest["subtitles"][lang] = {
                        "formats": {
                            k: v.get("path", "")
                            for k, v in lang_data.get("formats", {}).items()
                        },
                        "segment_count": len(lang_data.get("segments", [])),
                    }

            if clip.get("audio_translations"):
                clip_manifest["audio_translations"] = {}
                for lang, lang_data in clip["audio_translations"].items():
                    clip_manifest["audio_translations"][lang] = {
                        "video_path": lang_data.get("video_path", ""),
                        "voiceover_path": lang_data.get("voiceover_path", ""),
                    }

            if clip.get("thumbnails"):
                clip_manifest["thumbnails"] = [
                    t.get("path", "") for t in clip["thumbnails"]
                ]

            if clip.get("social_media"):
                clip_manifest["social_media"] = clip["social_media"].get(
                    "copy_file", ""
                )

            if clip.get("platform_versions"):
                clip_manifest["platform_versions"] = {
                    k: v.get("path", "")
                    for k, v in clip["platform_versions"].items()
                    if "path" in v
                }

            manifest["clips"].append(clip_manifest)

        # Quality report summary
        if "quality" in results:
            qr = results["quality"].get("quality_report", {}).get("summary", {})
            if qr:
                manifest["quality"] = qr

        # Save manifest to disk
        manifest_path = os.path.join(
            config.get("paths", {}).get("output_dir", "output"),
            "manifest.json",
        )
        os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        manifest["manifest_path"] = manifest_path
        self.log(f"Manifest saved to: {manifest_path}")
        return manifest
