"""
OPC AI Video Studio — Pipeline Runner
Async wrapper that runs the ManagerAgent in a background thread,
manages pause/resume at the review gate, and provides status/events.
"""

import asyncio
import json
import os
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from agents.manager_agent import ManagerAgent
from utils.run_paths import prepare_run_output


# ─────────────────────────────────────────────────────────────────────────────
# Run state tracking
# ─────────────────────────────────────────────────────────────────────────────

class RunState:
    """Tracks the state of a single pipeline run."""

    def __init__(self, run_id: str, video_path: str, video_name: str,
                 config: dict, skip_stages: List[str] = None):
        self.run_id = run_id
        self.video_path = video_path
        self.video_name = video_name
        self.config = config
        self.skip_stages = skip_stages or []

        # Status tracking
        self.status: str = "running"  # running, awaiting_review, complete, error
        self.current_stage: Optional[str] = None
        self.stages_completed: List[str] = []
        self.stages_failed: List[str] = []
        self.errors: List[str] = []
        self.clips_count: int = 0

        # Event log (for clients that connect late)
        self.events: List[Dict[str, Any]] = []

        # Results
        self.manifest: Optional[Dict] = None
        self.pipeline_data: Optional[Dict] = None
        self.elapsed: Optional[float] = None
        self.output_dir: str = config.get("paths", {}).get("output_dir", "output")

        # Timing
        self.created_at: str = datetime.now().isoformat()
        self.start_time: float = time.time()

        # Review gate
        self.review_clips: List[Dict] = []
        self._review_event = threading.Event()
        self._manager: Optional[ManagerAgent] = None


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline Runner — manages all runs
# ─────────────────────────────────────────────────────────────────────────────

class PipelineRunner:
    """Manages background pipeline execution with pause/resume support."""

    def __init__(self):
        self._runs: Dict[str, RunState] = {}
        self._lock = threading.Lock()

    # ─────────────────────────────────────────────────────────────────────
    # Run management
    # ─────────────────────────────────────────────────────────────────────

    def start_run(
        self,
        video_path: str,
        video_name: str,
        config: dict,
        skip_stages: List[str] = None,
    ) -> str:
        """Start a new pipeline run in a background thread. Returns run_id."""
        run_id = str(uuid.uuid4())[:8]
        config, _ = prepare_run_output(
            config,
            video_path,
            run_id,
            datetime.now().astimezone(),
        )

        with self._lock:
            state = RunState(run_id, video_path, video_name, config, skip_stages)
            self._runs[run_id] = state

        # Start the pipeline in a background thread
        thread = threading.Thread(
            target=self._run_pipeline_thread,
            args=(run_id,),
            daemon=True,
        )
        thread.start()

        return run_id

    def _run_pipeline_thread(self, run_id: str):
        """Background thread that runs the ManagerAgent."""
        state = self._runs[run_id]

        try:
            # Create the ManagerAgent with our progress callback and review gate
            def progress_callback(event: dict):
                self._handle_event(run_id, event)

            manager = ManagerAgent(
                config=state.config,
                progress_callback=progress_callback,
                review_gate=state._review_event,
            )
            state._manager = manager
            state.run_id = run_id  # sync the manager's run_id

            # Run the pipeline
            result = manager.execute({
                "video_path": state.video_path,
                "video_name": state.video_name,
                "skip_stages": state.skip_stages,
            })

            # Store results
            state.manifest = result.get("manifest")
            state.pipeline_data = result.get("pipeline_data")
            state.elapsed = result.get("elapsed")
            state.status = "complete"

        except Exception as e:
            state.status = "error"
            state.errors.append(str(e))
            state._handle_event_error(e)

    # ─────────────────────────────────────────────────────────────────────
    # Event handling
    # ─────────────────────────────────────────────────────────────────────

    def _handle_event(self, run_id: str, event: dict):
        """Process a progress event from the ManagerAgent."""
        state = self._runs.get(run_id)
        if not state:
            return

        event_type = event.get("event_type", "")
        data = event.get("data", {})

        # Store event in log
        state.events.append(event)

        # Update state based on event type
        if event_type == "stage_start":
            state.current_stage = data.get("stage")

        elif event_type == "stage_complete":
            stage = data.get("stage")
            if stage and stage not in state.stages_completed:
                state.stages_completed.append(stage)

        elif event_type == "stage_error":
            stage = data.get("stage")
            if stage and stage not in state.stages_failed:
                state.stages_failed.append(stage)
            if data.get("error"):
                state.errors.append(data["error"])

        elif event_type == "clip_found":
            state.clips_count = max(state.clips_count, data.get("index", 0) + 1)

        elif event_type == "review_ready":
            state.status = "awaiting_review"
            state.review_clips = data.get("clips", [])

        elif event_type == "pipeline_complete":
            state.status = "complete"

        elif event_type == "pipeline_error":
            state.status = "error"

    # ─────────────────────────────────────────────────────────────────────
    # Status queries
    # ─────────────────────────────────────────────────────────────────────

    def get_status(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get the current status of a run."""
        state = self._runs.get(run_id)
        if not state:
            return None

        elapsed = time.time() - state.start_time if state.elapsed is None else state.elapsed

        return {
            "run_id": run_id,
            "video_id": state.video_name,
            "status": state.status,
            "current_stage": state.current_stage,
            "stages_completed": state.stages_completed,
            "stages_failed": state.stages_failed,
            "clips_count": state.clips_count,
            "errors": state.errors,
            "elapsed_seconds": round(elapsed, 1),
            "created_at": state.created_at,
            "message": self._status_message(state),
        }

    def _status_message(self, state: RunState) -> str:
        """Generate a human-readable status message."""
        if state.status == "running":
            if state.current_stage:
                return f"Running stage: {state.current_stage}"
            return "Pipeline running"
        elif state.status == "awaiting_review":
            return f"Waiting for clip review ({len(state.review_clips)} clips)"
        elif state.status == "complete":
            return f"Complete — {state.clips_count} clips generated"
        elif state.status == "error":
            return f"Error: {state.errors[-1] if state.errors else 'unknown'}"
        return state.status

    def get_events(self, run_id: str, after_index: int = 0) -> List[Dict]:
        """Get events for a run, optionally after a given index (for SSE)."""
        state = self._runs.get(run_id)
        if not state:
            return []
        return state.events[after_index:]

    def get_clips(self, run_id: str) -> Optional[List[Dict]]:
        """Get the clips for a run (after clip stage has run)."""
        state = self._runs.get(run_id)
        if not state:
            return None

        # If we have pipeline_data with clips, return those
        if state.pipeline_data and state.pipeline_data.get("clips"):
            return state.pipeline_data["clips"]

        # If awaiting review, return the review clips
        if state.status == "awaiting_review":
            return state.review_clips

        return []

    def get_manifest(self, run_id: str) -> Optional[Dict]:
        """Get the final manifest for a completed run."""
        state = self._runs.get(run_id)
        if not state or not state.manifest:
            return None
        return state.manifest

    # ─────────────────────────────────────────────────────────────────────
    # Review gate — approve clips
    # ─────────────────────────────────────────────────────────────────────

    def approve(
        self,
        run_id: str,
        approved_indices: List[int] = None,
        edits: List[Dict] = None,
        approve_all: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Approve clips at the review gate and resume the pipeline."""
        state = self._runs.get(run_id)
        if not state:
            return None

        if state.status != "awaiting_review":
            return {
                "error": f"Run is not awaiting review (status: {state.status})",
                "status": state.status,
            }

        clips = state.review_clips if state.review_clips else []

        # Determine which clips to approve
        if approve_all:
            approved_index_set = set(range(len(clips)))
        elif approved_indices is not None:
            approved_index_set = {
                index for index in approved_indices
                if 0 <= index < len(clips)
            }
        else:
            approved_index_set = set(range(len(clips)))
        approved_clips = [
            clip for index, clip in enumerate(clips)
            if index in approved_index_set
        ]

        # Apply edits if provided
        if edits:
            for edit in edits:
                idx = edit.get("index")
                if idx is not None and idx < len(approved_clips):
                    if "title" in edit:
                        approved_clips[idx]["title"] = edit["title"]
                    if "start" in edit:
                        approved_clips[idx]["start"] = edit["start"]
                    if "end" in edit:
                        approved_clips[idx]["end"] = edit["end"]

        review_payload = {
            "run_id": run_id,
            "video": state.video_name,
            "submitted_at": datetime.now().astimezone().isoformat(),
            "approved_count": len(approved_clips),
            "rejected_count": len(clips) - len(approved_clips),
            "clips": [
                {
                    "index": index,
                    "filename": clip.get("filename", ""),
                    "title": clip.get("title", ""),
                    "start": clip.get("start", 0),
                    "end": clip.get("end", 0),
                    "decision": (
                        "approved" if index in approved_index_set else "dropped"
                    ),
                }
                for index, clip in enumerate(clips)
            ],
        }
        decision_path = Path(state.output_dir) / "review_decisions.json"
        with open(decision_path, "w", encoding="utf-8") as file:
            json.dump(review_payload, file, indent=2, ensure_ascii=False)

        # Set the approved clips on the manager and release the gate
        if state._manager:
            state._manager.set_approved_clips(approved_clips)

        state.status = "running"

        return {
            "run_id": run_id,
            "status": "resumed",
            "approved_count": len(approved_clips),
            "message": "Pipeline resumed",
        }

    # ─────────────────────────────────────────────────────────────────────
    # Utility
    # ─────────────────────────────────────────────────────────────────────

    def list_runs(self) -> List[str]:
        """List all run IDs."""
        return list(self._runs.keys())

    def run_exists(self, run_id: str) -> bool:
        """Check if a run exists."""
        return run_id in self._runs


# ─────────────────────────────────────────────────────────────────────────────
# Global singleton
# ─────────────────────────────────────────────────────────────────────────────

# Single instance shared across the FastAPI app
runner = PipelineRunner()


def load_config(config_path: str = "config.yaml") -> dict:
    """Load the YAML configuration file."""
    if not os.path.isabs(config_path):
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            config_path,
        )
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
