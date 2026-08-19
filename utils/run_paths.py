"""Create safe, isolated output directories for pipeline runs."""

import copy
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple


RUN_SUBDIRECTORIES = {
    "clips_dir": "clips",
    "subtitles_dir": "subtitles",
    "audio_dir": "audio",
    "thumbnails_dir": "thumbnails",
    "social_dir": "social",
    "formatted_dir": "formatted",
    "burned_dir": "burned",
}


def safe_path_part(
    value: str,
    fallback: str = "item",
    max_length: int = 60,
) -> str:
    """Return a readable filename component without path separators."""
    normalized = unicodedata.normalize("NFKC", str(value or "")).strip().lower()
    cleaned = "".join(
        character if character.isalnum() else "-"
        for character in normalized
    )
    cleaned = re.sub(r"-+", "-", cleaned).strip("-")
    return (cleaned[:max_length].rstrip("-") or fallback)


def prepare_run_output(
    config: dict,
    video_path: str,
    run_id: str,
    started_at: Optional[datetime] = None,
) -> Tuple[dict, Path]:
    """Clone config and point all output paths at a unique dated run folder.

    Folder format: ``<output root>/YYYY-MM-DD/HH-MM-SS_video-name_run-id``.
    The final directory is created exclusively so an existing run can never be
    overwritten silently.
    """
    prepared = copy.deepcopy(config)
    paths = prepared.setdefault("paths", {})
    output_root = Path(paths.get("output_dir", "output"))

    timestamp = started_at or datetime.now().astimezone()
    video_slug = safe_path_part(Path(video_path).stem, "video")
    run_slug = safe_path_part(run_id, "run", max_length=24)[-8:]
    run_name = f"{timestamp:%H-%M-%S}_{video_slug}_{run_slug}"
    run_dir = output_root / f"{timestamp:%Y-%m-%d}" / run_name

    run_dir.parent.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(exist_ok=False)

    paths["output_root"] = str(output_root)
    paths["output_dir"] = str(run_dir)
    for key, folder_name in RUN_SUBDIRECTORIES.items():
        directory = run_dir / folder_name
        directory.mkdir()
        paths[key] = str(directory)

    prepared["run"] = {
        "id": str(run_id),
        "started_at": timestamp.isoformat(),
        "output_dir": str(run_dir),
    }
    return prepared, run_dir
