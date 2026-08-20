"""Adapter for an isolated audio-separator runtime."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict

from utils.video_utils import extract_audio


class AudioSeparationError(RuntimeError):
    pass


def separate_background(
    video_path: str,
    output_dir: str,
    config: Dict[str, Any],
) -> Dict[str, str]:
    """Return an ambience/background stem with the original vocal stem removed."""
    settings = config.get("audio_agent", {}).get("separation", {})
    if not settings.get("enabled", True):
        raise AudioSeparationError("audio separation is disabled by configuration")
    model = settings.get("model", "UVR_MDXNET_KARA_2.onnx")
    model_dir = Path(settings.get("model_dir", "models/audio-separator"))
    executable = settings.get("executable", "")
    candidates = [
        executable,
        str(Path("models/audio-separator/.venv/Scripts/audio-separator.exe")),
        shutil.which("audio-separator") or "",
    ]
    command_path = next(
        (candidate for candidate in candidates if candidate and Path(candidate).is_file()),
        None,
    )
    if not command_path:
        raise AudioSeparationError(
            "audio-separator runtime is not installed; run the audio setup script"
        )

    destination = Path(output_dir) / "stems"
    destination.mkdir(parents=True, exist_ok=True)
    source_audio = destination / f"{Path(video_path).stem}_original.wav"
    extract_audio(video_path, str(source_audio))
    existing = {path.resolve() for path in destination.glob("*") if path.is_file()}
    model_dir.mkdir(parents=True, exist_ok=True)
    command = [
        command_path,
        str(source_audio),
        "--model_filename", model,
        "--model_file_dir", str(model_dir.resolve()),
        "--output_dir", str(destination.resolve()),
        "--output_format", "WAV",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise AudioSeparationError(
            f"audio-separator failed: {(result.stderr or result.stdout)[:600]}"
        )

    created = [
        path for path in destination.glob("*.wav")
        if path.resolve() not in existing and path.resolve() != source_audio.resolve()
    ]
    background = next(
        (
            path for path in created
            if any(token in path.name.lower() for token in (
                "instrumental", "no_vocals", "karaoke", "other",
            ))
        ),
        None,
    )
    if background is None:
        raise AudioSeparationError("audio-separator produced no background stem")
    return {
        "background_path": str(background),
        "source_audio_path": str(source_audio),
        "method": "audio-separator",
        "model": model,
    }
