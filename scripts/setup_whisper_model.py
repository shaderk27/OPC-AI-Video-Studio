"""
Pre-download the Whisper model so the first pipeline run doesn't need to fetch it.

The studio downloads the model from hf-mirror.com (automatic fallback when
huggingface.co is unreachable, e.g. on mainland China networks) and caches it
under models/whisper/. Run this once to download it ahead of time:

    python scripts/setup_whisper_model.py
"""

import os
import sys

# Make sure the project root is importable regardless of CWD
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import yaml  # noqa: E402

from utils.subtitle_utils import (  # noqa: E402
    _load_whisper_model,
    _resolve_hf_endpoint,
    WHISPER_MODELS_DIR,
)


def main() -> None:
    config_path = os.path.join(PROJECT_ROOT, "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_size = cfg.get("subtitle_agent", {}).get("whisper_model", "base")

    print(f"Endpoint : {_resolve_hf_endpoint()}")
    print(f"Model    : faster-whisper '{model_size}' (CPU, int8)")
    print(f"Save to  : {WHISPER_MODELS_DIR}")
    print("Downloading... (this can take a few minutes the first time)")
    _load_whisper_model(model_size)
    print("Done. The model is cached locally; the pipeline will use it offline.")


if __name__ == "__main__":
    main()
