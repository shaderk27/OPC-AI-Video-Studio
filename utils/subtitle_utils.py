"""
Subtitle Utilities
===================
Functions for transcription, translation, and subtitle file generation.
Uses faster-whisper for speech-to-text and deep-translator for translation.
"""

import os
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from deep_translator import GoogleTranslator


def _google_reachable() -> bool:
    """Probe once whether Google Translate is reachable (blocked on CN networks)."""
    try:
        requests.get("https://translate.google.com", timeout=3)
        return True
    except Exception:
        return False


# Decided once per run: if Google is blocked, every translation goes straight
# to MyMemory (mymemory.translated.net) instead of waiting for a timeout.
GOOGLE_REACHABLE = _google_reachable()

# ─────────────────────────────────────────────────────────────────────────────
# HuggingFace / Whisper model download settings
# huggingface.co is unreachable from many networks (e.g. mainland China).
# hf-mirror.com is a full mirror of huggingface.co and IS reachable there.
# Override at any time by setting the HF_ENDPOINT environment variable.
# ─────────────────────────────────────────────────────────────────────────────
STUDIO_ROOT = Path(__file__).resolve().parent.parent
WHISPER_MODELS_DIR = STUDIO_ROOT / "models" / "whisper"
HF_MIRROR = "https://hf-mirror.com"
HF_OFFICIAL = "https://huggingface.co"


def _resolve_hf_endpoint() -> str:
    """Pick the HuggingFace endpoint, preferring an explicit HF_ENDPOINT env var."""
    if os.environ.get("HF_ENDPOINT"):
        return os.environ["HF_ENDPOINT"]
    os.environ["HF_ENDPOINT"] = HF_MIRROR
    return HF_MIRROR


def _activate_hf_endpoint(endpoint: str) -> None:
    """Apply an endpoint even if huggingface_hub was imported previously."""
    os.environ["HF_ENDPOINT"] = endpoint
    constants = sys.modules.get("huggingface_hub.constants")
    if constants is not None:
        constants.ENDPOINT = endpoint


def _cached_whisper_model_path(model_size: str) -> Optional[Path]:
    """Return a complete local model directory, or ``None``."""
    root = WHISPER_MODELS_DIR
    required_files = ("config.json", "model.bin", "tokenizer.json", "vocabulary.txt")

    def complete(path: Path) -> bool:
        return all((path / filename).is_file() for filename in required_files)

    if not root.exists():
        return None
    # HF cache layout: <root>/models--Systran--faster-whisper-<size>/snapshots/<rev>/model.bin
    cache_dir = root / f"models--Systran--faster-whisper-{model_size}"
    if cache_dir.exists() and (cache_dir / "snapshots").exists():
        for snap in (cache_dir / "snapshots").iterdir():
            if complete(snap):
                return snap
    # local_dir layout: files placed directly in <root>/<size>/
    if complete(root / model_size):
        return root / model_size
    return None


def _whisper_model_cached(model_size: str) -> bool:
    """True if the Whisper model files already exist locally."""
    return _cached_whisper_model_path(model_size) is not None


def _download_whisper_model_direct(model_size: str) -> Path:
    """Download standard Faster Whisper files without Hub metadata requests."""
    if not re.fullmatch(r"[A-Za-z0-9._-]+", model_size):
        raise ValueError(f"Invalid Whisper model name: {model_size!r}")

    destination = WHISPER_MODELS_DIR / model_size
    destination.mkdir(parents=True, exist_ok=True)
    repo = f"Systran/faster-whisper-{model_size}"
    required_files = ("config.json", "model.bin", "tokenizer.json", "vocabulary.txt")

    for filename in required_files:
        target = destination / filename
        if target.is_file() and target.stat().st_size > 0:
            continue
        partial = destination / f"{filename}.part"
        url = f"{HF_MIRROR}/{repo}/resolve/main/{filename}?download=true"
        last_error = None
        for attempt in range(1, 7):
            offset = partial.stat().st_size if partial.is_file() else 0
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            try:
                with requests.get(
                    url, stream=True, timeout=(15, 180),
                    allow_redirects=True, headers=headers,
                ) as response:
                    response.raise_for_status()
                    resumed = offset > 0 and response.status_code == 206
                    mode = "ab" if resumed else "wb"
                    with partial.open(mode) as output:
                        for chunk in response.iter_content(
                            chunk_size=1024 * 1024
                        ):
                            if chunk:
                                output.write(chunk)
                partial.replace(target)
                last_error = None
                break
            except Exception as exc:
                # Keep the .part file. The next request resumes from its
                # current byte count instead of restarting a large model.
                last_error = exc
                if attempt < 6:
                    time.sleep(min(8, attempt * 2))
        if last_error is not None:
            raise last_error
    return destination


def _load_whisper_model(model_size: str, allow_fallback: bool = True):
    """
    Load faster-whisper model, routing downloads through a CN-reachable mirror.
    Models are cached under models/whisper/ so re-runs work fully offline.
    """
    endpoint = _resolve_hf_endpoint()
    # This must happen before importing faster_whisper/huggingface_hub. The Hub
    # reads HF_ENDPOINT during import and otherwise keeps the old endpoint.
    _activate_hf_endpoint(endpoint)
    from faster_whisper import WhisperModel

    kwargs = dict(
        device="cpu",
        compute_type="int8",
        download_root=str(WHISPER_MODELS_DIR),
    )
    cached_path = _cached_whisper_model_path(model_size)
    if cached_path:
        return WhisperModel(str(cached_path), device="cpu", compute_type="int8")

    endpoints = []
    for candidate in (endpoint, HF_MIRROR, HF_OFFICIAL):
        if candidate not in endpoints:
            endpoints.append(candidate)

    errors = []
    for candidate in endpoints:
        _activate_hf_endpoint(candidate)
        try:
            return WhisperModel(model_size, **kwargs)
        except Exception as exc:
            errors.append(f"{candidate}: {exc}")

    # Some mirrors serve files correctly but omit metadata headers required by
    # huggingface_hub. A streamed GET avoids that HEAD/metadata failure.
    try:
        direct_path = _download_whisper_model_direct(model_size)
        return WhisperModel(
            str(direct_path), device="cpu", compute_type="int8"
        )
    except Exception as exc:
        errors.append(f"direct mirror download: {exc}")

    fallback_path = _cached_whisper_model_path("base")
    if allow_fallback and model_size != "base" and fallback_path:
        warnings.warn(
            f"Could not download Whisper '{model_size}'. Falling back to the "
            "cached 'base' model so processing can continue.",
            RuntimeWarning,
            stacklevel=2,
        )
        return WhisperModel(
            str(fallback_path), device="cpu", compute_type="int8"
        )

    error_summary = "\n".join(f"  - {item}" for item in errors)
    raise RuntimeError(
        f"Could not load Whisper model '{model_size}'.\n"
        f"Tried endpoints: {', '.join(endpoints)}\n"
        f"Errors:\n{error_summary or '  - unknown model-loading error'}\n"
        f"Fix options:\n"
        f"  1) pre-download once: python scripts/setup_whisper_model.py\n"
        f"  2) temporarily set whisper_model: \"base\" in config.yaml"
    )


# Language code mapping for Google Translator
LANG_MAP = {
    "en": "en",
    "zh": "zh-CN",
    "es": "es",
    "fr": "fr",
    "de": "de",
    "ja": "ja",
    "ko": "ko",
    "ar": "ar",
    "hi": "hi",
    "pt": "pt",
    "ru": "ru",
    "it": "it",
    "tr": "tr",
    "vi": "vi",
    "th": "th",
    "id": "id",
    "nl": "nl",
    "pl": "pl",
    "uk": "uk",
    "sv": "sv",
}


def transcribe_audio(
    audio_path: str,
    model_size: str = "small",
    language: Optional[str] = None,
    beam_size: int = 5,
    vad_filter: bool = True,
    vad_min_silence_ms: int = 300,
    vocabulary_hints: Optional[List[str]] = None,
) -> List[Dict]:
    """
    Transcribe audio file using faster-whisper.
    Returns list of segment dicts: {start, end, text}
    """
    # Use CPU compute type for compatibility; model downloads are routed
    # through hf-mirror.com when huggingface.co is unreachable.
    model = _load_whisper_model(model_size)

    hints = [
        str(item).strip() for item in (vocabulary_hints or [])
        if str(item).strip()
    ]
    initial_prompt = None
    if hints:
        initial_prompt = "Important names and terms: " + ", ".join(hints[:100])

    segments, _ = model.transcribe(
        audio_path,
        language=language,
        beam_size=max(1, int(beam_size)),
        vad_filter=bool(vad_filter),
        vad_parameters={
            "min_silence_duration_ms": max(0, int(vad_min_silence_ms))
        } if vad_filter else None,
        initial_prompt=initial_prompt,
    )

    result = []
    for seg in segments:
        result.append({
            "start": round(seg.start, 3),
            "end": round(seg.end, 3),
            "text": " ".join(seg.text.split()),
        })

    return result


def _mymemory_translate(text: str, source: str, target: str) -> str:
    """Translate via MyMemory REST API (free, no key, reachable from CN networks)."""
    try:
        resp = requests.get(
            "https://api.mymemory.translated.net/get",
            params={"q": text, "langpair": f"{source}|{target}"},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        translated = (data.get("responseData") or {}).get("translatedText") or ""
        status = data.get("responseStatus")
        bad = translated.upper().startswith("'") and ("INVALID" in translated.upper() or "NO SUPPORT" in translated.upper())
        if status not in (200, "200") or not translated or bad:
            raise ValueError(f"MyMemory API error (status {status}): {translated[:120]}")
        return translated.strip()
    except Exception:
        return ""  # signal failure


def translate_text(text: str, target_lang: str, source_lang: str = "auto") -> str:
    """
    Translate text using Google Translator (free, no API key needed).
    Falls back to MyMemory (api.mymemory.translated.net) when Google is
    unreachable — Google is blocked on many networks (e.g. mainland China).
    """
    if not text.strip():
        return ""

    target = LANG_MAP.get(target_lang, target_lang)
    source = LANG_MAP.get(source_lang, source_lang) if source_lang != "auto" else "auto"

    if GOOGLE_REACHABLE:
        try:
            translated = GoogleTranslator(source=source, target=target).translate(text)
            if translated:
                return translated
        except Exception as e:
            print(f"  [WARNING] Google translation failed: {e}")

    # MyMemory needs a concrete source language (no "auto" support).
    if source != "auto":
        translated = _mymemory_translate(text, source, target)
        if translated:
            return translated

    print(f"  [WARNING] Translation failed for '{text[:50]}...'")
    return text  # Return original text on failure


def translate_segments(
    segments: List[Dict],
    target_lang: str,
    source_lang: str = "auto",
) -> List[Dict]:
    """
    Translate all segments' text to target language.
    Preserves timing, only translates the text field.
    """
    translated = []
    for seg in segments:
        translated_text = translate_text(seg["text"], target_lang, source_lang)
        translated.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": translated_text,
        })
    return translated


def format_timestamp(seconds: float, format_type: str = "srt") -> str:
    """Format seconds into subtitle timestamp string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)

    if format_type == "srt":
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    elif format_type == "vtt":
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
    else:
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def segments_to_srt(segments: List[Dict], output_path: str) -> str:
    """Convert segments to SRT subtitle file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    lines = []
    for i, seg in enumerate(segments, 1):
        start_ts = format_timestamp(seg["start"], "srt")
        end_ts = format_timestamp(seg["end"], "srt")
        lines.append(str(i))
        lines.append(f"{start_ts} --> {end_ts}")
        lines.append(seg["text"])
        lines.append("")  # Blank line between entries

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return output_path


def segments_to_vtt(segments: List[Dict], output_path: str) -> str:
    """Convert segments to WebVTT subtitle file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    lines = ["WEBVTT", ""]

    for i, seg in enumerate(segments, 1):
        start_ts = format_timestamp(seg["start"], "vtt")
        end_ts = format_timestamp(seg["end"], "vtt")
        lines.append(f"{i}")
        lines.append(f"{start_ts} --> {end_ts}")
        lines.append(seg["text"])
        lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return output_path


def segments_to_txt(segments: List[Dict], output_path: str) -> str:
    """Export plain text transcript."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    text = " ".join(seg["text"] for seg in segments)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    return output_path


def detect_language_from_segments(segments: List[Dict]) -> str:
    """
    Detect the dominant language from transcribed segments.
    Returns ISO 639-1 language code.
    """
    if not segments:
        return "en"

    # Use the first few segments to detect
    sample_text = " ".join(seg["text"] for seg in segments[:5])

    # Simple heuristic: check for CJK characters
    cjk_count = sum(1 for c in sample_text if '\u4e00' <= c <= '\u9fff')
    if cjk_count > len(sample_text) * 0.3:
        return "zh"

    # Check for other scripts
    hiragana = sum(1 for c in sample_text if '\u3040' <= c <= '\u309f')
    katakana = sum(1 for c in sample_text if '\u30a0' <= c <= '\u30ff')
    if hiragana + katakana > 5:
        return "ja"

    hangul = sum(1 for c in sample_text if '\uac00' <= c <= '\ud7af')
    if hangul > 5:
        return "ko"

    # Default to English for Latin script
    return "en"
