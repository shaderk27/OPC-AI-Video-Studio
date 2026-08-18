"""
Audio Utilities
================
Functions for text-to-speech generation and audio processing.
Supports Edge TTS (default; works in mainland China) and Google TTS (gTTS).
"""

import asyncio
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

from gtts import gTTS

from utils.subtitle_utils import translate_text

# Language code mapping for gTTS
TTS_LANG_MAP = {
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

# Translation language map (reuse from subtitle_utils)
TRANSLATE_LANG_MAP = {
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


# Microsoft Edge TTS voices (free, no API key, reachable from mainland China)
EDGE_VOICES = {
    "en": "en-US-JennyNeural",
    "zh": "zh-CN-XiaoxiaoNeural",
    "es": "es-ES-ElviraNeural",
    "fr": "fr-FR-DeniseNeural",
    "de": "de-DE-KatjaNeural",
    "ja": "ja-JP-NanamiNeural",
    "ko": "ko-KR-SunHiNeural",
    "ar": "ar-SA-ZariyahNeural",
    "hi": "hi-IN-SwaraNeural",
    "pt": "pt-BR-FranciscaNeural",
    "ru": "ru-RU-SvetlanaNeural",
    "it": "it-IT-ElsaNeural",
    "tr": "tr-TR-EmelNeural",
    "vi": "vi-VN-HoaiMyNeural",
    "th": "th-TH-PremwadeeNeural",
    "id": "id-ID-GadisNeural",
    "nl": "nl-NL-ColetteNeural",
    "pl": "pl-PL-ZofiaNeural",
    "uk": "uk-UA-PolinaNeural",
    "sv": "sv-SE-SofieNeural",
}


def _edge_tts_save(text: str, output_path: str, lang: str) -> None:
    """Synthesize speech with Microsoft Edge TTS and save to output_path."""
    import edge_tts

    voice = EDGE_VOICES.get(lang, EDGE_VOICES.get(lang.split("-")[0]))
    if not voice:
        raise ValueError(f"No Edge TTS voice mapped for language '{lang}'")

    async def _run():
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(output_path)

    asyncio.run(_run())


def text_to_speech(
    text: str,
    output_path: str,
    lang: str = "en",
    slow: bool = False,
    engine: str = "edge",
) -> str:
    """
    Convert text to speech audio file (MP3).
    engine: "edge" (Microsoft Edge TTS, default, works in mainland China)
            "gtts" (Google TTS — blocked on many CN networks)
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if engine == "edge":
        try:
            _edge_tts_save(text, output_path, lang)
            return output_path
        except Exception as e:
            print(f"  [WARNING] Edge TTS failed for lang '{lang}': {e}")
            # fall through to gTTS (may also fail if Google is blocked)

    tts_lang = TTS_LANG_MAP.get(lang, lang)
    tts = gTTS(text=text, lang=tts_lang, slow=slow)
    tts.save(output_path)

    return output_path


def segments_to_speech(
    segments: List[Dict],
    output_path: str,
    lang: str = "en",
) -> str:
    """
    Convert subtitle segments to continuous speech audio.
    Each segment is spoken sequentially, preserving the narrative flow.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Combine all segment text into one continuous speech
    full_text = " ".join(seg["text"] for seg in segments)

    if not full_text.strip():
        raise ValueError("No text to convert to speech")

    return text_to_speech(full_text, output_path, lang=lang)


def translate_and_speak(
    segments: List[Dict],
    output_path: str,
    target_lang: str,
    source_lang: str = "auto",
    engine: str = "edge",
) -> str:
    """
    Translate subtitle segments and generate speech in target language.
    This is the core of the Audio Expert Agent pipeline.
    Translation uses Google -> MyMemory fallback (see subtitle_utils).
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Translate each segment
    translated_texts = []
    for seg in segments:
        if seg["text"].strip():
            translated = translate_text(seg["text"], target_lang, source_lang)
            translated_texts.append(translated)
        else:
            translated_texts.append("")

    # Combine and generate TTS
    full_text = " ".join(translated_texts)

    if not full_text.strip():
        raise ValueError("No text to convert to speech after translation")

    return text_to_speech(full_text, output_path, lang=target_lang, engine=engine)


def adjust_audio_speed(
    input_path: str,
    output_path: str,
    speed: float = 1.0,
) -> str:
    """
    Adjust audio playback speed using ffmpeg atempo filter.
    """
    import subprocess
    import imageio_ffmpeg

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # atempo supports 0.5-2.0 range; chain for extreme values
    filters = []
    remaining_speed = speed
    while remaining_speed > 2.0:
        filters.append("atempo=2.0")
        remaining_speed /= 2.0
    while remaining_speed < 0.5:
        filters.append("atempo=0.5")
        remaining_speed /= 0.5
    filters.append(f"atempo={remaining_speed}")

    cmd = [
        ffmpeg, "-y",
        "-i", input_path,
        "-filter:a", ",".join(filters),
        "-vn",
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Speed adjustment failed: {result.stderr[:500]}")

    return output_path


def get_audio_duration(audio_path: str) -> float:
    """Get duration of audio file in seconds."""
    # Reuse the ffprobe-free implementation from video_utils
    # (imageio-ffmpeg ships no ffprobe, so duration is derived from ffmpeg output)
    from utils.video_utils import get_audio_duration as _shared_duration
    return _shared_duration(audio_path)
