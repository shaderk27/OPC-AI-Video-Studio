"""
Audio Utilities
================
Functions for text-to-speech generation and audio processing.
Supports Edge TTS (default; works in mainland China) and Google TTS (gTTS).
"""

import asyncio
import base64
import os
import subprocess
import tempfile
import wave
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


def get_edge_voice(language: str) -> str:
    """Resolve an ISO/BCP-47 language code to a compatible Edge neural voice."""
    normalized = str(language or "").strip().replace("_", "-").lower()
    base = normalized.split("-", 1)[0]
    voice = EDGE_VOICES.get(normalized) or EDGE_VOICES.get(base)
    if not voice:
        raise ValueError(f"No Edge TTS voice mapped for language '{language}'")
    return voice


def _edge_tts_save(text: str, output_path: str, lang: str) -> None:
    """Synthesize speech with Microsoft Edge TTS and save to output_path."""
    import edge_tts

    voice = get_edge_voice(lang)

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
    allow_gtts_fallback: bool = True,
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
            if not allow_gtts_fallback:
                raise
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


def gemini_text_to_speech(
    text: str,
    output_path: str,
    *,
    style: str,
    voice: str = "Kore",
    model: str = "gemini-3.1-flash-tts-preview",
    client=None,
) -> str:
    """Generate a mono 24 kHz WAV cue with Gemini TTS."""
    if not text.strip():
        raise ValueError("Cannot synthesize empty dubbing text")
    if client is None:
        from google import genai
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        client = genai.Client(api_key=api_key)

    prompt = f"{style.strip() or 'Speak naturally'}:\n{text.strip()}"
    interaction = client.interactions.create(
        model=model,
        input=prompt,
        response_format={"type": "audio"},
        generation_config={
            "speech_config": [{"voice": voice}],
        },
    )
    audio_data = getattr(getattr(interaction, "output_audio", None), "data", None)
    if not audio_data:
        raise RuntimeError("Gemini TTS returned no audio data")
    if isinstance(audio_data, str):
        audio_data = base64.b64decode(audio_data)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with wave.open(output_path, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(24000)
        stream.writeframes(audio_data)
    return output_path


def normalize_audio_to_wav(input_path: str, output_path: str) -> str:
    """Normalize any supported cue audio to mono 24 kHz PCM WAV."""
    import imageio_ffmpeg

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", input_path,
        "-af", _EDGE_SILENCE_TRIM_FILTER,
        "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le", output_path,
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Audio normalization failed: {result.stderr[:500]}")
    return output_path


_EDGE_SILENCE_TRIM_FILTER = (
    "silenceremove=start_periods=1:start_silence=0.03:start_threshold=-50dB,"
    "areverse,"
    "silenceremove=start_periods=1:start_silence=0.03:start_threshold=-50dB,"
    "areverse"
)


def trim_audio_silence(input_path: str, output_path: str) -> str:
    """Trim only leading/trailing digital silence while retaining speech pauses."""
    import imageio_ffmpeg

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", input_path,
        "-af", _EDGE_SILENCE_TRIM_FILTER,
        "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le", output_path,
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Audio silence trim failed: {result.stderr[:500]}")
    return output_path


def build_aligned_voiceover(
    cue_audio: List[Dict],
    output_path: str,
    total_duration: float,
) -> str:
    """Place independently synthesized cues on their approved timeline."""
    import imageio_ffmpeg

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    if not cue_audio:
        command = [
            ffmpeg, "-y", "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono",
            "-t", f"{total_duration:.3f}", "-c:a", "pcm_s16le", output_path,
        ]
    else:
        command = [ffmpeg, "-y"]
        filters = []
        labels = []
        for index, cue in enumerate(cue_audio):
            command.extend(["-i", cue["path"]])
            delay = max(0, int(round(float(cue["start"]) * 1000)))
            label = f"cue{index}"
            filters.append(
                f"[{index}:a]aresample=24000,adelay={delay}|{delay}[{label}]"
            )
            labels.append(f"[{label}]")
        filters.append(
            f"{''.join(labels)}amix=inputs={len(labels)}:duration=longest:normalize=0,"
            f"alimiter=limit=0.95,apad=whole_dur={total_duration:.3f}[out]"
        )
        command.extend([
            "-filter_complex", ";".join(filters), "-map", "[out]",
            "-t", f"{total_duration:.3f}", "-ac", "1", "-ar", "24000",
            "-c:a", "pcm_s16le", output_path,
        ])
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Voiceover alignment failed: {result.stderr[:500]}")
    return output_path
