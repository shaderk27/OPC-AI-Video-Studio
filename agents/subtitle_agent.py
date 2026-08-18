"""
Subtitle Expert Agent
======================
Role: Multi-Language Subtitle Generator

This agent takes video clips and generates subtitles using speech-to-text
(faster-whisper), then translates those subtitles into user-requested languages.

Pipeline:
1. Extract audio from clip
2. Transcribe audio to text segments with timestamps
3. Detect source language
4. Translate segments to each target language
5. Export in multiple formats (SRT, VTT, TXT)

Output: Subtitle files for each clip in each requested language.
"""

import os
from pathlib import Path
from typing import Any, Dict, List

from agents.base_agent import BaseAgent
from utils.video_utils import extract_audio
from utils.subtitle_utils import (
    transcribe_audio,
    translate_segments,
    detect_language_from_segments,
    segments_to_srt,
    segments_to_vtt,
    segments_to_txt,
    format_timestamp,
)


class SubtitleAgent(BaseAgent):
    """AI agent that generates and translates subtitles for video clips."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(
            name="SubtitleExpert",
            role="Multi-Language Subtitle Generator",
            config=config,
        )
        self.sub_config = config.get("subtitle_agent", {})
        self.paths = config.get("paths", {})
        self.whisper_model = self.sub_config.get("whisper_model", "base")
        self.source_language = self.sub_config.get("source_language")
        self.target_languages = self.sub_config.get("target_languages", ["en"])
        self.formats = self.sub_config.get("formats", ["srt", "vtt"])
        self.max_chars = self.sub_config.get("max_chars_per_line", 42)
        self.max_duration = self.sub_config.get("max_subtitle_duration", 7.0)

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate subtitles for all clips.

        input_data: {"clips": [{"path", "filename", ...}]}
        returns: {"clips": [... with added "subtitles" field]}
        """
        clips = input_data.get("clips", [])
        subtitles_dir = self.paths.get("subtitles_dir", "output/subtitles")
        self.ensure_dir(subtitles_dir)

        updated_clips = []

        for i, clip in enumerate(clips):
            clip_path = clip["path"]
            clip_name = Path(clip["filename"]).stem
            self.log(f"Processing clip {i+1}/{len(clips)}: {clip_name}")

            # Step 1: Extract audio from clip
            temp_audio = os.path.join(subtitles_dir, f"{clip_name}_audio.wav")
            extract_audio(clip_path, temp_audio)

            # Step 2: Transcribe
            self.log(f"  Transcribing (model: {self.whisper_model})...")
            segments = transcribe_audio(
                temp_audio,
                model_size=self.whisper_model,
                language=self.source_language,
            )

            # Clean up temp audio
            try:
                os.remove(temp_audio)
            except OSError:
                pass

            if not segments:
                self.log(f"  WARNING: No speech detected in {clip_name}")
                clip["subtitles"] = {}
                updated_clips.append(clip)
                continue

            # Step 3: Detect source language
            detected_lang = detect_language_from_segments(segments)
            self.log(f"  Detected source language: {detected_lang}")

            # Step 4: Split long segments for better readability
            segments = self._split_long_segments(segments)

            # Step 5: Generate subtitles for source language + all target languages
            clip_subtitles = {}
            languages_to_process = set([detected_lang] + self.target_languages)

            for lang in languages_to_process:
                self.log(f"  Generating subtitles for: {lang}")

                if lang == detected_lang:
                    # Source language - no translation needed
                    lang_segments = segments
                else:
                    # Translate
                    lang_segments = translate_segments(segments, lang, source_lang=detected_lang)

                lang_subtitles = {}

                for fmt in self.formats:
                    ext = fmt
                    sub_filename = f"{clip_name}_{lang}.{ext}"
                    sub_path = os.path.join(subtitles_dir, sub_filename)

                    if fmt == "srt":
                        segments_to_srt(lang_segments, sub_path)
                    elif fmt == "vtt":
                        segments_to_vtt(lang_segments, sub_path)

                    lang_subtitles[fmt] = {
                        "path": sub_path,
                        "filename": sub_filename,
                    }

                # Also export plain text transcript
                txt_filename = f"{clip_name}_{lang}.txt"
                txt_path = os.path.join(subtitles_dir, txt_filename)
                segments_to_txt(lang_segments, txt_path)
                lang_subtitles["txt"] = {
                    "path": txt_path,
                    "filename": txt_filename,
                }

                clip_subtitles[lang] = {
                    "formats": lang_subtitles,
                    "segments": lang_segments,
                }

            clip["subtitles"] = clip_subtitles
            clip["detected_language"] = detected_lang
            updated_clips.append(clip)

        self.log(f"Generated subtitles for {len(updated_clips)} clips "
                 f"in {len(self.target_languages)} languages")
        return {"clips": updated_clips, **{k: v for k, v in input_data.items() if k != "clips"}}

    def _split_long_segments(self, segments: List[Dict]) -> List[Dict]:
        """
        Split segments that are too long or have too many characters.
        Ensures subtitle readability.
        """
        result = []
        for seg in segments:
            text = seg["text"]
            duration = seg["end"] - seg["start"]

            # Check if splitting is needed
            needs_split = (
                len(text) > self.max_chars * 2 or
                duration > self.max_duration
            )

            if not needs_split:
                result.append(seg)
                continue

            # Split by sentence boundaries first
            import re
            sentences = re.split(r'(?<=[.!?。！？])\s+', text)

            if len(sentences) <= 1:
                # Split by commas or just by character count
                if len(text) > self.max_chars * 2:
                    mid = len(text) // 2
                    # Find nearest space
                    for offset in range(min(20, mid)):
                        if text[mid - offset] == ' ':
                            mid = mid - offset
                            break
                        if text[mid + offset] == ' ':
                            mid = mid + offset
                            break
                    sentences = [text[:mid].strip(), text[mid:].strip()]
                else:
                    result.append(seg)
                    continue

            # Distribute time across sentences
            total_chars = sum(len(s) for s in sentences)
            current_time = seg["start"]
            seg_duration = seg["end"] - seg["start"]

            for sentence in sentences:
                if not sentence.strip():
                    continue
                sentence_duration = (len(sentence) / total_chars) * seg_duration
                result.append({
                    "start": round(current_time, 3),
                    "end": round(current_time + sentence_duration, 3),
                    "text": sentence.strip(),
                })
                current_time += sentence_duration

        return result
