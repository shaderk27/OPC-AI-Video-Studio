"""
Audio Expert Agent
===================
Role: Audio Translation & Voiceover Specialist

This agent takes video clips and their transcribed subtitles, translates
the audio content into user-requested languages using a TTS pipeline,
and merges the translated voiceover back into the video.

Pipeline:
1. Get transcribed segments from Subtitle Agent
2. Translate text to target language
3. Generate speech (TTS) in target language
4. Merge translated audio with original video (original audio ducked)
5. Output: Video clip with translated voiceover

Output: For each clip, a translated version per target language.
"""

import os
from pathlib import Path
from typing import Any, Dict, List

from agents.base_agent import BaseAgent
from utils.video_utils import merge_audio_into_video, get_video_info
from utils.audio_utils import (
    translate_and_speak,
    text_to_speech,
    get_audio_duration,
)
from utils.subtitle_utils import translate_segments


class AudioAgent(BaseAgent):
    """AI agent that translates audio into multiple languages."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(
            name="AudioExpert",
            role="Audio Translation & Voiceover Specialist",
            config=config,
        )
        self.audio_config = config.get("audio_agent", {})
        self.paths = config.get("paths", {})
        self.tts_engine = self.audio_config.get("tts_engine", "gtts")
        self.target_languages = self.audio_config.get("target_languages", ["en"])
        self.original_volume = self.audio_config.get("original_audio_volume", 0.15)
        self.voiceover_volume = self.audio_config.get("voiceover_volume", 1.0)
        self.tts_speed = self.audio_config.get("tts_speed", 1.0)

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Translate audio for all clips.

        input_data: {"clips": [{"path", "subtitles": {lang: {"segments": [...]}}}]}
        returns: {"clips": [... with added "audio_translations" field]}
        """
        clips = input_data.get("clips", [])
        audio_dir = self.paths.get("audio_dir", "output/audio")
        self.ensure_dir(audio_dir)

        updated_clips = []

        for i, clip in enumerate(clips):
            clip_path = clip["path"]
            clip_name = Path(clip["filename"]).stem
            self.log(f"Processing clip {i+1}/{len(clips)}: {clip_name}")

            # Get source segments from subtitles
            subtitles = clip.get("subtitles", {})
            detected_lang = clip.get("detected_language", "en")

            # Find source language segments
            source_segments = None
            if detected_lang in subtitles:
                source_segments = subtitles[detected_lang].get("segments")
            elif subtitles:
                # Use first available language
                first_lang = list(subtitles.keys())[0]
                source_segments = subtitles[first_lang].get("segments")

            if not source_segments:
                self.log(f"  WARNING: No subtitle segments for {clip_name}, skipping audio translation")
                clip["audio_translations"] = {}
                updated_clips.append(clip)
                continue

            audio_translations = {}

            for lang in self.target_languages:
                if lang == detected_lang:
                    self.log(f"  Skipping {lang} (same as source language)")
                    continue

                self.log(f"  Translating audio to: {lang}")

                try:
                    # Step 1: Translate segments and generate TTS
                    voiceover_filename = f"{clip_name}_voiceover_{lang}.mp3"
                    voiceover_path = os.path.join(audio_dir, voiceover_filename)

                    translate_and_speak(
                        segments=source_segments,
                        output_path=voiceover_path,
                        target_lang=lang,
                        source_lang=detected_lang,
                        engine=self.tts_engine,
                    )

                    # Step 2: Adjust speed if needed. Combines the configured
                    # tts_speed with a fit-to-clip speedup: when the translated
                    # voiceover is longer than the clip, -shortest would cut the
                    # ending off mid-word. Speed it up so it fits instead.
                    from utils.audio_utils import adjust_audio_speed
                    speed = self.tts_speed
                    try:
                        clip_duration = get_video_info(clip_path)["duration"]
                        vo_duration = get_audio_duration(voiceover_path)
                        if vo_duration > clip_duration + 0.3:
                            fit_speed = vo_duration / clip_duration
                            # Cap at 1.35x so speech doesn't sound rushed
                            speed = min(max(speed, fit_speed), 1.35)
                            self.log(f"  Voiceover {vo_duration:.1f}s > clip "
                                     f"{clip_duration:.1f}s -> fitting at {speed:.2f}x")
                    except Exception as e:
                        self.log(f"  WARNING: could not fit voiceover to clip: {e}")

                    if abs(speed - 1.0) > 0.01:
                        adjusted_path = voiceover_path.replace(".mp3", "_adjusted.mp3")
                        adjust_audio_speed(voiceover_path, adjusted_path, speed)
                        voiceover_path = adjusted_path

                    # Step 3: Merge voiceover with original video
                    translated_clip_filename = f"{clip_name}_{lang}.mp4"
                    translated_clip_path = os.path.join(audio_dir, translated_clip_filename)

                    merge_audio_into_video(
                        video_path=clip_path,
                        audio_path=voiceover_path,
                        output_path=translated_clip_path,
                        original_volume=self.original_volume,
                        voiceover_volume=self.voiceover_volume,
                    )

                    audio_translations[lang] = {
                        "video_path": translated_clip_path,
                        "video_filename": translated_clip_filename,
                        "voiceover_path": voiceover_path,
                        "voiceover_filename": voiceover_filename,
                    }

                    self.log(f"  Created translated clip: {translated_clip_filename}")

                except Exception as e:
                    self.log(f"  ERROR translating to {lang}: {e}", "error")

            clip["audio_translations"] = audio_translations
            updated_clips.append(clip)

        self.log(f"Audio translation complete for {len(updated_clips)} clips")
        return {"clips": updated_clips, **{k: v for k, v in input_data.items() if k != "clips"}}
