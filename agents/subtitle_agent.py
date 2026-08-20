"""Gemini-backed Subtitle Localization Expert."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from agents.base_agent import BaseAgent
from utils.gemini_agent_client import GeminiAgentClient, GeminiAgentError
from utils.subtitle_utils import (
    detect_language_from_segments,
    segments_to_srt,
    segments_to_txt,
    segments_to_vtt,
    transcribe_audio,
)
from utils.video_utils import extract_audio


SUBTITLE_PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "clip_id": {"type": "string"},
        "source_language": {"type": "string"},
        "target_language": {"type": "string"},
        "summary": {"type": "string"},
        "cues": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "cue_id": {"type": "string"},
                    "translated_text": {"type": "string"},
                    "line_break_text": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "warning": {"type": "string"},
                },
                "required": [
                    "cue_id", "translated_text", "line_break_text",
                    "confidence", "warning",
                ],
            },
        },
    },
    "required": [
        "clip_id", "source_language", "target_language", "summary", "cues",
    ],
}


SUBTITLE_SYSTEM_INSTRUCTION = """
You are the Subtitle Localization Expert. Produce accurate, concise subtitles
that preserve meaning, names, numbers, scores, terminology, and emotional tone.
Respect every supplied cue ID and timing window. Never invent facts, omit cues,
or alter cue IDs. line_break_text may contain at most one newline. Return only
the requested JSON schema and no commentary.
""".strip()


class SubtitleAgent(BaseAgent):
    """Plans subtitles with Gemini, then renders approved plans to files."""

    def __init__(self, config: Dict[str, Any], gemini_client: Any = None):
        super().__init__(
            name="SubtitleExpert",
            role="Gemini Subtitle Localization Expert",
            config=config,
        )
        self.sub_config = config.get("subtitle_agent", {})
        self.paths = config.get("paths", {})
        self.whisper_model = self.sub_config.get("whisper_model", "small")
        self.whisper_beam_size = self.sub_config.get("beam_size", 5)
        self.whisper_vad_filter = self.sub_config.get("vad_filter", True)
        self.whisper_vad_min_silence_ms = self.sub_config.get(
            "vad_min_silence_ms", 300
        )
        self.whisper_vocabulary_hints = self.sub_config.get("vocabulary_hints", [])
        self.source_language = self.sub_config.get("source_language")
        self.target_languages = list(self.sub_config.get("target_languages", ["en"]))
        self.formats = self.sub_config.get("formats", ["srt", "vtt"])
        self.max_chars = int(self.sub_config.get("max_chars_per_line", 42))
        self.max_duration = float(self.sub_config.get("max_subtitle_duration", 7.0))
        self.max_reading_speed = float(
            self.sub_config.get("max_reading_chars_per_second", 24.0)
        )
        self.gemini = GeminiAgentClient(
            config=config,
            section="subtitle_agent",
            agent_id="subtitle_expert",
            role="Subtitle Localization Expert",
            system_instruction=SUBTITLE_SYSTEM_INSTRUCTION,
            schema=SUBTITLE_PLAN_SCHEMA,
            client=gemini_client,
        )

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        phase = input_data.get("_localization_phase", "complete")
        if phase == "plan":
            return self.create_plans(input_data)
        if phase == "render":
            return self.render_plans(input_data)
        return self.render_plans(self.create_plans(input_data))

    def create_plans(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Transcribe each clip and create one strict Gemini plan per language."""
        if not self.gemini.available:
            raise GeminiAgentError(
                f"Subtitle Expert requires Gemini: {self.gemini.configuration_error()}"
            )

        clips = input_data.get("clips", [])
        subtitles_dir = self.paths.get("subtitles_dir", "output/subtitles")
        self.ensure_dir(subtitles_dir)
        topics = input_data.get("content_analysis", {}).get("topics", [])

        for index, clip in enumerate(clips):
            clip_path = clip["path"]
            clip_name = Path(clip.get("filename") or clip_path).stem
            self.log(f"Planning subtitles for clip {index + 1}/{len(clips)}: {clip_name}")
            temp_audio = os.path.join(subtitles_dir, f"{clip_name}_subtitle_source.wav")
            try:
                extract_audio(clip_path, temp_audio)
                segments = transcribe_audio(
                    temp_audio,
                    model_size=self.whisper_model,
                    language=self.source_language,
                    beam_size=self.whisper_beam_size,
                    vad_filter=self.whisper_vad_filter,
                    vad_min_silence_ms=self.whisper_vad_min_silence_ms,
                    vocabulary_hints=self.whisper_vocabulary_hints,
                )
            finally:
                try:
                    os.remove(temp_audio)
                except OSError:
                    pass

            if not segments:
                clip["subtitle_plans"] = {}
                clip["source_subtitle_segments"] = []
                clip["subtitles"] = {}
                clip["localization_warning"] = "No speech detected"
                continue

            detected = detect_language_from_segments(segments)
            prepared = self._assign_cue_ids(self._split_long_segments(segments))
            clip["detected_language"] = detected
            clip["source_subtitle_segments"] = prepared
            plans: Dict[str, Dict[str, Any]] = {}
            languages = list(dict.fromkeys([detected, *self.target_languages]))

            for language in languages:
                payload = self._plan_payload(
                    clip_name, detected, language, prepared, topics
                )
                raw, trace_path = self.gemini.generate(
                    payload,
                    clip_id=clip_name,
                    language=language,
                )
                plan, errors = self._validate_plan(
                    raw, clip_name, detected, language, prepared
                )
                trace_paths = [trace_path]
                if errors:
                    repair_payload = {
                        "request": "Repair the rejected subtitle plan.",
                        "validation_errors": errors,
                        "rejected_plan": raw,
                        "original_request": payload,
                    }
                    repaired, repair_trace = self.gemini.generate(
                        repair_payload,
                        clip_id=clip_name,
                        language=language,
                        request_kind="repair",
                        repair_attempt=1,
                    )
                    trace_paths.append(repair_trace)
                    plan, errors = self._validate_plan(
                        repaired, clip_name, detected, language, prepared
                    )
                if errors:
                    raise GeminiAgentError(
                        f"Subtitle plan for {clip_name}/{language} failed validation: "
                        + "; ".join(errors)
                    )
                plan.update({
                    "approved": False,
                    "trace_paths": trace_paths,
                    "agent": "subtitle_expert",
                    "model": self.gemini.model,
                    "prompt_version": self.gemini.prompt_version,
                })
                plans[language] = plan

            clip["subtitle_plans"] = plans
        return self._result(input_data, clips)

    def render_plans(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Render approved plans without making another translation request."""
        clips = input_data.get("clips", [])
        subtitles_dir = self.paths.get("subtitles_dir", "output/subtitles")
        self.ensure_dir(subtitles_dir)

        for clip in clips:
            clip_name = Path(clip.get("filename") or clip["path"]).stem
            rendered: Dict[str, Dict[str, Any]] = {}
            for language, plan in clip.get("subtitle_plans", {}).items():
                if plan.get("approved") is False and input_data.get(
                    "localization_review_applied", False
                ):
                    continue
                segments = [
                    {
                        "start": cue["start"],
                        "end": cue["end"],
                        "text": cue.get("line_break_text")
                        or cue.get("translated_text", ""),
                    }
                    for cue in plan.get("cues", [])
                ]
                formats: Dict[str, Dict[str, str]] = {}
                for fmt in self.formats:
                    path = os.path.join(subtitles_dir, f"{clip_name}_{language}.{fmt}")
                    if fmt == "srt":
                        segments_to_srt(segments, path)
                    elif fmt == "vtt":
                        segments_to_vtt(segments, path)
                    else:
                        continue
                    formats[fmt] = {"path": path, "filename": Path(path).name}
                txt_path = os.path.join(subtitles_dir, f"{clip_name}_{language}.txt")
                segments_to_txt(segments, txt_path)
                formats["txt"] = {"path": txt_path, "filename": Path(txt_path).name}
                rendered[language] = {
                    "formats": formats,
                    "segments": segments,
                    "agent": plan.get("agent"),
                    "model": plan.get("model"),
                    "prompt_version": plan.get("prompt_version"),
                    "trace_paths": plan.get("trace_paths", []),
                    "approved": bool(plan.get("approved", True)),
                }
            clip["subtitles"] = rendered
        return self._result(input_data, clips)

    def _plan_payload(
        self,
        clip_id: str,
        source_language: str,
        target_language: str,
        segments: List[Dict[str, Any]],
        topics: List[Any],
    ) -> Dict[str, Any]:
        return {
            "request": "Create a complete subtitle localization plan.",
            "clip_id": clip_id,
            "source_language": source_language,
            "target_language": target_language,
            "topics": topics[:10],
            "constraints": {
                "max_characters_per_line": self.max_chars,
                "maximum_lines": 2,
                "max_reading_characters_per_second": self.max_reading_speed,
                "timestamps_are_immutable": True,
            },
            "cues": [
                {
                    "cue_id": cue["cue_id"],
                    "start": cue["start"],
                    "end": cue["end"],
                    "source_text": cue["text"],
                }
                for cue in segments
            ],
        }

    def _validate_plan(
        self,
        raw: Dict[str, Any],
        clip_id: str,
        source_language: str,
        target_language: str,
        source_cues: List[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], List[str]]:
        errors: List[str] = []
        if raw.get("clip_id") != clip_id:
            errors.append("clip_id does not match")
        if str(raw.get("target_language", "")).lower() != target_language.lower():
            errors.append("target_language does not match")
        returned = raw.get("cues")
        if not isinstance(returned, list):
            return raw, errors + ["cues must be a list"]

        source_by_id = {cue["cue_id"]: cue for cue in source_cues}
        returned_ids = [cue.get("cue_id") for cue in returned if isinstance(cue, dict)]
        if returned_ids != list(source_by_id):
            errors.append("cue IDs must appear exactly once and in source order")

        normalized = {
            "clip_id": clip_id,
            "source_language": source_language,
            "target_language": target_language,
            "summary": str(raw.get("summary", "")).strip(),
            "cues": [],
        }
        for cue in returned:
            if not isinstance(cue, dict) or cue.get("cue_id") not in source_by_id:
                continue
            source = source_by_id[cue["cue_id"]]
            translated = str(cue.get("translated_text", "")).strip()
            line_break = str(cue.get("line_break_text", translated)).strip()
            if not translated:
                errors.append(f"{cue.get('cue_id')} has empty translated_text")
            lines = line_break.splitlines() or [""]
            if len(lines) > 2 or any(len(line) > self.max_chars for line in lines):
                errors.append(f"{cue.get('cue_id')} exceeds line limits")
            duration = max(0.1, float(source["end"]) - float(source["start"]))
            compact_length = len(re.sub(r"\s+", "", line_break))
            if compact_length / duration > self.max_reading_speed:
                errors.append(f"{cue.get('cue_id')} exceeds reading speed")
            normalized["cues"].append({
                "cue_id": cue["cue_id"],
                "start": float(source["start"]),
                "end": float(source["end"]),
                "source_text": source["text"],
                "translated_text": translated,
                "line_break_text": line_break,
                "confidence": min(1.0, max(0.0, float(cue.get("confidence", 0)))),
                "warning": str(cue.get("warning", "")).strip(),
            })
        combined_text = " ".join(
            cue.get("translated_text", "") for cue in normalized["cues"]
        )
        if combined_text and not self._uses_expected_script(target_language, combined_text):
            errors.append(f"translated text is incompatible with {target_language} script")
        return normalized, list(dict.fromkeys(errors))

    @staticmethod
    def _uses_expected_script(language: str, text: str) -> bool:
        language = language.lower().split("-")[0]
        ranges = {
            "ko": (("\uac00", "\ud7af"),),
            "zh": (("\u3400", "\u9fff"),),
            "ja": (("\u3040", "\u30ff"), ("\u3400", "\u9fff")),
            "ar": (("\u0600", "\u06ff"),),
            "hi": (("\u0900", "\u097f"),),
            "ru": (("\u0400", "\u04ff"),),
            "th": (("\u0e00", "\u0e7f"),),
        }
        expected = ranges.get(language)
        if expected is None:
            return True
        return any(start <= character <= end for start, end in expected for character in text)

    @staticmethod
    def _assign_cue_ids(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [
            {
                "cue_id": f"cue_{index:04d}",
                "start": float(segment["start"]),
                "end": float(segment["end"]),
                "text": str(segment["text"]).strip(),
            }
            for index, segment in enumerate(segments, start=1)
            if str(segment.get("text", "")).strip()
        ]

    def _split_long_segments(self, segments: List[Dict]) -> List[Dict]:
        result: List[Dict[str, Any]] = []
        for segment in segments:
            text = str(segment.get("text", "")).strip()
            duration = float(segment["end"]) - float(segment["start"])
            if len(text) <= self.max_chars * 2 and duration <= self.max_duration:
                result.append(segment)
                continue
            sentences = re.split(r"(?<=[.!?。！？])\s+", text)
            if len(sentences) <= 1:
                words = text.split()
                if len(words) > 1:
                    midpoint = max(1, len(words) // 2)
                    sentences = [" ".join(words[:midpoint]), " ".join(words[midpoint:])]
                else:
                    midpoint = max(1, len(text) // 2)
                    sentences = [text[:midpoint], text[midpoint:]]
            sentences = [item.strip() for item in sentences if item.strip()]
            total = max(1, sum(len(item) for item in sentences))
            current = float(segment["start"])
            for sentence in sentences:
                portion = duration * len(sentence) / total
                result.append({
                    "start": round(current, 3),
                    "end": round(current + portion, 3),
                    "text": sentence,
                })
                current += portion
        return result

    @staticmethod
    def _result(input_data: Dict[str, Any], clips: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {"clips": clips, **{key: value for key, value in input_data.items() if key != "clips"}}
