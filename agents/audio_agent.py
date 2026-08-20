"""Gemini-backed Audio Localization and Dubbing Expert."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from agents.base_agent import BaseAgent
from utils.audio_separation import AudioSeparationError, separate_background
from utils.audio_utils import (
    adjust_audio_speed,
    build_aligned_voiceover,
    gemini_text_to_speech,
    get_audio_duration,
    normalize_audio_to_wav,
    text_to_speech,
)
from utils.gemini_agent_client import GeminiAgentClient, GeminiAgentError
from utils.video_utils import get_video_info, merge_voiceover_with_background


DUBBING_PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "clip_id": {"type": "string"},
        "target_language": {"type": "string"},
        "voice": {"type": "string"},
        "overall_style": {"type": "string"},
        "cues": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "cue_id": {"type": "string"},
                    "start": {"type": "number", "minimum": 0},
                    "end": {"type": "number", "minimum": 0},
                    "spoken_text": {"type": "string"},
                    "performance_direction": {"type": "string"},
                    "estimated_pace": {"type": "string"},
                    "warning": {"type": "string"},
                },
                "required": [
                    "cue_id", "start", "end", "spoken_text",
                    "performance_direction", "estimated_pace", "warning",
                ],
            },
        },
    },
    "required": ["clip_id", "target_language", "voice", "overall_style", "cues"],
}


AUDIO_SYSTEM_INSTRUCTION = """
You are the Audio Localization and Dubbing Expert. Adapt all source speech into
natural spoken target-language narration while preserving facts, names, scores,
emotion, sequence, and intent. Spoken wording must fit each immutable time
window and should not simply copy concise subtitle wording. Never invent facts,
omit cues, change cue IDs, or overlap cue windows. Return only the JSON schema.
""".strip()


class AudioAgent(BaseAgent):
    """Creates a Gemini dubbing plan and renders aligned multilingual audio."""

    def __init__(self, config: Dict[str, Any], gemini_client: Any = None):
        super().__init__(
            name="AudioExpert",
            role="Gemini Audio Localization and Dubbing Expert",
            config=config,
        )
        self.audio_config = config.get("audio_agent", {})
        self.paths = config.get("paths", {})
        self.target_languages = list(self.audio_config.get("target_languages", ["en"]))
        tts = self.audio_config.get("tts", {})
        self.tts_model = tts.get("model", "gemini-3.1-flash-tts-preview")
        self.tts_voice = tts.get("voice", "Kore")
        self.edge_fallback = bool(tts.get("edge_fallback", True))
        self.voiceover_volume = float(self.audio_config.get("voiceover_volume", 1.0))
        self.background_volume = float(self.audio_config.get("background_volume", 0.9))
        self.original_audio_volume = float(
            self.audio_config.get("original_audio_volume", 0.45)
        )
        self.max_speed = min(1.2, max(1.0, float(self.audio_config.get("max_tts_speed", 1.2))))
        self.gemini = GeminiAgentClient(
            config=config,
            section="audio_agent",
            agent_id="audio_expert",
            role="Audio Localization and Dubbing Expert",
            system_instruction=AUDIO_SYSTEM_INSTRUCTION,
            schema=DUBBING_PLAN_SCHEMA,
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
        if not self.gemini.available:
            raise GeminiAgentError(
                f"Audio Expert requires Gemini: {self.gemini.configuration_error()}"
            )
        clips = input_data.get("clips", [])
        analysis = input_data.get("content_analysis", {})
        for clip in clips:
            clip_name = Path(clip.get("filename") or clip["path"]).stem
            source_cues = clip.get("source_subtitle_segments", [])
            if not source_cues:
                clip["dubbing_plans"] = {}
                continue
            plans: Dict[str, Dict[str, Any]] = {}
            for language in self.target_languages:
                subtitle_plan = clip.get("subtitle_plans", {}).get(language, {})
                payload = {
                    "request": "Create a complete spoken dubbing plan.",
                    "clip_id": clip_name,
                    "source_language": clip.get("detected_language", ""),
                    "target_language": language,
                    "content_context": {
                        "summary": analysis.get("summary", ""),
                        "topics": analysis.get("topics", [])[:8],
                        "sentiment": analysis.get("sentiment", {}),
                    },
                    "subtitle_context": subtitle_plan.get("cues", []),
                    "constraints": {
                        "timestamps_are_immutable": True,
                        "single_voice": True,
                        "complete_source_coverage": True,
                        "spoken_text_must_fit_each_window": True,
                    },
                    "source_cues": source_cues,
                }
                raw, trace_path = self.gemini.generate(
                    payload, clip_id=clip_name, language=language
                )
                plan, errors = self._validate_plan(raw, clip_name, language, source_cues)
                traces = [trace_path]
                if errors:
                    repaired, repair_trace = self.gemini.generate(
                        {
                            "request": "Repair the rejected dubbing plan.",
                            "validation_errors": errors,
                            "rejected_plan": raw,
                            "original_request": payload,
                        },
                        clip_id=clip_name,
                        language=language,
                        request_kind="repair",
                        repair_attempt=1,
                    )
                    traces.append(repair_trace)
                    plan, errors = self._validate_plan(
                        repaired, clip_name, language, source_cues
                    )
                if errors:
                    raise GeminiAgentError(
                        f"Dubbing plan for {clip_name}/{language} failed validation: "
                        + "; ".join(errors)
                    )
                plan.update({
                    "approved": False,
                    "trace_paths": traces,
                    "agent": "audio_expert",
                    "model": self.gemini.model,
                    "prompt_version": self.gemini.prompt_version,
                })
                plans[language] = plan
            clip["dubbing_plans"] = plans
        return self._result(input_data, clips)

    def render_plans(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        clips = input_data.get("clips", [])
        audio_dir = self.paths.get("audio_dir", "output/audio")
        self.ensure_dir(audio_dir)
        for clip in clips:
            clip_path = clip["path"]
            clip_name = Path(clip.get("filename") or clip_path).stem
            duration = float(get_video_info(clip_path).get("duration", 0))
            if duration <= 0:
                raise RuntimeError(f"Could not determine duration for {clip_name}")
            background_path = ""
            separation: Dict[str, Any]
            try:
                separation = separate_background(clip_path, audio_dir, self.config)
                background_path = separation["background_path"]
            except AudioSeparationError as exc:
                separation = {
                    "method": "speech-aware-ducking",
                    "warning": str(exc),
                    "fallback_used": True,
                }

            translations: Dict[str, Dict[str, Any]] = {}
            for language, plan in clip.get("dubbing_plans", {}).items():
                if plan.get("approved") is False and input_data.get(
                    "localization_review_applied", False
                ):
                    continue
                cue_audio: List[Dict[str, Any]] = []
                provider_counts = {"gemini": 0, "edge": 0}
                for cue in plan.get("cues", []):
                    rendered = self._render_cue(
                        cue, clip_name, language, plan, audio_dir
                    )
                    cue_audio.append(rendered)
                    provider_counts[rendered["provider"]] += 1
                aligned_path = os.path.join(
                    audio_dir, f"{clip_name}_voiceover_{language}_aligned.wav"
                )
                build_aligned_voiceover(cue_audio, aligned_path, duration)
                video_filename = f"{clip_name}_{language}_dubbed.mp4"
                video_path = os.path.join(audio_dir, video_filename)
                merge_voiceover_with_background(
                    video_path=clip_path,
                    voiceover_path=aligned_path,
                    output_path=video_path,
                    background_path=background_path,
                    cue_windows=plan.get("cues", []),
                    background_volume=self.background_volume,
                    voiceover_volume=self.voiceover_volume,
                    original_audio_volume=self.original_audio_volume,
                )
                translations[language] = {
                    "video_path": video_path,
                    "video_filename": video_filename,
                    "voiceover_path": aligned_path,
                    "voiceover_filename": Path(aligned_path).name,
                    "tts_provider": (
                        "mixed" if all(provider_counts.values())
                        else "edge" if provider_counts["edge"] else "gemini"
                    ),
                    "tts_fallback_used": provider_counts["edge"] > 0,
                    "tts_provider_counts": provider_counts,
                    "separation": separation,
                    "trace_paths": plan.get("trace_paths", []),
                    "approved": bool(plan.get("approved", True)),
                    "cues": cue_audio,
                    "agent": plan.get("agent"),
                    "model": plan.get("model"),
                    "prompt_version": plan.get("prompt_version"),
                }
            clip["audio_translations"] = translations
            clip["audio_separation"] = separation
        return self._result(input_data, clips)

    def _render_cue(
        self,
        cue: Dict[str, Any],
        clip_name: str,
        language: str,
        plan: Dict[str, Any],
        audio_dir: str,
    ) -> Dict[str, Any]:
        cue_id = cue["cue_id"]
        window = max(0.1, float(cue["end"]) - float(cue["start"]))
        text = cue["spoken_text"]
        path, provider = self._synthesize(
            text, cue, clip_name, language, audio_dir, plan, suffix="initial"
        )
        actual = get_audio_duration(path)
        shortened = False
        if actual > window + 0.1:
            text, repair_trace = self._shorten_cue(cue, clip_name, language, actual, window)
            plan.setdefault("trace_paths", []).append(repair_trace)
            cue["spoken_text"] = text
            path, provider = self._synthesize(
                text, cue, clip_name, language, audio_dir, plan,
                suffix="shortened",
            )
            actual = get_audio_duration(path)
            shortened = True
        speed = 1.0
        if actual > window + 0.1:
            speed = actual / window
            if speed > self.max_speed:
                raise RuntimeError(
                    f"Dubbing cue {clip_name}/{language}/{cue_id} is "
                    f"{actual:.2f}s for a {window:.2f}s window after shortening"
                )
            adjusted = os.path.join(
                audio_dir, f"{clip_name}_{language}_{cue_id}_speed.wav"
            )
            adjust_audio_speed(path, adjusted, speed)
            path = adjusted
            actual = get_audio_duration(path)
        return {
            "cue_id": cue_id,
            "start": cue["start"],
            "end": cue["end"],
            "path": path,
            "provider": provider,
            "fallback_used": provider == "edge",
            "duration": actual,
            "speed": round(speed, 3),
            "shortened": shortened,
            "spoken_text": text,
        }

    def _synthesize(
        self,
        text: str,
        cue: Dict[str, Any],
        clip_name: str,
        language: str,
        audio_dir: str,
        plan: Dict[str, Any],
        suffix: str,
    ) -> Tuple[str, str]:
        stem = f"{clip_name}_{language}_{cue['cue_id']}_{suffix}"
        wav_path = os.path.join(audio_dir, stem + ".wav")
        style = cue.get("performance_direction") or "Speak naturally"
        payload = {
            "cue_id": cue["cue_id"],
            "text": text,
            "style": style,
            "voice": self.tts_voice,
        }
        started = time.monotonic()
        try:
            gemini_text_to_speech(
                text,
                wav_path,
                style=style,
                voice=self.tts_voice,
                model=self.tts_model,
                client=self.gemini._get_client(),
            )
            trace = self.gemini.record_tool_call(
                tool="gemini_tts",
                clip_id=clip_name,
                language=language,
                payload=payload,
                status="complete",
                latency_seconds=time.monotonic() - started,
                model=self.tts_model,
                result={"output": Path(wav_path).name, "provider": "gemini"},
            )
            plan.setdefault("trace_paths", []).append(trace)
            return wav_path, "gemini"
        except Exception as gemini_error:
            failure_trace = self.gemini.record_tool_call(
                tool="gemini_tts",
                clip_id=clip_name,
                language=language,
                payload=payload,
                status="failed",
                latency_seconds=time.monotonic() - started,
                model=self.tts_model,
                error=str(gemini_error),
                fallback_used=self.edge_fallback,
            )
            plan.setdefault("trace_paths", []).append(failure_trace)
            if not self.edge_fallback:
                raise RuntimeError(f"Gemini TTS failed: {gemini_error}") from gemini_error
            edge_path = os.path.join(audio_dir, stem + "_edge.mp3")
            edge_started = time.monotonic()
            try:
                text_to_speech(
                    text,
                    edge_path,
                    lang=language,
                    engine="edge",
                    allow_gtts_fallback=False,
                )
                normalize_audio_to_wav(edge_path, wav_path)
            except Exception as edge_error:
                edge_trace = self.gemini.record_tool_call(
                    tool="edge_tts",
                    clip_id=clip_name,
                    language=language,
                    payload=payload,
                    status="failed",
                    latency_seconds=time.monotonic() - edge_started,
                    model="edge-tts",
                    error=str(edge_error),
                    fallback_used=True,
                )
                plan.setdefault("trace_paths", []).append(edge_trace)
                raise RuntimeError(
                    f"Gemini TTS and Edge TTS both failed: {edge_error}"
                ) from edge_error
            edge_trace = self.gemini.record_tool_call(
                tool="edge_tts",
                clip_id=clip_name,
                language=language,
                payload=payload,
                status="complete",
                latency_seconds=time.monotonic() - edge_started,
                model="edge-tts",
                result={"output": Path(wav_path).name, "provider": "edge"},
                fallback_used=True,
            )
            plan.setdefault("trace_paths", []).append(edge_trace)
            return wav_path, "edge"

    def _shorten_cue(
        self,
        cue: Dict[str, Any],
        clip_name: str,
        language: str,
        actual_duration: float,
        target_duration: float,
    ) -> Tuple[str, str]:
        raw, trace = self.gemini.generate(
            {
                "request": "Shorten this single dubbing cue without losing facts.",
                "clip_id": clip_name,
                "target_language": language,
                "voice": self.tts_voice,
                "overall_style": cue.get("performance_direction", "natural"),
                "actual_duration": actual_duration,
                "target_duration": target_duration,
                "cues": [cue],
            },
            clip_id=clip_name,
            language=language,
            request_kind="duration_repair",
            repair_attempt=1,
        )
        cues = raw.get("cues", [])
        if len(cues) != 1 or cues[0].get("cue_id") != cue["cue_id"]:
            raise GeminiAgentError("Audio duration repair returned the wrong cue")
        text = str(cues[0].get("spoken_text", "")).strip()
        if not text:
            raise GeminiAgentError("Audio duration repair returned empty text")
        return text, trace

    @staticmethod
    def _validate_plan(
        raw: Dict[str, Any],
        clip_id: str,
        language: str,
        source_cues: List[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], List[str]]:
        errors: List[str] = []
        if raw.get("clip_id") != clip_id:
            errors.append("clip_id does not match")
        if str(raw.get("target_language", "")).lower() != language.lower():
            errors.append("target_language does not match")
        returned = raw.get("cues")
        if not isinstance(returned, list):
            return raw, errors + ["cues must be a list"]
        source_by_id = {cue["cue_id"]: cue for cue in source_cues}
        ids = [cue.get("cue_id") for cue in returned if isinstance(cue, dict)]
        if ids != list(source_by_id):
            errors.append("cue IDs must appear exactly once and in source order")
        normalized = {
            "clip_id": clip_id,
            "target_language": language,
            "voice": str(raw.get("voice", "Kore")),
            "overall_style": str(raw.get("overall_style", "natural")),
            "cues": [],
        }
        previous_end = -1.0
        for cue in returned:
            if not isinstance(cue, dict) or cue.get("cue_id") not in source_by_id:
                continue
            source = source_by_id[cue["cue_id"]]
            if abs(float(cue.get("start", -1)) - float(source["start"])) > 0.25:
                errors.append(f"{cue['cue_id']} changed its start timestamp")
            if abs(float(cue.get("end", -1)) - float(source["end"])) > 0.25:
                errors.append(f"{cue['cue_id']} changed its end timestamp")
            spoken = str(cue.get("spoken_text", "")).strip()
            if not spoken:
                errors.append(f"{cue['cue_id']} has empty spoken_text")
            start = float(source["start"])
            end = float(source["end"])
            if start < previous_end - 0.001:
                errors.append(f"{cue['cue_id']} overlaps the previous cue")
            previous_end = end
            normalized["cues"].append({
                "cue_id": cue["cue_id"],
                "start": start,
                "end": end,
                "source_text": source.get("text", ""),
                "spoken_text": spoken,
                "performance_direction": str(
                    cue.get("performance_direction", "Speak naturally")
                ),
                "estimated_pace": str(cue.get("estimated_pace", "normal")),
                "warning": str(cue.get("warning", "")),
            })
        return normalized, list(dict.fromkeys(errors))

    @staticmethod
    def _result(input_data: Dict[str, Any], clips: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {"clips": clips, **{key: value for key, value in input_data.items() if key != "clips"}}
