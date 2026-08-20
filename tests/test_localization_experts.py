import json
import base64
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException
import main

from agents.audio_agent import AudioAgent, AUDIO_SYSTEM_INSTRUCTION
from agents.subtitle_agent import SubtitleAgent, SUBTITLE_SYSTEM_INSTRUCTION
from api import webapp
from utils.gemini_agent_client import GeminiAgentClient
from utils.audio_utils import build_aligned_voiceover, gemini_text_to_speech
from utils.localization_review import apply_localization_review, build_localization_review


class _Response:
    def __init__(self, parsed):
        self.parsed = parsed
        self.text = json.dumps(parsed)
        self.usage_metadata = SimpleNamespace(
            prompt_token_count=10,
            candidates_token_count=5,
            total_token_count=15,
        )


class _Models:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return _Response(self.responses.pop(0))


class _Client:
    def __init__(self, responses=()):
        self.models = _Models(responses)


def _config(root, section):
    return {
        "paths": {
            "output_dir": str(root),
            "agent_traces_dir": str(Path(root) / "agent_traces"),
            "subtitles_dir": str(Path(root) / "subtitles"),
            "audio_dir": str(Path(root) / "audio"),
        },
        section: {
            "target_languages": ["en"],
            "formats": ["srt", "vtt"],
            "gemini": {
                "enabled": True,
                "required": True,
                "model": "gemini-3.5-flash-lite",
                "prompt_version": f"{section}-test",
            },
        },
    }


class GeminiFoundationTests(unittest.TestCase):
    def test_structured_request_writes_a_complete_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = _Client([{"ok": True}])
            wrapper = GeminiAgentClient(
                _config(tmp, "subtitle_agent"),
                "subtitle_agent",
                "subtitle_expert",
                "Subtitle Localization Expert",
                SUBTITLE_SYSTEM_INSTRUCTION,
                {"type": "object"},
                client,
            )
            result, trace_path = wrapper.generate(
                {"cue": "hello"}, clip_id="clip-1", language="ko"
            )

            self.assertEqual({"ok": True}, result)
            call = client.models.calls[0]
            self.assertEqual("application/json", call["config"]["response_mime_type"])
            self.assertEqual(SUBTITLE_SYSTEM_INSTRUCTION, call["config"]["system_instruction"])
            trace = json.loads(Path(trace_path).read_text(encoding="utf-8"))
            self.assertEqual("subtitle_expert", trace["agent_id"])
            self.assertEqual("complete", trace["status"])
            self.assertEqual(15, trace["usage"]["total_token_count"])

    def test_gemini_tts_uses_interactions_audio_contract(self):
        class Interactions:
            def __init__(self):
                self.call = None

            def create(self, **kwargs):
                self.call = kwargs
                pcm = b"\x00\x00" * 240
                return SimpleNamespace(
                    output_audio=SimpleNamespace(data=base64.b64encode(pcm).decode())
                )

        with tempfile.TemporaryDirectory() as tmp:
            client = SimpleNamespace(interactions=Interactions())
            output = Path(tmp) / "speech.wav"
            gemini_text_to_speech(
                "Goal!",
                str(output),
                style="Excited commentator",
                client=client,
            )
            with wave.open(str(output), "rb") as stream:
                self.assertEqual(24000, stream.getframerate())
                self.assertEqual(1, stream.getnchannels())
            self.assertEqual(
                {"type": "audio"}, client.interactions.call["response_format"]
            )


class SubtitleExpertTests(unittest.TestCase):
    def test_validation_keeps_whisper_timestamps_authoritative(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = SubtitleAgent(_config(tmp, "subtitle_agent"), _Client())
            source = [{"cue_id": "cue_0001", "start": 1.25, "end": 3.5, "text": "Goal!"}]
            raw = {
                "clip_id": "clip",
                "source_language": "en",
                "target_language": "ko",
                "summary": "goal",
                "cues": [{
                    "cue_id": "cue_0001",
                    "translated_text": "골입니다!",
                    "line_break_text": "골입니다!",
                    "confidence": 0.9,
                    "warning": "",
                    "start": 99,
                    "end": 100,
                }],
            }
            plan, errors = agent._validate_plan(raw, "clip", "en", "ko", source)
            self.assertEqual([], errors)
            self.assertEqual(1.25, plan["cues"][0]["start"])
            self.assertEqual(3.5, plan["cues"][0]["end"])

    def test_invalid_cue_coverage_triggers_one_repair_call(self):
        invalid = {
            "clip_id": "clip",
            "source_language": "en",
            "target_language": "en",
            "summary": "",
            "cues": [],
        }
        valid = {
            **invalid,
            "cues": [{
                "cue_id": "cue_0001",
                "translated_text": "Goal!",
                "line_break_text": "Goal!",
                "confidence": 1.0,
                "warning": "",
            }],
        }
        with tempfile.TemporaryDirectory() as tmp:
            client = _Client([invalid, valid])
            agent = SubtitleAgent(_config(tmp, "subtitle_agent"), client)
            clip = {"path": str(Path(tmp) / "clip.mp4"), "filename": "clip.mp4"}
            with patch("agents.subtitle_agent.extract_audio"), patch(
                "agents.subtitle_agent.transcribe_audio",
                return_value=[{"start": 0.0, "end": 2.0, "text": "Goal!"}],
            ):
                result = agent.create_plans({"clips": [clip], "content_analysis": {}})

            self.assertEqual(2, len(client.models.calls))
            self.assertEqual(
                "cue_0001",
                result["clips"][0]["subtitle_plans"]["en"]["cues"][0]["cue_id"],
            )
            self.assertIn("validation_errors", client.models.calls[1]["contents"])


class AudioExpertTests(unittest.TestCase):
    def test_audio_prompt_is_isolated_from_subtitle_role(self):
        self.assertIn("spoken", AUDIO_SYSTEM_INSTRUCTION.lower())
        self.assertNotEqual(AUDIO_SYSTEM_INSTRUCTION, SUBTITLE_SYSTEM_INSTRUCTION)

    def test_validation_rejects_timestamp_changes(self):
        raw = {
            "clip_id": "clip",
            "target_language": "ko",
            "voice": "Kore",
            "overall_style": "energetic",
            "cues": [{
                "cue_id": "cue_0001",
                "start": 1.0,
                "end": 5.0,
                "spoken_text": "골입니다!",
                "performance_direction": "excited",
                "estimated_pace": "fast",
                "warning": "",
            }],
        }
        source = [{"cue_id": "cue_0001", "start": 0.0, "end": 2.0, "text": "Goal!"}]
        plan, errors = AudioAgent._validate_plan(raw, "clip", "ko", source)
        self.assertTrue(any("timestamp" in error for error in errors))
        self.assertEqual(0.0, plan["cues"][0]["start"])
        self.assertEqual(2.0, plan["cues"][0]["end"])

    def test_aligned_voiceover_pads_trailing_silence_to_clip_duration(self):
        completed = SimpleNamespace(returncode=0, stderr="")
        with tempfile.TemporaryDirectory() as tmp, patch(
            "utils.audio_utils.subprocess.run", return_value=completed
        ) as run:
            build_aligned_voiceover(
                [{"path": str(Path(tmp) / "cue.wav"), "start": 0.5}],
                str(Path(tmp) / "aligned.wav"),
                2.0,
            )
        command = run.call_args.args[0]
        filters = command[command.index("-filter_complex") + 1]
        self.assertIn("apad=whole_dur=2.000", filters)

    def test_gemini_tts_failure_uses_explicit_edge_fallback_and_traces_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _config(tmp, "audio_agent")
            cfg["audio_agent"]["tts"] = {
                "model": "gemini-3.1-flash-tts-preview",
                "voice": "Kore",
                "edge_fallback": True,
            }
            agent = AudioAgent(cfg, _Client())
            plan = {"trace_paths": []}

            def write_edge(_text, path, **kwargs):
                self.assertFalse(kwargs["allow_gtts_fallback"])
                Path(path).write_bytes(b"edge")

            def normalize(_source, destination):
                with wave.open(destination, "wb") as stream:
                    stream.setnchannels(1)
                    stream.setsampwidth(2)
                    stream.setframerate(24000)
                    stream.writeframes(b"\x00\x00" * 100)

            cue = {"cue_id": "cue_0001", "performance_direction": "excited"}
            with patch("agents.audio_agent.gemini_text_to_speech", side_effect=RuntimeError("quota")), patch(
                "agents.audio_agent.text_to_speech", side_effect=write_edge
            ), patch("agents.audio_agent.normalize_audio_to_wav", side_effect=normalize):
                path, provider = agent._synthesize(
                    "골입니다!", cue, "clip", "ko", tmp, plan, "initial"
                )

            self.assertEqual("edge", provider)
            self.assertTrue(Path(path).is_file())
            self.assertEqual(2, len(plan["trace_paths"]))
            traces = [json.loads(Path(path).read_text(encoding="utf-8")) for path in plan["trace_paths"]]
            self.assertEqual(["failed", "complete"], [trace["status"] for trace in traces])
            self.assertTrue(all(trace["fallback_used"] for trace in traces))


class LocalizationReviewTests(unittest.TestCase):
    def test_review_edits_text_but_cannot_change_timing(self):
        clips = [{
            "filename": "clip.mp4",
            "subtitle_plans": {"ko": {"cues": [{
                "cue_id": "cue_0001", "start": 0.0, "end": 2.0,
                "translated_text": "old", "line_break_text": "old",
            }]}},
            "dubbing_plans": {"ko": {"cues": [{
                "cue_id": "cue_0001", "start": 0.0, "end": 2.0,
                "spoken_text": "old speech",
            }]}},
        }]
        review = build_localization_review(clips)
        item = review["items"][0]
        item["approved"] = True
        item["subtitle_plan"]["cues"][0].update(
            {"line_break_text": "new", "translated_text": "new", "start": 90}
        )
        item["dubbing_plan"]["cues"][0].update({"spoken_text": "new speech", "end": 99})

        apply_localization_review(clips, review)
        self.assertEqual(0.0, clips[0]["subtitle_plans"]["ko"]["cues"][0]["start"])
        self.assertEqual(2.0, clips[0]["dubbing_plans"]["ko"]["cues"][0]["end"])
        self.assertEqual("new speech", clips[0]["dubbing_plans"]["ko"]["cues"][0]["spoken_text"])

    def test_api_persists_decision_before_releasing_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_id = "localization-test"
            original = {
                "items": [{
                    "clip_index": 1,
                    "language": "ko",
                    "approved": False,
                    "subtitle_plan": None,
                    "dubbing_plan": None,
                }],
                "count": 1,
            }
            run = webapp._new_run()
            run.update({
                "id": run_id,
                "status": "awaiting_localization_review",
                "output_dir": tmp,
                "localization_review": original,
            })
            webapp.RUNS[run_id] = run
            try:
                response = webapp.api_continue_after_localization_review(
                    run_id, {"approve_all": True}
                )
                decision_path = Path(tmp) / "localization_review_decisions.json"
                self.assertTrue(decision_path.is_file())
                self.assertTrue(run["localization_event"].is_set())
                self.assertEqual(1, response["approved_count"])
                self.assertTrue(json.loads(decision_path.read_text(encoding="utf-8"))["items"][0]["approved"])
            finally:
                webapp.RUNS.pop(run_id, None)

    def test_legacy_languages_alias_populates_both_experts(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _config(tmp, "subtitle_agent")
            cfg["audio_agent"] = _config(tmp, "audio_agent")["audio_agent"]
            cfg["clip_agent"] = {"gemini_review": {"required": False}}
            cfg["content_analyzer"] = {"gemini_analysis": {}}
            cfg["format_agent"] = {"formats": []}
            cfg["quality_agent"] = {}
            cfg["manager_agent"] = {}
            request = webapp.WebRunRequest(video="input/example.mp4", languages=["ko"])
            webapp._apply_overrides(cfg, request, {})
            self.assertEqual(["ko"], cfg["subtitle_agent"]["target_languages"])
            self.assertEqual(["ko"], cfg["audio_agent"]["target_languages"])

    def test_api_rejects_duplicate_clip_language_items(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_id = "localization-duplicates"
            item = {"clip_index": 1, "language": "ko", "approved": True}
            run = webapp._new_run()
            run.update({
                "id": run_id,
                "status": "awaiting_localization_review",
                "output_dir": tmp,
                "localization_review": {"items": [item], "count": 1},
            })
            webapp.RUNS[run_id] = run
            try:
                with self.assertRaises(HTTPException):
                    webapp.api_continue_after_localization_review(
                        run_id, {"items": [item, item]}
                    )
            finally:
                webapp.RUNS.pop(run_id, None)


class CombinedLocalizationOrchestrationTests(unittest.TestCase):
    def test_main_hands_subtitle_plan_to_audio_then_reviews_before_render(self):
        order = []
        clip = {"filename": "clip.mp4", "path": "clip.mp4", "duration": 4.0}
        clip_result = {
            "clips": [clip],
            "video_info": {"duration": 4.0, "has_audio": True},
        }

        def subtitle_plan(input_data):
            order.append("subtitle_plan")
            current = input_data["clips"][0]
            current["source_subtitle_segments"] = [
                {"cue_id": "cue_0001", "start": 0.0, "end": 2.0, "text": "Goal!"}
            ]
            current["subtitle_plans"] = {"ko": {
                "approved": False,
                "cues": [{
                    "cue_id": "cue_0001", "start": 0.0, "end": 2.0,
                    "source_text": "Goal!", "translated_text": "골!",
                    "line_break_text": "골!", "warning": "",
                }],
            }}
            return {"clips": input_data["clips"]}

        def audio_plan(input_data):
            order.append("audio_plan")
            self.assertIn("subtitle_plans", input_data["clips"][0])
            input_data["clips"][0]["dubbing_plans"] = {"ko": {
                "approved": False,
                "cues": [{
                    "cue_id": "cue_0001", "start": 0.0, "end": 2.0,
                    "source_text": "Goal!", "spoken_text": "골입니다!",
                    "warning": "",
                }],
            }}
            return {"clips": input_data["clips"]}

        def review(payload):
            order.append("review")
            self.assertIsNotNone(payload["items"][0]["subtitle_plan"])
            self.assertIsNotNone(payload["items"][0]["dubbing_plan"])
            payload["items"][0]["approved"] = True
            return payload

        def subtitle_render(input_data):
            order.append("subtitle_render")
            self.assertTrue(input_data["localization_review_applied"])
            self.assertTrue(input_data["clips"][0]["subtitle_plans"]["ko"]["approved"])
            input_data["clips"][0]["subtitles"] = {"ko": {"formats": {}}}
            return {"clips": input_data["clips"]}

        def audio_render(input_data):
            order.append("audio_render")
            self.assertTrue(input_data["clips"][0]["dubbing_plans"]["ko"]["approved"])
            input_data["clips"][0]["audio_translations"] = {"ko": {}}
            return {"clips": input_data["clips"]}

        config = {
            "orchestrator": {"pipeline": ["clip", "subtitle", "audio"]},
            "clip_agent": {},
            "subtitle_agent": {"target_languages": ["ko"]},
            "audio_agent": {"target_languages": ["ko"]},
        }
        with patch("main.os.path.exists", return_value=True), patch(
            "main.print_banner"
        ), patch("main.print_pipeline_diagram"), patch("main.print_summary"), patch(
            "main.save_manifest"
        ), patch("agents.clip_agent.ClipAgent.run", return_value=clip_result), patch(
            "agents.subtitle_agent.SubtitleAgent.run", side_effect=subtitle_plan
        ), patch(
            "agents.audio_agent.AudioAgent.run", side_effect=audio_plan
        ), patch(
            "agents.subtitle_agent.SubtitleAgent.render_plans", side_effect=subtitle_render
        ), patch(
            "agents.audio_agent.AudioAgent.render_plans", side_effect=audio_render
        ):
            result = main.run_studio(
                "input/match.mp4",
                config,
                on_review=lambda clips: clips,
                on_localization_review=review,
            )

        self.assertEqual(
            ["subtitle_plan", "audio_plan", "review", "subtitle_render", "audio_render"],
            order,
        )
        self.assertIn("ko", result["clips"][0]["subtitles"])
        self.assertIn("ko", result["clips"][0]["audio_translations"])


if __name__ == "__main__":
    unittest.main()
