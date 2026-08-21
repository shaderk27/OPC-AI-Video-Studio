import json
import base64
import re
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
from utils.audio_separation import AudioSeparationError
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


class _Files:
    def __init__(self):
        self.uploaded = []
        self.deleted = []

    def upload(self, *, file):
        uploaded = SimpleNamespace(
            name=f"files/{Path(file).name}", state=SimpleNamespace(name="ACTIVE")
        )
        self.uploaded.append(str(file))
        return uploaded

    def get(self, *, name):
        return SimpleNamespace(name=name, state=SimpleNamespace(name="ACTIVE"))

    def delete(self, *, name):
        self.deleted.append(name)


class _Client:
    def __init__(self, responses=()):
        self.models = _Models(responses)
        self.files = _Files()


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

    def test_transient_gemini_failure_is_retried_and_traced(self):
        class FlakyModels:
            def __init__(self):
                self.calls = 0

            def generate_content(self, **_kwargs):
                self.calls += 1
                if self.calls == 1:
                    raise ConnectionError("temporary TLS failure")
                return _Response({"ok": True})

        with tempfile.TemporaryDirectory() as tmp, patch(
            "utils.gemini_agent_client.time.sleep"
        ):
            cfg = _config(tmp, "subtitle_agent")
            cfg["subtitle_agent"]["gemini"]["transport_retries"] = 1
            client = SimpleNamespace(models=FlakyModels())
            wrapper = GeminiAgentClient(
                cfg,
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
            trace = json.loads(Path(trace_path).read_text(encoding="utf-8"))
        self.assertEqual({"ok": True}, result)
        self.assertEqual(["failed", "complete"], [
            attempt["status"] for attempt in trace["transport_attempts"]
        ])


class SubtitleExpertTests(unittest.TestCase):
    def test_line_breaks_are_wrapped_locally_without_failing_valid_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = SubtitleAgent(_config(tmp, "subtitle_agent"), _Client())
            text = "Di Maria is the spare man and Argentina continue their attack."
            source = [{
                "cue_id": "cue_0001", "start": 0.0, "end": 4.0,
                "text": text,
            }]
            raw = {
                "clip_id": "clip", "source_language": "en",
                "target_language": "en", "summary": "football",
                "cues": [{
                    "cue_id": "cue_0001", "translated_text": text,
                    "line_break_text": text, "confidence": 1.0, "warning": "",
                }],
            }
            plan, errors = agent._validate_plan(raw, "clip", "en", "en", source)

            self.assertEqual([], errors)
            lines = plan["cues"][0]["line_break_text"].splitlines()
            self.assertEqual(2, len(lines))
            self.assertTrue(all(len(line) <= agent.max_chars for line in lines))

    def test_overlong_translation_is_split_without_losing_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = SubtitleAgent(_config(tmp, "subtitle_agent"), _Client())
            text = (
                "This season, 8 at-bats, 3 hits, and a grand slam. "
                "Oh, the ball got away! The ball got away!"
            )
            source = [{
                "cue_id": "cue_0001", "start": 1.0, "end": 7.0,
                "text": "Korean source commentary",
            }]
            raw = {
                "clip_id": "clip", "source_language": "ko",
                "target_language": "en", "summary": "baseball",
                "cues": [{
                    "cue_id": "cue_0001", "translated_text": text,
                    "line_break_text": text, "confidence": 0.95, "warning": "",
                }],
            }

            plan, errors = agent._validate_plan(raw, "clip", "ko", "en", source)

            self.assertEqual([], errors)
            self.assertGreater(len(plan["cues"]), 1)
            self.assertEqual("cue_0001", plan["cues"][0]["cue_id"])
            self.assertEqual(1.0, plan["cues"][0]["start"])
            self.assertEqual(7.0, plan["cues"][-1]["end"])
            self.assertEqual(text, " ".join(
                cue["translated_text"] for cue in plan["cues"]
            ))
            self.assertTrue(all(
                len(lines := cue["line_break_text"].splitlines()) <= 2
                and all(len(line) <= agent.max_chars for line in lines)
                for cue in plan["cues"]
            ))
            self.assertTrue(all(
                left["end"] == right["start"]
                for left, right in zip(plan["cues"], plan["cues"][1:])
            ))

    def test_canonical_transcript_expands_insufficient_whisper_cues(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = SubtitleAgent(_config(tmp, "subtitle_agent"), _Client())
            source = [{
                "cue_id": "cue_0001", "start": 0.0, "end": 14.0,
                "text": "Short Whisper guess.",
            }]
            transcript = " ".join(["football commentary"] * 14)
            expanded = agent._resegment_for_canonical_transcript(source, transcript)

            self.assertGreater(len(expanded), 1)
            self.assertEqual(0.0, expanded[0]["start"])
            self.assertEqual(14.0, expanded[-1]["end"])
            self.assertEqual(
                [f"cue_{index:04d}" for index in range(1, len(expanded) + 1)],
                [cue["cue_id"] for cue in expanded],
            )
            self.assertTrue(all(
                len(re.sub(r"\s+", "", cue["text"])) <= agent.max_chars * 2
                for cue in expanded
            ))

    def test_context_corrects_source_before_other_languages_translate(self):
        canonical = {
            "language": "en",
            "transcript": "Di Maria is the spare man.",
            "uncertain_phrases": [],
        }
        source_plan = {
            "clip_id": "clip", "source_language": "en",
            "target_language": "en", "summary": "Football attack",
            "cues": [{
                "cue_id": "cue_0001",
                "translated_text": "Di Maria is the spare man.",
                "line_break_text": "Di Maria is the spare man.",
                "confidence": 0.95, "warning": "",
            }],
        }
        chinese_plan = {
            "clip_id": "clip", "source_language": "en",
            "target_language": "zh", "summary": "足球进攻",
            "cues": [{
                "cue_id": "cue_0001", "translated_text": "迪马利亚无人盯防。",
                "line_break_text": "迪马利亚无人盯防。",
                "confidence": 0.95, "warning": "",
            }],
        }
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _config(tmp, "subtitle_agent")
            cfg["subtitle_agent"]["target_languages"] = ["zh"]
            client = _Client([canonical, chinese_plan])
            agent = SubtitleAgent(cfg, client)
            clip = {
                "path": str(Path(tmp) / "clip.mp4"), "filename": "clip.mp4",
                "start": 100.0, "end": 104.0, "duration": 4.0,
            }
            analysis = {
                "content_type": "sports commentary",
                "summary": "Argentina attacks through Di Maria.",
                "topics": [{"term": "Angel Di Maria"}],
                "entities": [{"name": "Angel Di Maria", "type": "athlete"}],
                "corrected_terms": [{"heard": "Dima Rier", "corrected": "Di Maria"}],
            }
            def write_audio(_source, output):
                Path(output).parent.mkdir(parents=True, exist_ok=True)
                Path(output).write_bytes(b"audio")
                return output

            with patch(
                "agents.subtitle_agent.extract_audio", side_effect=write_audio
            ), patch(
                "agents.subtitle_agent.transcribe_audio",
                return_value=[{
                    "start": 0.0, "end": 4.0,
                    "text": "Dima Rier is the passman.",
                }],
            ) as transcribe:
                agent.create_plans({
                    "clips": [clip], "content_analysis": analysis,
                    "full_transcript": [{
                        "start": 100.0, "end": 104.0,
                        "text": "Dima Rier is the spare man.",
                    }],
                })

            self.assertIn(
                "Di Maria", transcribe.call_args.kwargs["vocabulary_hints"]
            )
            source_contents = client.models.calls[0]["contents"]
            self.assertIsInstance(source_contents, list)
            self.assertEqual(
                "gemini-3.1-pro-preview", client.models.calls[0]["model"]
            )
            correction_request = json.loads(source_contents[0])
            self.assertIn(
                "Angel Di Maria", correction_request["known_names_and_terms"]
            )
            translation_request = json.loads(client.models.calls[1]["contents"])
            self.assertEqual(
                canonical["transcript"],
                translation_request["content_context"]["authoritative_audio_transcript"],
            )
            self.assertEqual(
                "Dima Rier is the spare man.",
                translation_request["content_context"]["nearby_transcript"][0]["text"],
            )
            self.assertEqual(
                "Di Maria is the spare man.",
                translation_request["cues"][0]["source_text"],
            )
            self.assertEqual(
                "Di Maria is the spare man.",
                translation_request["cues"][0]["raw_asr_text"],
            )
            self.assertEqual(1, len(client.files.uploaded))
            self.assertEqual(1, len(client.files.deleted))

    def test_low_vad_coverage_retranscribes_and_keeps_later_speech(self):
        response = {
            "clip_id": "clip",
            "source_language": "en",
            "target_language": "en",
            "summary": "Complete commentary",
            "cues": [
                {"cue_id": "cue_0001", "translated_text": "Opening call.",
                 "line_break_text": "Opening call.", "confidence": 1, "warning": ""},
                {"cue_id": "cue_0002", "translated_text": "Later",
                 "line_break_text": "Later", "confidence": 1, "warning": ""},
                {"cue_id": "cue_0003", "translated_text": "commentary.",
                 "line_break_text": "commentary.", "confidence": 1, "warning": ""},
            ],
        }
        primary = [{"start": 0.0, "end": 2.0, "text": "Opening call."}]
        recovered = [
            {"start": 0.0, "end": 2.0, "text": "Opening call."},
            {"start": 2.0, "end": 19.0, "text": "Later commentary."},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            agent = SubtitleAgent(
                _config(tmp, "subtitle_agent"), _Client([response])
            )
            clip = {
                "path": str(Path(tmp) / "clip.mp4"),
                "filename": "clip.mp4",
                "start": 41.0,
                "end": 61.0,
                "duration": 20.0,
            }
            with patch("agents.subtitle_agent.extract_audio"), patch(
                "agents.subtitle_agent.transcribe_audio",
                side_effect=[primary, recovered],
            ) as transcribe:
                result = agent.create_plans({
                    "clips": [clip],
                    "content_analysis": {},
                    "full_transcript": [{
                        "start": 42.0, "end": 61.0, "text": "Commentary"
                    }],
                })

        current = result["clips"][0]
        self.assertEqual(2, transcribe.call_count)
        self.assertFalse(transcribe.call_args_list[1].kwargs["vad_filter"])
        self.assertEqual(19.0, current["source_subtitle_segments"][-1]["end"])
        self.assertTrue(current["subtitle_transcription"]["recovery_used"])
        self.assertTrue(current["subtitle_plans"]["en"]["approved"])

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

    def test_adjacent_tiny_whisper_fragment_is_merged_before_agent_handoff(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = SubtitleAgent(_config(tmp, "subtitle_agent"), _Client())
            segments = [
                {"start": 0.0, "end": 1.0, "text": "What a goal!"},
                {"start": 1.1, "end": 3.0, "text": "The crowd erupts."},
            ]
            merged = agent._merge_short_segments(segments)
        self.assertEqual(1, len(merged))
        self.assertEqual(0.0, merged[0]["start"])
        self.assertEqual(3.0, merged[0]["end"])
        self.assertIn("crowd", merged[0]["text"])

    def test_each_target_cue_rejects_control_text_even_if_other_cues_match_script(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = SubtitleAgent(_config(tmp, "subtitle_agent"), _Client())
            source = [
                {"cue_id": "cue_0001", "start": 0.0, "end": 2.0, "text": "Goal"},
                {"cue_id": "cue_0002", "start": 2.0, "end": 4.0, "text": "Amazing"},
            ]
            raw = {
                "clip_id": "clip", "source_language": "en",
                "target_language": "ko", "summary": "goal",
                "cues": [
                    {"cue_id": "cue_0001", "translated_text": "\x80\x80",
                     "line_break_text": "\x80\x80", "confidence": 0.8, "warning": ""},
                    {"cue_id": "cue_0002", "translated_text": "놀랍습니다",
                     "line_break_text": "놀랍습니다", "confidence": 0.9, "warning": ""},
                ],
            }
            _plan, errors = agent._validate_plan(raw, "clip", "en", "ko", source)
        self.assertTrue(any("control characters" in error for error in errors))
        self.assertTrue(any("cue_0001" in error and "script" in error for error in errors))


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

    def test_exact_maximum_speed_ratio_does_not_fail_from_float_rounding(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = AudioAgent(_config(tmp, "audio_agent"), _Client())
            cue = {
                "cue_id": "cue_0002", "start": 0.0, "end": 2.0,
                "spoken_text": "実況です", "performance_direction": "excited",
            }
            plan = {"trace_paths": []}
            with patch.object(
                agent, "_synthesize",
                side_effect=[("initial.wav", "gemini"), ("short.wav", "gemini")],
            ), patch.object(
                agent, "_shorten_cue", return_value=("短い実況", "repair.json")
            ), patch(
                "agents.audio_agent.get_audio_duration",
                side_effect=[2.5, 2.4000001, 2.0],
            ), patch("agents.audio_agent.adjust_audio_speed"):
                rendered = agent._render_cue(cue, "match", "ja", plan, tmp)

            self.assertEqual(1.2, rendered["speed"])
            self.assertTrue(rendered["shortened"])

    def test_one_language_render_failure_does_not_discard_other_languages(self):
        with tempfile.TemporaryDirectory() as tmp:
            clip_path = Path(tmp) / "match.mp4"
            clip_path.write_bytes(b"fixture")
            agent = AudioAgent(_config(tmp, "audio_agent"), _Client())
            cue = {"cue_id": "cue_0001", "start": 0.0, "end": 2.0}
            clip = {
                "path": str(clip_path), "filename": "match.mp4",
                "dubbing_plans": {
                    "ja": {"approved": True, "cues": [dict(cue)]},
                    "ko": {"approved": True, "cues": [dict(cue)]},
                },
            }

            def render(_cue, _clip, language, _plan, _audio_dir):
                if language == "ja":
                    raise RuntimeError("cue cannot fit")
                return {
                    "path": str(Path(tmp) / "ko.wav"), "provider": "gemini",
                    "start": 0.0, "end": 2.0,
                }

            with patch(
                "agents.audio_agent.get_video_info", return_value={"duration": 5.0}
            ), patch(
                "agents.audio_agent.separate_background",
                side_effect=AudioSeparationError("separator unavailable"),
            ), patch.object(agent, "_render_cue", side_effect=render), patch(
                "agents.audio_agent.build_aligned_voiceover"
            ), patch("agents.audio_agent.merge_voiceover_with_background"):
                result = agent.render_plans({"clips": [clip]})

            rendered_clip = result["clips"][0]
            self.assertIn("ko", rendered_clip["audio_translations"])
            self.assertIn("ja", rendered_clip["audio_failures"])
            self.assertIn("cue cannot fit", rendered_clip["audio_failures"]["ja"]["error"])

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
            self.assertEqual("ko-KR-SunHiNeural", traces[-1]["result"]["voice"])


class LocalizationReviewTests(unittest.TestCase):
    def test_review_only_edits_audio_and_subtitles_remain_auto_approved(self):
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
        self.assertIsNone(item["subtitle_plan"])
        item["dubbing_plan"]["cues"][0].update({"spoken_text": "new speech", "end": 99})

        apply_localization_review(clips, review)
        self.assertEqual(0.0, clips[0]["subtitle_plans"]["ko"]["cues"][0]["start"])
        self.assertEqual("old", clips[0]["subtitle_plans"]["ko"]["cues"][0]["translated_text"])
        self.assertTrue(clips[0]["subtitle_plans"]["ko"]["approved"])
        self.assertEqual("automatic", clips[0]["subtitle_plans"]["ko"]["approval_mode"])
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
    def test_main_auto_renders_subtitles_then_reviews_audio(self):
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
                "approved": True,
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

        def subtitle_render(input_data):
            order.append("subtitle_render")
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
                on_localization_review=lambda _payload: self.fail(
                    "automatic dubbing must not request human approval"
                ),
            )

        self.assertEqual(
            ["subtitle_plan", "subtitle_render", "audio_plan", "audio_render"],
            order,
        )
        self.assertIn("ko", result["clips"][0]["subtitles"])
        self.assertIn("ko", result["clips"][0]["audio_translations"])


if __name__ == "__main__":
    unittest.main()
