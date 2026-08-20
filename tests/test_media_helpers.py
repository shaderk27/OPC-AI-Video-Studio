import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agents.format_agent import FormatAgent, parse_aspect_ratio
from utils.audio_utils import get_edge_voice
from utils.video_utils import burn_subtitles_into_video


class EdgeVoiceMappingTests(unittest.TestCase):
    def test_resolves_iso_and_bcp47_language_codes(self):
        self.assertEqual("ko-KR-SunHiNeural", get_edge_voice("ko"))
        self.assertEqual("ko-KR-SunHiNeural", get_edge_voice("ko-KR"))
        self.assertEqual("zh-CN-XiaoxiaoNeural", get_edge_voice("zh_CN"))

    def test_rejects_unsupported_language(self):
        with self.assertRaisesRegex(ValueError, "No Edge TTS voice"):
            get_edge_voice("xx")


class WindowsSubtitlePathTests(unittest.TestCase):
    def test_burn_uses_unique_cwd_relative_subtitle_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subtitle = root / "clip subtitle's ko.srt"
            subtitle.write_text(
                "1\n00:00:00,000 --> 00:00:01,000\nGoal!\n",
                encoding="utf-8",
            )
            output = root / "burned output.mp4"
            completed = SimpleNamespace(returncode=0, stderr="")
            with patch(
                "utils.video_utils.subprocess.run", return_value=completed
            ) as run:
                result = burn_subtitles_into_video(
                    str(root / "source video.mp4"), str(subtitle), str(output)
                )

        command = run.call_args.args[0]
        self.assertEqual("subtitles='subtitle.srt'", command[command.index("-vf") + 1])
        self.assertNotIn(":", command[command.index("-vf") + 1])
        self.assertTrue(run.call_args.kwargs["cwd"])
        self.assertEqual(str(output.resolve()), result)


class AspectRatioParsingTests(unittest.TestCase):
    def test_accepts_supported_ratio_forms(self):
        self.assertEqual((9, 16), parse_aspect_ratio("9:16"))
        self.assertEqual((9, 16), parse_aspect_ratio("9x16"))
        self.assertEqual((16, 9), parse_aspect_ratio("16/9"))
        self.assertEqual((1, 1), parse_aspect_ratio([1, 1]))

    def test_rejects_ambiguous_or_invalid_ratios(self):
        for value in (916, "916", "0:16", "hello"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_aspect_ratio(value)

    def test_format_groups_normalize_ratio_text(self):
        agent = FormatAgent({"format_agent": {"formats": [{
            "platform": "TikTok", "width": 1080, "height": 1920,
            "aspect_ratio": "9x16",
        }]}})
        groups = agent._group_formats()
        self.assertEqual("9:16", groups[(1080, 1920)][0]["aspect_ratio"])


if __name__ == "__main__":
    unittest.main()
