import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from utils.video_utils import (
    _parse_ffmpeg_info,
    _run_media_command,
    create_review_proxy,
)


class MediaCommandDecodingTests(unittest.TestCase):
    @patch("utils.video_utils.subprocess.run")
    def test_media_commands_decode_utf8_with_replacement(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess([], 0, "", "")

        _run_media_command(["ffmpeg", "-version"])

        mock_run.assert_called_once_with(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

    def test_ffmpeg_parser_tolerates_missing_stderr(self):
        self.assertEqual(
            _parse_ffmpeg_info(None),
            {
                "duration": 0.0,
                "width": 0,
                "height": 0,
                "fps": 0,
                "codec": "none",
                "has_audio": False,
                "audio_codec": "none",
            },
        )

    @patch("utils.video_utils._run_media_command")
    def test_review_proxy_forces_even_dimensions(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess([], 0, "", "")
        with tempfile.TemporaryDirectory() as temp_dir:
            create_review_proxy(
                "anamorphic.mp4",
                10,
                20,
                str(Path(temp_dir) / "proxy.mp4"),
            )

        command = mock_run.call_args.args[0]
        video_filter = command[command.index("-vf") + 1]
        self.assertIn("force_divisible_by=2", video_filter)
        self.assertIn("setsar=1", video_filter)

    @patch("utils.video_utils._run_media_command")
    def test_review_proxy_reports_ffmpeg_error_tail(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(
            [], 1, "", "version banner\n" + "x" * 2000 + "actual encoder error"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(RuntimeError, "actual encoder error"):
                create_review_proxy(
                    "source.mp4",
                    10,
                    20,
                    str(Path(temp_dir) / "proxy.mp4"),
                )


if __name__ == "__main__":
    unittest.main()
