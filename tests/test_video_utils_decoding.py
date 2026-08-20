import subprocess
import unittest
from unittest.mock import patch

from utils.video_utils import _parse_ffmpeg_info, _run_media_command


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


if __name__ == "__main__":
    unittest.main()
