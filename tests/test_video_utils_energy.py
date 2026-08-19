import subprocess
import unittest
from unittest.mock import patch

from utils.video_utils import get_audio_energy_segments


class AudioEnergyExtractionTests(unittest.TestCase):
    @patch("utils.video_utils.get_video_info")
    @patch("utils.video_utils.subprocess.run")
    def test_reads_ffmpeg_metadata_from_stdout(self, mock_run, mock_video_info):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=(
                "frame:0 pts_time:0.5\n"
                "lavfi.astats.Overall.RMS_level=-10.0\n"
                "frame:1 pts_time:1.5\n"
                "lavfi.astats.Overall.RMS_level=-20.0\n"
                "frame:2 pts_time:2.5\n"
                "lavfi.astats.Overall.RMS_level=-5.0\n"
            ),
            stderr="ffmpeg diagnostics only",
        )
        mock_video_info.return_value = {"duration": 4.0}

        segments = get_audio_energy_segments("match.mp4", segment_duration=2.0)

        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0], {
            "start": 2.0,
            "end": 4.0,
            "energy": -5.0,
        })
        self.assertEqual(segments[1], {
            "start": 0.0,
            "end": 2.0,
            "energy": -15.0,
        })


if __name__ == "__main__":
    unittest.main()
