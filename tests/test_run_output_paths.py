import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from utils.run_paths import RUN_SUBDIRECTORIES, prepare_run_output, safe_path_part


class RunOutputPathTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_root = Path(self.temp_dir.name) / "output"
        self.config = {
            "paths": {
                "output_dir": str(self.output_root),
                "clips_dir": "old/clips",
                "social_dir": "old/social",
            }
        }
        self.started_at = datetime(
            2026, 8, 19, 14, 32, 8,
            tzinfo=timezone(timedelta(hours=8)),
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_creates_dated_run_folder_and_all_agent_directories(self):
        prepared, run_dir = prepare_run_output(
            self.config,
            "input/Shenzhen Travel Vlog.mp4",
            "a3f1b2c4",
            self.started_at,
        )

        expected = (
            self.output_root
            / "2026-08-19"
            / "14-32-08_shenzhen-travel-vlog_a3f1b2c4"
        )
        self.assertEqual(run_dir, expected)
        self.assertTrue(run_dir.is_dir())
        self.assertEqual(prepared["paths"]["output_dir"], str(expected))
        self.assertEqual(prepared["paths"]["output_root"], str(self.output_root))
        for key, folder_name in RUN_SUBDIRECTORIES.items():
            self.assertEqual(prepared["paths"][key], str(expected / folder_name))
            self.assertTrue((expected / folder_name).is_dir())

        self.assertEqual(self.config["paths"]["clips_dir"], "old/clips")
        self.assertEqual(self.config["paths"]["output_dir"], str(self.output_root))

    def test_existing_run_folder_is_never_reused(self):
        prepare_run_output(
            self.config, "input/video.mp4", "same-run", self.started_at
        )

        with self.assertRaises(FileExistsError):
            prepare_run_output(
                self.config, "input/video.mp4", "same-run", self.started_at
            )

    def test_different_run_ids_keep_reprocessing_separate(self):
        _, first = prepare_run_output(
            self.config, "input/video.mp4", "first001", self.started_at
        )
        _, second = prepare_run_output(
            self.config, "input/video.mp4", "second02", self.started_at
        )

        self.assertNotEqual(first, second)
        self.assertTrue(first.is_dir())
        self.assertTrue(second.is_dir())

    def test_safe_path_part_preserves_readable_chinese_and_removes_separators(self):
        self.assertEqual(safe_path_part("深圳 / 旅行：日落"), "深圳-旅行-日落")


if __name__ == "__main__":
    unittest.main()
