import unittest
from unittest.mock import patch

from agents.quality_agent import QualityAgent


class QualityAgentHookScoreTests(unittest.TestCase):
    @patch.object(QualityAgent, "_check_clip")
    def test_report_contains_average_and_per_clip_hook_scores(self, check_clip):
        check_clip.return_value = {"issues": [], "info": {}}
        agent = QualityAgent({
            "quality_agent": {"generate_report": False},
        })
        clips = [
            {"filename": "one.mp4", "path": "one.mp4", "hook_score": 80},
            {"filename": "two.mp4", "path": "two.mp4", "hook_score": 60},
        ]

        result = agent.execute({"clips": clips})
        report = result["quality_report"]

        self.assertEqual(report["average_hook_score"], 70.0)
        self.assertEqual(report["summary"]["average_hook_score"], 70.0)
        self.assertEqual(report["clip_reports"][0]["hook_score"], 80)
        self.assertEqual(report["clip_reports"][1]["hook_score"], 60)


if __name__ == "__main__":
    unittest.main()
