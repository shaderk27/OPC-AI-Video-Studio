import unittest
from unittest.mock import patch

import main
from api.webapp import RUNS, _new_run, api_continue_after_review


class PostClipReviewGateTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "orchestrator": {"pipeline": ["clip", "subtitle"]},
            "clip_agent": {},
            "subtitle_agent": {},
        }
        self.generated_clips = [
            {"filename": "one.mp4", "path": "one.mp4", "id": 1},
            {"filename": "two.mp4", "path": "two.mp4", "id": 2},
        ]

    def _run(self, on_review, subtitle_run):
        clip_result = {
            "clips": list(self.generated_clips),
            "video_info": {"duration": 60, "has_audio": True},
        }
        with patch("main.os.path.exists", return_value=True), \
                patch("main.print_banner"), \
                patch("main.print_pipeline_diagram"), \
                patch("main.print_summary"), \
                patch("main.save_manifest"), \
                patch("agents.clip_agent.ClipAgent.run", return_value=clip_result), \
                patch("agents.subtitle_agent.SubtitleAgent.run", side_effect=subtitle_run) as subtitle_mock:
            result = main.run_studio(
                "input/match.mp4", self.config, on_review=on_review
            )
        return result, subtitle_mock

    def test_only_approved_clips_reach_downstream_agent(self):
        received = []

        def subtitle_run(input_data):
            received.extend(input_data["clips"])
            return {"clips": input_data["clips"]}

        result, subtitle_mock = self._run(
            lambda clips: [clips[1]], subtitle_run
        )

        subtitle_mock.assert_called_once()
        self.assertEqual([clip["id"] for clip in received], [2])
        self.assertEqual([clip["id"] for clip in result["clips"]], [2])
        self.assertEqual(result["review"], {
            "generated_count": 2,
            "approved_count": 1,
            "rejected_count": 1,
        })

    def test_zero_approved_clips_skip_all_downstream_agents(self):
        result, subtitle_mock = self._run(lambda clips: [], None)

        subtitle_mock.assert_not_called()
        self.assertEqual(result["clips"], [])
        self.assertEqual(result["review"]["approved_count"], 0)
        self.assertEqual(result["review"]["rejected_count"], 2)


class ReviewGateApiTests(unittest.TestCase):
    def tearDown(self):
        RUNS.pop("review-test", None)

    def test_review_submission_records_decisions_and_releases_worker(self):
        run = _new_run()
        run["status"] = "awaiting_review"
        run["review_clips"] = [{"id": 1}, {"id": 2}, {"id": 3}]
        run["decisions"] = {
            "1": {"decision": "pending"},
            "2": {"decision": "pending"},
            "3": {"decision": "pending"},
        }
        RUNS["review-test"] = run

        response = api_continue_after_review(
            "review-test", {"approved_indices": [1, 3]}
        )

        self.assertEqual(response["approved_count"], 2)
        self.assertEqual(response["rejected_count"], 1)
        self.assertEqual(run["approved_indices"], [1, 3])
        self.assertEqual(run["decisions"]["1"]["decision"], "approved")
        self.assertEqual(run["decisions"]["2"]["decision"], "dropped")
        self.assertEqual(run["decisions"]["3"]["decision"], "approved")
        self.assertTrue(run["review_event"].is_set())


if __name__ == "__main__":
    unittest.main()
