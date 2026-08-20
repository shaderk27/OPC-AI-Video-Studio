import tempfile
import unittest
from unittest.mock import patch

from agents.clip_agent import ClipAgent
from utils.gemini_clip_reviewer import GeminiRequiredError


class _VideoReviewer:
    enabled = True
    available = True
    unavailable_reason = ""
    model = "gemini-3.5-flash-lite"
    video_review_enabled = True

    def __init__(self, reviews=None, error=None):
        self.reviews = reviews or {}
        self.error = error

    def review_candidate_videos(self, candidates):
        if self.error:
            raise self.error
        return self.reviews


def _review(accepted=True, score=80, start_offset=2, end_offset=22):
    return {
        "accepted": accepted,
        "event_type": "goal",
        "visual_editorial_score": score,
        "score_breakdown": {
            "hook": score,
            "flow": score,
            "payoff": score,
            "visual_interest": score,
            "standalone": score,
        },
        "recommended_start_offset": start_offset,
        "recommended_end_offset": end_offset,
        "reason": "Complete action and reaction.",
    }


class ClipAgentGeminiVideoReviewTests(unittest.TestCase):
    def _agent(self, output_dir, required=False, max_clips=2):
        return ClipAgent({
            "paths": {"output_dir": output_dir},
            "clip_agent": {
                "min_clip_duration": 15,
                "max_clip_duration": 60,
                "max_clips": max_clips,
                "min_clips": min(3, max_clips),
                "gemini_review": {
                    "enabled": True,
                    "required": required,
                    "video_review_enabled": True,
                    "video_candidate_limit": max(4, max_clips * 2),
                    "video_context_seconds": 4,
                    "video_min_score": 60,
                    "video_backfill_min_score": 45,
                    "video_blend_weight": 0.8,
                },
            },
        })

    @patch("agents.clip_agent.create_review_proxy")
    def test_backfills_strong_candidate_and_applies_boundary_correction(self, proxy):
        with tempfile.TemporaryDirectory() as output_dir:
            agent = self._agent(output_dir)
            agent.gemini_reviewer = _VideoReviewer({
                0: _review(accepted=True, score=80),
                1: _review(accepted=False, score=75),
            })
            candidates = [
                {"start": 10, "end": 30, "score": 50, "title": "Goal"},
                {"start": 50, "end": 70, "score": 70, "title": "Crowd noise"},
            ]

            result = agent._apply_gemini_video_review(
                "source.mp4", candidates, total_duration=100
            )

        self.assertEqual(len(result), 2)
        self.assertEqual((result[0]["start"], result[0]["end"]), (8.0, 28.0))
        self.assertEqual(result[0]["score"], 74.0)
        self.assertTrue(result[0]["gemini_video_accepted"])
        self.assertEqual(result[0]["pre_gemini_video_start"], 10)
        self.assertFalse(candidates[1]["gemini_video_accepted"])
        self.assertTrue(candidates[1]["gemini_video_backfilled"])
        self.assertEqual(proxy.call_count, 2)

    @patch("agents.clip_agent.create_review_proxy")
    def test_required_mode_fails_when_all_videos_are_rejected(self, proxy):
        with tempfile.TemporaryDirectory() as output_dir:
            agent = self._agent(output_dir, required=True)
            agent.gemini_reviewer = _VideoReviewer({
                0: _review(accepted=False, score=30)
            })

            with self.assertRaisesRegex(GeminiRequiredError, "rejected all"):
                agent._apply_gemini_video_review(
                    "source.mp4",
                    [{"start": 10, "end": 30, "score": 50}],
                    total_duration=100,
                )

    @patch("agents.clip_agent.create_review_proxy")
    def test_selects_target_count_with_quality_backfill(self, proxy):
        with tempfile.TemporaryDirectory() as output_dir:
            agent = self._agent(output_dir, max_clips=5)
            scores = [90, 80, 72, 65, 50, 40]
            agent.gemini_reviewer = _VideoReviewer({
                index: _review(accepted=index < 2, score=score)
                for index, score in enumerate(scores)
            })
            candidates = [
                {
                    "start": 10 + index * 25,
                    "end": 30 + index * 25,
                    "score": 70 - index,
                    "title": f"Candidate {index}",
                }
                for index in range(6)
            ]

            result = agent._apply_gemini_video_review(
                "source.mp4", candidates, total_duration=200
            )

        self.assertEqual(len(result), 5)
        self.assertEqual(
            sum(item["gemini_video_backfilled"] for item in result), 3
        )
        self.assertEqual(agent.selection_funnel["strictly_accepted"], 2)
        self.assertEqual(agent.selection_funnel["backfilled"], 3)

    def test_boundary_correction_is_clamped_to_duration_limits(self):
        with tempfile.TemporaryDirectory() as output_dir:
            agent = self._agent(output_dir)

            self.assertEqual(
                agent._clamp_reviewed_boundaries(95, 99, 100),
                (85.0, 100.0),
            )
            self.assertEqual(
                agent._clamp_reviewed_boundaries(0, 90, 100),
                (30.0, 90.0),
            )


if __name__ == "__main__":
    unittest.main()
