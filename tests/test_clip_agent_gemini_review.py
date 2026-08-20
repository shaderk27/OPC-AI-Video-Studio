import unittest

from agents.clip_agent import ClipAgent


class _Reviewer:
    enabled = True
    available = True
    unavailable_reason = ""
    model = "gemini-2.5-flash-lite"

    def __init__(self, reviews=None, error=None):
        self.reviews = reviews or {}
        self.error = error

    def review_candidates(self, *args, **kwargs):
        if self.error:
            raise self.error
        return self.reviews


class ClipAgentGeminiReviewTests(unittest.TestCase):
    def setUp(self):
        self.agent = ClipAgent({
            "clip_agent": {
                "gemini_review": {
                    "enabled": True,
                    "blend_weight": 0.5,
                },
            },
        })

    def test_blends_editorial_and_heuristic_scores(self):
        self.agent.gemini_reviewer = _Reviewer({
            0: {
                "editorial_score": 90,
                "score_breakdown": {
                    "hook": 90, "flow": 90, "value": 90, "standalone": 90,
                },
                "reason": "Complete and engaging.",
            },
        })
        candidates = [{"score": 50, "final_selection_score": 50}]

        result = self.agent._apply_gemini_review(candidates, [], [], "en")

        self.assertEqual(result[0]["score"], 70.0)
        self.assertEqual(result[0]["pre_llm_selection_score"], 50.0)
        self.assertEqual(result[0]["llm_editorial_score"], 90.0)
        self.assertTrue(result[0]["llm_reviewed"])
        self.assertEqual(result[0]["llm_reason"], "Complete and engaging.")

    def test_request_failure_keeps_original_ranking(self):
        self.agent.gemini_reviewer = _Reviewer(error=TimeoutError("timed out"))
        candidates = [{"score": 63, "final_selection_score": 63}]

        result = self.agent._apply_gemini_review(candidates, [], [], "en")

        self.assertEqual(result, [{"score": 63, "final_selection_score": 63}])


if __name__ == "__main__":
    unittest.main()
