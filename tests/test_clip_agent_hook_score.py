import unittest

from agents.clip_agent import ClipAgent


class ClipAgentHookScoreTests(unittest.TestCase):
    def setUp(self):
        self.agent = ClipAgent({})

    def test_hook_score_is_weighted_and_explainable(self):
        candidate = {
            "text_preview": "Wait! 3 reasons this penalty changed the final.",
            "sentiment_intensity": 3,
            "audio_energy_score": 100,
        }

        result = self.agent._score_hooks([candidate])[0]

        self.assertEqual(result["hook_score"], 93.0)
        self.assertEqual(result["hook_score_breakdown"], {
            "opening_strength": 100.0,
            "emotional_intensity": 100.0,
            "audio_energy": 100.0,
            "specificity": 65.0,
            "standalone_coherence": 100.0,
        })

    def test_hook_score_stays_in_zero_to_one_hundred_range(self):
        candidates = [
            {},
            {
                "text_preview": "Look! Amazing!",
                "sentiment_intensity": 999,
                "audio_energy_score": 999,
            },
        ]

        results = self.agent._score_hooks(candidates)

        for candidate in results:
            self.assertGreaterEqual(candidate["hook_score"], 0)
            self.assertLessEqual(candidate["hook_score"], 100)


if __name__ == "__main__":
    unittest.main()
