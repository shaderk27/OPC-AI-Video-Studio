import unittest

from agents.clip_agent import ClipAgent


class ClipAgentTopicRelevanceTests(unittest.TestCase):
    def setUp(self):
        self.agent = ClipAgent({
            "clip_agent": {
                "max_clips": 3,
                "topic_relevance_weight": 0.55,
                "hook_score_weight": 0.45,
                "topic_relevance_threshold": 60,
            },
        })

    def test_explicit_topic_and_time_overlap_score_one_hundred(self):
        candidate = {
            "start": 100.0,
            "end": 115.0,
            "text_preview": "Lionel Messi steps up for the penalty.",
        }
        topics = [{
            "term": "Messi",
            "frequency": 6,
            "time_ranges": [[98.0, 101.0], [111.0, 114.0]],
        }]

        result = self.agent._score_topic_relevance([candidate], topics)[0]

        self.assertEqual(result["topic_relevance"], 100.0)
        self.assertEqual(result["matched_topic"], "Messi")
        self.assertTrue(result["is_topic_relevant"])

    def test_time_overlap_alone_does_not_claim_semantic_relevance(self):
        candidate = {
            "start": 100.0,
            "end": 115.0,
            "text_preview": "The crowd becomes louder.",
        }
        topics = [{
            "term": "Messi",
            "frequency": 6,
            "time_ranges": [[98.0, 114.0]],
        }]

        result = self.agent._score_topic_relevance([candidate], topics)[0]

        self.assertEqual(result["topic_relevance"], 45.0)
        self.assertFalse(result["is_topic_relevant"])

    def test_common_pronoun_topic_is_ignored(self):
        candidate = {
            "start": 10.0,
            "end": 20.0,
            "text_preview": "This is the winning goal.",
        }
        topics = [{
            "term": "his",
            "frequency": 10,
            "time_ranges": [[10.0, 20.0]],
        }]

        result = self.agent._score_topic_relevance([candidate], topics)[0]

        self.assertEqual(result["topic_relevance"], 0.0)
        self.assertEqual(result["matched_topic"], "")

    def test_final_rank_uses_relevance_and_hook_weights(self):
        candidate = {
            "score": 999,
            "topic_relevance": 80,
            "hook_score": 60,
        }

        result = self.agent._rank_candidates([candidate], has_topics=True)[0]

        self.assertEqual(result["candidate_score"], 999.0)
        self.assertEqual(result["final_selection_score"], 71.0)
        self.assertEqual(result["score"], 71.0)

    def test_selection_prefers_distinct_topics_before_duplicates(self):
        highlights = [
            {"score": 95, "matched_topic": "A", "title": "A1", "is_topic_relevant": True},
            {"score": 90, "matched_topic": "A", "title": "A2", "is_topic_relevant": True},
            {"score": 85, "matched_topic": "B", "title": "B1", "is_topic_relevant": True},
            {"score": 80, "matched_topic": "C", "title": "C1", "is_topic_relevant": True},
        ]

        selected = self.agent._select_topic_diverse(highlights)

        self.assertEqual([item["title"] for item in selected], ["A1", "B1", "C1"])

    def test_relevant_duplicate_precedes_higher_scoring_irrelevant_clip(self):
        highlights = [
            {"score": 95, "matched_topic": "A", "title": "A1", "is_topic_relevant": True},
            {"score": 90, "matched_topic": "X", "title": "X1", "is_topic_relevant": False},
            {"score": 85, "matched_topic": "B", "title": "B1", "is_topic_relevant": True},
            {"score": 80, "matched_topic": "A", "title": "A2", "is_topic_relevant": True},
        ]

        selected = self.agent._select_topic_diverse(highlights)

        self.assertEqual([item["title"] for item in selected], ["A1", "B1", "A2"])


if __name__ == "__main__":
    unittest.main()
