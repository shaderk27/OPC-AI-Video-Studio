import unittest

from agents.content_analyzer_agent import ContentAnalyzerAgent


class ContentAnalyzerSentimentTests(unittest.TestCase):
    def setUp(self):
        self.agent = ContentAnalyzerAgent({
            "content_analyzer": {"key_moment_count": 10},
        })

    def test_mixed_emotion_preserves_intensity_when_polarity_cancels(self):
        segments = [{
            "start": 0.0,
            "end": 4.0,
            "text": "Amazing victory, but a terrible pain.",
        }]

        result = self.agent._analyze_sentiment(segments)
        segment = result["segment_scores"][0]

        self.assertEqual(segment["score"], 0)
        self.assertEqual(segment["label"], "neutral")
        self.assertEqual(segment["positive_hits"], 2)
        self.assertEqual(segment["negative_hits"], 2)
        self.assertEqual(segment["intensity"], 4)
        self.assertEqual(result["emotional_arc"][0]["avg_intensity"], 4)

    def test_key_moment_counts_negative_and_mixed_emotional_words(self):
        segments = [{
            "start": 0.0,
            "end": 4.0,
            "text": "Amazing victory, but a terrible pain.",
        }]
        sentiment = self.agent._analyze_sentiment(segments)
        density = {
            "all_segments": [{"start": 0.0, "density_score": 10}],
        }

        moment = self.agent._find_key_moments(
            segments, sentiment, density
        )[0]

        self.assertEqual(moment["sentiment_score"], 4)
        self.assertEqual(moment["sentiment_polarity"], 0)
        self.assertEqual(moment["keyword_hits"], 4)
        self.assertEqual(moment["combined_score"], 90)


if __name__ == "__main__":
    unittest.main()
