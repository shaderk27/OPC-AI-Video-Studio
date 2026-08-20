import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agents.content_analyzer_agent import ContentAnalyzerAgent
from utils.gemini_clip_reviewer import GeminiRequiredError
from utils.gemini_content_analyzer import (
    GeminiContentAnalyzer,
    GeminiContentAnalysisError,
)


def _analysis(start=0, end=20, topic="football"):
    return {
        "content_type": "sports",
        "summary": f"Summary for {topic}.",
        "topics": [{
            "term": topic,
            "relevance": 90,
            "start_seconds": start,
            "end_seconds": end,
        }],
        "entities": [{
            "name": "Lionel Messi", "type": "person", "context": "Player"
        }],
        "chapters": [{
            "start_seconds": start,
            "end_seconds": end,
            "title": "Opening",
            "summary": "The sequence begins.",
        }],
        "key_moments": [{
            "start_seconds": start + 2,
            "end_seconds": end,
            "title": "Important statement",
            "reason": "A meaningful transcript moment.",
            "importance": 85,
        }],
        "sentiment": {"overall": "positive", "emotional_arc": "Rising"},
        "narrative": {
            "opening": "Setup", "development": "Build",
            "climax": "Payoff", "resolution": "Reaction",
        },
        "corrected_terms": [{"heard": "Messy", "corrected": "Messi"}],
    }


class _Models:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(parsed=self.responses.pop(0), text="")


class _Client:
    def __init__(self, responses):
        self.models = _Models(responses)


def _config(**overrides):
    settings = {
        "enabled": True,
        "required": True,
        "model": "gemini-3.5-flash-lite",
        "chunk_max_chars": 2000,
        "max_topics": 8,
        "max_key_moments": 12,
    }
    settings.update(overrides)
    return {"content_analyzer": {"gemini_analysis": settings}}


class GeminiContentAnalyzerTests(unittest.TestCase):
    def test_returns_downstream_compatible_semantic_analysis(self):
        client = _Client([_analysis()])
        analyzer = GeminiContentAnalyzer(_config(), client=client)

        result = analyzer.analyze(
            [{"start": 0, "end": 20, "text": "Messy scores the goal."}],
            duration_seconds=30,
            detected_language="en",
            local_topics=[{"term": "Messy"}],
            local_summary="A football sequence.",
        )

        self.assertEqual(result["content_type"], "sports")
        self.assertEqual(result["topics"][0]["term"], "football")
        self.assertEqual(result["topics"][0]["time_ranges"], [[0.0, 20.0]])
        self.assertEqual(result["key_moments"][0]["combined_score"], 85.0)
        self.assertEqual(result["corrected_terms"][0]["corrected"], "Messi")
        self.assertIn("timestamped transcript", client.models.calls[0]["contents"])
        self.assertEqual(client.models.calls[0]["config"]["temperature"], 0)

    def test_chunks_long_transcript_and_merges_topics(self):
        segments = [
            {"start": 0, "end": 10, "text": "A" * 1500},
            {"start": 10, "end": 20, "text": "B" * 1500},
        ]
        client = _Client([
            _analysis(0, 10, "football"),
            _analysis(10, 20, "football"),
        ])
        analyzer = GeminiContentAnalyzer(_config(chunk_max_chars=2000), client=client)

        result = analyzer.analyze(segments, 20, "en", [], "")

        self.assertEqual(result["chunks_analyzed"], 2)
        self.assertEqual(len(client.models.calls), 2)
        self.assertEqual(result["topics"][0]["time_ranges"],
                         [[0.0, 10.0], [10.0, 20.0]])

    def test_discards_individual_invalid_timestamps(self):
        parsed = _analysis(start=40, end=50)
        parsed["chapters"].append({
            "start_seconds": 5,
            "end_seconds": 10,
            "title": "Valid",
            "summary": "Valid chapter.",
        })
        analyzer = GeminiContentAnalyzer(_config(), client=_Client([]))

        result = analyzer._validate_analysis(parsed, duration_seconds=30)

        self.assertEqual(len(result["chapters"]), 1)
        self.assertEqual(result["chapters"][0]["title"], "Valid")

    def test_required_agent_mode_propagates_analysis_failure(self):
        agent = ContentAnalyzerAgent(_config())
        fake = SimpleNamespace(
            available=True,
            required=True,
            enabled=True,
            model="gemini-3.5-flash-lite",
            analyze=lambda *args, **kwargs: (_ for _ in ()).throw(
                GeminiContentAnalysisError("timed out")
            ),
        )
        agent.gemini_analyzer = fake
        agent.gemini_required = True

        with self.assertRaisesRegex(GeminiRequiredError, "Content Analyzer"):
            agent._run_gemini_analysis([], 30, "en", [], "")


if __name__ == "__main__":
    unittest.main()
