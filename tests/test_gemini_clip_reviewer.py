import json
import os
import unittest
from unittest.mock import patch

from utils.gemini_clip_reviewer import GeminiClipReviewer, GeminiReviewError


class _FakeResponse:
    def __init__(self, parsed=None, text=""):
        self.parsed = parsed
        self.text = text


class _FakeModels:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class _FakeClient:
    def __init__(self, response):
        self.models = _FakeModels(response)


def _config(**overrides):
    settings = {
        "enabled": True,
        "model": "gemini-3.5-flash-lite",
        "candidate_limit": 2,
        "timeout_seconds": 10,
        "context_seconds": 1,
        "max_transcript_chars": 500,
    }
    settings.update(overrides)
    return {"clip_agent": {"gemini_review": settings}}


class GeminiClipReviewerTests(unittest.TestCase):
    def test_builds_transcript_prompt_and_calculates_editorial_score(self):
        response = _FakeResponse(parsed={
            "reviews": [{
                "candidate_id": "candidate_0",
                "hook": 100,
                "flow": 80,
                "value": 60,
                "standalone": 40,
                "reason": "Clear hook and payoff.",
            }],
        })
        client = _FakeClient(response)
        reviewer = GeminiClipReviewer(
            _config(candidate_limit=1), client=client
        )

        reviews = reviewer.review_candidates(
            [{
                "start": 10,
                "end": 20,
                "title": "A useful moment",
                "score": 75,
                "hook_score": 70,
            }],
            transcript_segments=[
                {"start": 9.5, "end": 12, "text": "Here is the surprising result."},
                {"start": 30, "end": 31, "text": "Unrelated material."},
            ],
            topics=[{"term": "result", "frequency": 2}],
            detected_language="en",
        )

        self.assertEqual(reviews[0]["editorial_score"], 73.0)
        self.assertEqual(reviews[0]["reason"], "Clear hook and payoff.")
        call = client.models.calls[0]
        self.assertEqual(call["model"], "gemini-3.5-flash-lite")
        self.assertEqual(call["config"]["response_mime_type"], "application/json")
        self.assertIn("Here is the surprising result.", call["contents"])
        self.assertNotIn("Unrelated material.", call["contents"])

    def test_accepts_json_text_when_parsed_response_is_unavailable(self):
        response = _FakeResponse(text=json.dumps({
            "reviews": [{
                "candidate_id": "candidate_0",
                "hook": 50,
                "flow": 50,
                "value": 50,
                "standalone": 50,
                "reason": "Average candidate.",
            }],
        }))
        reviewer = GeminiClipReviewer(
            _config(candidate_limit=1), client=_FakeClient(response)
        )

        reviews = reviewer.review_candidates([{"score": 1}])

        self.assertEqual(reviews[0]["editorial_score"], 50.0)

    def test_rejects_partial_reviews(self):
        response = _FakeResponse(parsed={
            "reviews": [{
                "candidate_id": "candidate_0",
                "hook": 50,
                "flow": 50,
                "value": 50,
                "standalone": 50,
                "reason": "Only one was returned.",
            }],
        })
        reviewer = GeminiClipReviewer(_config(), client=_FakeClient(response))

        with self.assertRaisesRegex(GeminiReviewError, "every submitted candidate"):
            reviewer.review_candidates([{"score": 2}, {"score": 1}])

    def test_missing_api_key_disables_remote_call_safely(self):
        with patch.dict(os.environ, {}, clear=True):
            reviewer = GeminiClipReviewer(_config())

        self.assertFalse(reviewer.available)
        self.assertEqual(reviewer.review_candidates([{"score": 1}]), {})
        self.assertEqual(reviewer.unavailable_reason, "GEMINI_API_KEY is not set")


if __name__ == "__main__":
    unittest.main()
