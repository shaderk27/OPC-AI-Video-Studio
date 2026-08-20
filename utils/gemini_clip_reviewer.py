"""Gemini-backed editorial review for ClipAgent candidates.

This module is deliberately isolated from the rest of the pipeline.  Gemini is
an optional ranking signal: it never cuts video, changes timestamps, or becomes
required for the existing heuristic workflow to operate.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv


# Resolve from this file rather than the current working directory so CLI and
# web entrypoints load the same project-level .env file. Existing shell or
# deployment environment variables take precedence.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)


REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "reviews": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "candidate_id": {"type": "string"},
                    "hook": {"type": "number", "minimum": 0, "maximum": 100},
                    "flow": {"type": "number", "minimum": 0, "maximum": 100},
                    "value": {"type": "number", "minimum": 0, "maximum": 100},
                    "standalone": {"type": "number", "minimum": 0, "maximum": 100},
                    "reason": {"type": "string"},
                },
                "required": [
                    "candidate_id", "hook", "flow", "value", "standalone", "reason"
                ],
            },
        }
    },
    "required": ["reviews"],
}


class GeminiReviewError(RuntimeError):
    """Raised when Gemini does not return a complete, usable review."""


class GeminiClipReviewer:
    """Review transcript-based clip candidates with Gemini structured output."""

    def __init__(self, config: Dict[str, Any], client: Any = None):
        review_config = config.get("clip_agent", {}).get("gemini_review", {})
        self.enabled = bool(review_config.get("enabled", False))
        self.model = review_config.get("model", "gemini-2.5-flash-lite")
        self.candidate_limit = max(1, int(review_config.get("candidate_limit", 12)))
        self.timeout_ms = max(1000, int(review_config.get("timeout_seconds", 45) * 1000))
        self.context_seconds = max(0.0, float(review_config.get("context_seconds", 5.0)))
        self.max_transcript_chars = max(
            200, int(review_config.get("max_transcript_chars", 2400))
        )
        self.api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        self._client = client

    @property
    def available(self) -> bool:
        """Whether a review call can be attempted."""
        return self.enabled and bool(self._client or self.api_key)

    @property
    def unavailable_reason(self) -> str:
        if not self.enabled:
            return "Gemini clip review is disabled"
        if not self._client and not self.api_key:
            return "GEMINI_API_KEY is not set"
        return ""

    def review_candidates(
        self,
        candidates: List[Dict[str, Any]],
        transcript_segments: Optional[List[Dict[str, Any]]] = None,
        topics: Optional[List[Dict[str, Any]]] = None,
        detected_language: str = "",
    ) -> Dict[int, Dict[str, Any]]:
        """Return validated reviews keyed by index in ``candidates``.

        Only the highest-ranked configured number of candidates are sent.  A
        response must cover every submitted candidate exactly once; otherwise
        the whole response is rejected so partially reviewed candidates do not
        receive an unfair ranking advantage.
        """
        if not self.available or not candidates:
            return {}

        ranked_indices = sorted(
            range(len(candidates)),
            key=lambda index: float(candidates[index].get("score", 0)),
            reverse=True,
        )[:self.candidate_limit]
        submitted = self._build_candidate_payload(
            candidates, ranked_indices, transcript_segments or []
        )
        prompt = self._build_prompt(submitted, topics or [], detected_language)

        try:
            response = self._get_client().models.generate_content(
                model=self.model,
                contents=prompt,
                config={
                    "temperature": 0.2,
                    "response_mime_type": "application/json",
                    "response_json_schema": REVIEW_SCHEMA,
                },
            )
        except Exception as exc:
            raise GeminiReviewError(f"Gemini request failed: {exc}") from exc

        parsed = getattr(response, "parsed", None)
        if not isinstance(parsed, dict):
            response_text = getattr(response, "text", "") or ""
            try:
                parsed = json.loads(response_text)
            except (TypeError, json.JSONDecodeError) as exc:
                raise GeminiReviewError("Gemini returned invalid JSON") from exc

        expected_ids = {item["candidate_id"] for item in submitted}
        reviews = self._validate_reviews(parsed, expected_ids)
        return {
            int(candidate_id.removeprefix("candidate_")): review
            for candidate_id, review in reviews.items()
        }

    def _get_client(self):
        if self._client is None:
            try:
                from google import genai
            except ImportError as exc:
                raise GeminiReviewError(
                    "google-genai is not installed; install project requirements"
                ) from exc

            self._client = genai.Client(
                api_key=self.api_key,
                http_options={"timeout": self.timeout_ms},
            )
        return self._client

    def _build_candidate_payload(
        self,
        candidates: List[Dict[str, Any]],
        indices: List[int],
        transcript_segments: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        payload = []
        for index in indices:
            candidate = candidates[index]
            context = self._transcript_context(candidate, transcript_segments)
            payload.append({
                "candidate_id": f"candidate_{index}",
                "start_seconds": round(float(candidate.get("start", 0)), 3),
                "end_seconds": round(float(candidate.get("end", 0)), 3),
                "title": str(candidate.get("title", ""))[:200],
                "transcript": context,
                "matched_topic": str(candidate.get("matched_topic", ""))[:120],
                "heuristic_score": round(float(candidate.get("score", 0)), 1),
                "hook_score": round(float(candidate.get("hook_score", 0)), 1),
                "topic_relevance": round(
                    float(candidate.get("topic_relevance", 0)), 1
                ),
                "audio_energy_score": round(
                    float(candidate.get("audio_energy_score", 0)), 1
                ),
            })
        return payload

    def _transcript_context(
        self,
        candidate: Dict[str, Any],
        transcript_segments: List[Dict[str, Any]],
    ) -> str:
        start = float(candidate.get("start", 0)) - self.context_seconds
        end = float(candidate.get("end", 0)) + self.context_seconds
        lines = []
        for segment in transcript_segments:
            seg_start = float(segment.get("start", 0))
            seg_end = float(segment.get("end", seg_start))
            if seg_start < end and seg_end > start:
                text = " ".join(str(segment.get("text", "")).split())
                if text:
                    lines.append(f"[{seg_start:.1f}-{seg_end:.1f}] {text}")

        context = "\n".join(lines)
        if not context:
            context = str(candidate.get("text_preview", ""))
        return context[:self.max_transcript_chars]

    @staticmethod
    def _build_prompt(
        candidates: List[Dict[str, Any]],
        topics: List[Dict[str, Any]],
        detected_language: str,
    ) -> str:
        compact_topics = [
            {
                "term": str(topic.get("term", ""))[:120],
                "frequency": topic.get("frequency", 0),
            }
            for topic in topics[:12]
            if str(topic.get("term", "")).strip()
        ]
        evidence = {
            "language": detected_language or "unknown",
            "video_topics": compact_topics,
            "candidates": candidates,
        }
        return (
            "You are an editorial reviewer selecting self-contained short-form video clips. "
            "Review only the supplied transcript and metadata; no video frames are available. "
            "Score each candidate independently from 0 to 100 for: hook (immediate attention), "
            "flow (logical progression and payoff), value (information, emotion, or entertainment), "
            "and standalone (understandable without the full video). Do not reward a candidate just "
            "because its heuristic score is high. Do not alter timestamps. Return exactly one review "
            "for every candidate_id and no extra candidates. Keep each reason under 240 characters.\n\n"
            f"EVIDENCE:\n{json.dumps(evidence, ensure_ascii=False, separators=(',', ':'))}"
        )

    @staticmethod
    def _validate_reviews(
        parsed: Dict[str, Any], expected_ids: set
    ) -> Dict[str, Dict[str, Any]]:
        raw_reviews = parsed.get("reviews")
        if not isinstance(raw_reviews, list):
            raise GeminiReviewError("Gemini response is missing the reviews list")

        validated = {}
        for item in raw_reviews:
            if not isinstance(item, dict):
                raise GeminiReviewError("Gemini returned a malformed review")
            candidate_id = item.get("candidate_id")
            if candidate_id not in expected_ids or candidate_id in validated:
                raise GeminiReviewError("Gemini returned unknown or duplicate candidate IDs")

            scores = {}
            for name in ("hook", "flow", "value", "standalone"):
                value = item.get(name)
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise GeminiReviewError(f"Gemini returned an invalid {name} score")
                scores[name] = round(min(100.0, max(0.0, float(value))), 1)

            editorial_score = round(
                scores["hook"] * 0.30 +
                scores["flow"] * 0.25 +
                scores["value"] * 0.25 +
                scores["standalone"] * 0.20,
                1,
            )
            validated[candidate_id] = {
                "editorial_score": editorial_score,
                "score_breakdown": scores,
                "reason": str(item.get("reason", ""))[:240],
            }

        if set(validated) != expected_ids:
            raise GeminiReviewError("Gemini did not review every submitted candidate")
        return validated
