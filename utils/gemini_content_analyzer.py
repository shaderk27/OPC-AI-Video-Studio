"""Transcript-only Gemini intelligence for the Content Analyzer Agent."""

import json
import math
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)


CONTENT_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "content_type": {"type": "string"},
        "summary": {"type": "string"},
        "topics": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "term": {"type": "string"},
                    "relevance": {"type": "number", "minimum": 0, "maximum": 100},
                    "start_seconds": {"type": "number", "minimum": 0},
                    "end_seconds": {"type": "number", "minimum": 0},
                },
                "required": ["term", "relevance", "start_seconds", "end_seconds"],
            },
        },
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string"},
                    "context": {"type": "string"},
                },
                "required": ["name", "type", "context"],
            },
        },
        "chapters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start_seconds": {"type": "number", "minimum": 0},
                    "end_seconds": {"type": "number", "minimum": 0},
                    "title": {"type": "string"},
                    "summary": {"type": "string"},
                },
                "required": ["start_seconds", "end_seconds", "title", "summary"],
            },
        },
        "key_moments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start_seconds": {"type": "number", "minimum": 0},
                    "end_seconds": {"type": "number", "minimum": 0},
                    "title": {"type": "string"},
                    "reason": {"type": "string"},
                    "importance": {"type": "number", "minimum": 0, "maximum": 100},
                },
                "required": [
                    "start_seconds", "end_seconds", "title", "reason", "importance"
                ],
            },
        },
        "sentiment": {
            "type": "object",
            "properties": {
                "overall": {"type": "string"},
                "emotional_arc": {"type": "string"},
            },
            "required": ["overall", "emotional_arc"],
        },
        "narrative": {
            "type": "object",
            "properties": {
                "opening": {"type": "string"},
                "development": {"type": "string"},
                "climax": {"type": "string"},
                "resolution": {"type": "string"},
            },
            "required": ["opening", "development", "climax", "resolution"],
        },
        "corrected_terms": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "heard": {"type": "string"},
                    "corrected": {"type": "string"},
                },
                "required": ["heard", "corrected"],
            },
        },
    },
    "required": [
        "content_type", "summary", "topics", "entities", "chapters",
        "key_moments", "sentiment", "narrative", "corrected_terms"
    ],
}


class GeminiContentAnalysisError(RuntimeError):
    """Raised when Gemini transcript analysis cannot produce usable output."""


class GeminiContentAnalyzer:
    """Analyze timestamped Whisper segments without uploading the video."""

    def __init__(self, config: Dict[str, Any], client: Any = None):
        settings = config.get("content_analyzer", {}).get("gemini_analysis", {})
        self.enabled = bool(settings.get("enabled", False))
        self.required = bool(settings.get("required", False))
        self.model = settings.get("model", "gemini-3.5-flash-lite")
        self.timeout_ms = max(
            1000, int(float(settings.get("timeout_seconds", 120)) * 1000)
        )
        self.transport_retries = max(0, int(settings.get("transport_retries", 2)))
        self.chunk_max_chars = max(2000, int(settings.get("chunk_max_chars", 50000)))
        self.max_topics = max(1, int(settings.get("max_topics", 8)))
        self.max_key_moments = max(1, int(settings.get("max_key_moments", 12)))
        self.api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        self._client = client

    @property
    def available(self) -> bool:
        return self.enabled and bool(self._client or self.api_key)

    @property
    def unavailable_reason(self) -> str:
        if not self.enabled:
            return "Gemini Content Analyzer is disabled"
        if not self._client and not self.api_key:
            return "GEMINI_API_KEY is not set"
        return ""

    def configuration_error(self) -> str:
        if not self.available:
            return self.unavailable_reason
        try:
            self._get_client()
        except GeminiContentAnalysisError as exc:
            return str(exc)
        return ""

    def analyze(
        self,
        segments: List[Dict[str, Any]],
        duration_seconds: float,
        detected_language: str,
        local_topics: List[Dict[str, Any]],
        local_summary: str,
    ) -> Dict[str, Any]:
        if not self.available or not segments:
            return {}

        chunks = self._chunk_segments(segments)
        analyses = []
        for index, chunk in enumerate(chunks):
            prompt = self._build_prompt(
                chunk,
                index=index,
                chunk_count=len(chunks),
                duration_seconds=duration_seconds,
                detected_language=detected_language,
                local_topics=local_topics,
                local_summary=local_summary,
            )
            response = self._generate_with_transport_retry(
                prompt,
                chunk_number=index + 1,
                chunk_count=len(chunks),
            )

            parsed = getattr(response, "parsed", None)
            if not isinstance(parsed, dict):
                try:
                    parsed = json.loads(getattr(response, "text", "") or "")
                except (TypeError, json.JSONDecodeError) as exc:
                    raise GeminiContentAnalysisError(
                        f"Gemini returned invalid JSON for transcript chunk {index + 1}"
                    ) from exc
            analyses.append(self._validate_analysis(parsed, duration_seconds))

        return self._merge_analyses(analyses)

    def _generate_with_transport_retry(
        self,
        prompt: str,
        *,
        chunk_number: int,
        chunk_count: int,
    ) -> Any:
        """Retry transient network failures without masking API/config errors."""
        max_attempts = self.transport_retries + 1
        for attempt in range(1, max_attempts + 1):
            try:
                return self._get_client().models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config={
                        "response_mime_type": "application/json",
                        "response_json_schema": CONTENT_ANALYSIS_SCHEMA,
                        "temperature": 0,
                    },
                )
            except Exception as exc:
                transient = self._is_transient_transport_error(exc)
                if transient and attempt < max_attempts:
                    time.sleep(min(4.0, float(2 ** (attempt - 1))))
                    continue

                detail = str(exc)
                if transient:
                    detail = (
                        f"network connection failed after {attempt} attempts: {detail}. "
                        "Check DNS/internet access to generativelanguage.googleapis.com "
                        "and retry the run"
                    )
                raise GeminiContentAnalysisError(
                    f"Gemini transcript analysis failed for chunk {chunk_number}/"
                    f"{chunk_count}: {detail}"
                ) from exc

        raise AssertionError("Gemini transport retry loop exited unexpectedly")

    @staticmethod
    def _is_transient_transport_error(exc: Exception) -> bool:
        """Recognize retryable DNS, connection, TLS and timeout failures."""
        retryable_markers = (
            "getaddrinfo failed",
            "name or service not known",
            "temporary failure in name resolution",
            "nodename nor servname provided",
            "connection reset",
            "connection aborted",
            "connection refused",
            "server disconnected",
            "network is unreachable",
            "timed out",
            "timeout",
            "ssl eof",
            "unexpected eof",
        )
        seen = set()
        current: Any = exc
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            if any(marker in str(current).lower() for marker in retryable_markers):
                return True
            current = getattr(current, "__cause__", None) or getattr(
                current, "__context__", None
            )
        return False

    def _get_client(self):
        if self._client is None:
            try:
                from google import genai
            except ImportError as exc:
                raise GeminiContentAnalysisError(
                    "google-genai is not installed; install project requirements"
                ) from exc
            self._client = genai.Client(
                api_key=self.api_key,
                http_options={"timeout": self.timeout_ms},
            )
        return self._client

    def _chunk_segments(self, segments: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        chunks: List[List[Dict[str, Any]]] = []
        current: List[Dict[str, Any]] = []
        current_chars = 0
        for segment in segments:
            text = " ".join(str(segment.get("text", "")).split())
            line_size = len(text) + 40
            if current and current_chars + line_size > self.chunk_max_chars:
                chunks.append(current)
                current = []
                current_chars = 0
            current.append(segment)
            current_chars += line_size
        if current:
            chunks.append(current)
        return chunks

    def _build_prompt(
        self,
        segments: List[Dict[str, Any]],
        index: int,
        chunk_count: int,
        duration_seconds: float,
        detected_language: str,
        local_topics: List[Dict[str, Any]],
        local_summary: str,
    ) -> str:
        transcript = "\n".join(
            f"[{float(seg.get('start', 0)):.2f}-{float(seg.get('end', 0)):.2f}] "
            f"{' '.join(str(seg.get('text', '')).split())}"
            for seg in segments
        )
        hints = {
            "chunk": index + 1,
            "chunk_count": chunk_count,
            "video_duration_seconds": round(float(duration_seconds), 3),
            "detected_language": detected_language or "unknown",
            "local_topic_hints": [str(t.get("term", "")) for t in local_topics[:12]],
            "local_summary_hint": str(local_summary)[:1000],
        }
        return (
            "You are a transcript-based video content analyst. Analyze only the timestamped "
            "transcript supplied below; no video frames are available. Topic hints may contain "
            "Whisper mistakes, so correct obvious names only when context strongly supports the "
            "correction. Identify content type, semantic topics, named entities, chapters, key "
            "moments, sentiment, narrative structure, and a concise summary. Key moments should "
            "be useful downstream clip candidates, but do not pretend to have seen visual events. "
            "All timestamps must satisfy 0 <= start < end <= video duration and remain grounded "
            "in transcript timestamps. Keep summaries and reasons concise.\n\n"
            f"CONTEXT:\n{json.dumps(hints, ensure_ascii=False, separators=(',', ':'))}\n\n"
            f"TRANSCRIPT:\n{transcript}"
        )

    @staticmethod
    def _safe_window(item: Dict[str, Any], duration: float):
        try:
            start = max(0.0, min(duration, float(item.get("start_seconds", 0))))
            end = max(0.0, min(duration, float(item.get("end_seconds", 0))))
        except (TypeError, ValueError):
            return None
        if not math.isfinite(start) or not math.isfinite(end):
            return None
        if end <= start:
            return None
        return round(start, 3), round(end, 3)

    def _validate_analysis(self, parsed: Dict[str, Any], duration_seconds: float) -> Dict[str, Any]:
        duration = max(0.0, float(duration_seconds))
        topics = []
        for item in parsed.get("topics", []) if isinstance(parsed.get("topics"), list) else []:
            if not isinstance(item, dict) or not str(item.get("term", "")).strip():
                continue
            window = self._safe_window(item, duration)
            if not window:
                continue
            relevance = item.get("relevance", 0)
            if isinstance(relevance, bool) or not isinstance(relevance, (int, float)):
                continue
            topics.append({
                "term": str(item["term"]).strip()[:120],
                "relevance": round(max(0.0, min(100.0, float(relevance))), 1),
                "time_ranges": [[window[0], window[1]]],
            })

        chapters = []
        for item in parsed.get("chapters", []) if isinstance(parsed.get("chapters"), list) else []:
            if not isinstance(item, dict):
                continue
            window = self._safe_window(item, duration)
            if window:
                chapters.append({
                    "start": window[0], "end": window[1],
                    "duration": round(window[1] - window[0], 3),
                    "title": str(item.get("title", "Chapter"))[:120],
                    "summary": str(item.get("summary", ""))[:500],
                })

        key_moments = []
        for item in parsed.get("key_moments", []) if isinstance(parsed.get("key_moments"), list) else []:
            if not isinstance(item, dict):
                continue
            window = self._safe_window(item, duration)
            importance = item.get("importance", 0)
            if not window or isinstance(importance, bool) or not isinstance(importance, (int, float)):
                continue
            reason = str(item.get("reason", ""))[:240]
            title = str(item.get("title", "Key moment"))[:120]
            score = round(max(0.0, min(100.0, float(importance))), 1)
            key_moments.append({
                "start": window[0], "end": window[1],
                "duration": round(window[1] - window[0], 3),
                "title": title, "text": title, "text_preview": reason,
                "reason": reason, "combined_score": score,
                "importance": score, "density_score": score,
                "sentiment_score": 0, "keyword_hits": 0,
            })

        entities = []
        for item in parsed.get("entities", []) if isinstance(parsed.get("entities"), list) else []:
            if isinstance(item, dict) and str(item.get("name", "")).strip():
                entities.append({
                    "name": str(item["name"]).strip()[:120],
                    "type": str(item.get("type", "unknown"))[:60],
                    "context": str(item.get("context", ""))[:300],
                })

        corrected_terms = []
        for item in parsed.get("corrected_terms", []) if isinstance(parsed.get("corrected_terms"), list) else []:
            if isinstance(item, dict) and item.get("heard") and item.get("corrected"):
                corrected_terms.append({
                    "heard": str(item["heard"])[:120],
                    "corrected": str(item["corrected"])[:120],
                })

        sentiment = parsed.get("sentiment") if isinstance(parsed.get("sentiment"), dict) else {}
        narrative = parsed.get("narrative") if isinstance(parsed.get("narrative"), dict) else {}
        return {
            "content_type": str(parsed.get("content_type", "unknown"))[:80],
            "summary": str(parsed.get("summary", ""))[:2000],
            "topics": topics,
            "entities": entities,
            "chapters": chapters,
            "key_moments": key_moments,
            "sentiment": {
                "overall": str(sentiment.get("overall", "unknown"))[:40],
                "emotional_arc": str(sentiment.get("emotional_arc", ""))[:500],
            },
            "narrative": {
                key: str(narrative.get(key, ""))[:500]
                for key in ("opening", "development", "climax", "resolution")
            },
            "corrected_terms": corrected_terms,
        }

    def _merge_analyses(self, analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not analyses:
            return {}
        content_types = Counter(
            item.get("content_type", "unknown") for item in analyses
        )
        topic_map: Dict[str, Dict[str, Any]] = {}
        entity_map: Dict[str, Dict[str, Any]] = {}
        correction_map: Dict[str, Dict[str, str]] = {}
        for analysis in analyses:
            for topic in analysis["topics"]:
                key = topic["term"].lower()
                if key not in topic_map:
                    topic_map[key] = dict(topic)
                else:
                    current = topic_map[key]
                    current["relevance"] = max(
                        current["relevance"], topic["relevance"]
                    )
                    current.setdefault("time_ranges", []).extend(
                        topic["time_ranges"]
                    )
            for entity in analysis["entities"]:
                entity_map.setdefault(entity["name"].lower(), entity)
            for correction in analysis["corrected_terms"]:
                correction_map.setdefault(correction["heard"].lower(), correction)

        topics = sorted(
            topic_map.values(), key=lambda item: item["relevance"], reverse=True
        )[:self.max_topics]
        for topic in topics:
            topic["frequency"] = max(1, round(topic.pop("relevance") / 10))

        chapters = sorted(
            [chapter for item in analyses for chapter in item["chapters"]],
            key=lambda item: item["start"],
        )
        key_moments = sorted(
            [moment for item in analyses for moment in item["key_moments"]],
            key=lambda item: item["combined_score"], reverse=True,
        )[:self.max_key_moments]
        key_moments.sort(key=lambda item: item["start"])

        overall_sentiment = Counter(
            item["sentiment"]["overall"] for item in analyses
        ).most_common(1)[0][0]
        narrative = {
            key: " ".join(
                item["narrative"][key] for item in analyses
                if item["narrative"][key]
            )[:1200]
            for key in ("opening", "development", "climax", "resolution")
        }
        return {
            "content_type": content_types.most_common(1)[0][0],
            "summary": " ".join(item["summary"] for item in analyses if item["summary"])[:4000],
            "topics": topics,
            "entities": list(entity_map.values()),
            "chapters": chapters,
            "key_moments": key_moments,
            "sentiment": {
                "overall": overall_sentiment,
                "emotional_arc": " ".join(
                    item["sentiment"]["emotional_arc"] for item in analyses
                    if item["sentiment"]["emotional_arc"]
                )[:1200],
            },
            "narrative": narrative,
            "corrected_terms": list(correction_map.values()),
            "chunks_analyzed": len(analyses),
        }
