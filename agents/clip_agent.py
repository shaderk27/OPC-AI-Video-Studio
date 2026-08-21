"""
Clip Expert Agent
==================
Role: Video Highlight Cutter & Clip Editor

This agent analyzes long-form video content (10+ minutes) and identifies
the most interesting, unique, and highlight-worthy moments, then cuts them
into short clips (15-60 seconds) optimized for social media.

Detection Strategies:
1. Transcript-based: Analyze speech for keywords, sentiment shifts, topic changes
2. Audio energy-based: Detect high-energy moments (cheers, laughter, emphasis)
3. Silence-based: Find clean cut points at silence boundaries
4. Combined (default): Uses all strategies together

Output: A set of short clips saved to the clips directory.
"""

import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

from agents.base_agent import BaseAgent
from utils.video_utils import (
    get_video_info,
    cut_clip,
    extract_audio,
    detect_silence,
    get_audio_energy_segments,
    create_review_proxy,
)
from utils.subtitle_utils import transcribe_audio, detect_language_from_segments
from utils.gemini_clip_reviewer import GeminiClipReviewer, GeminiRequiredError


class ClipAgent(BaseAgent):
    """AI agent that cuts interesting highlights from long videos."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(
            name="ClipExpert",
            role="Video Highlight Cutter & Clip Editor",
            config=config,
        )
        self.clip_config = config.get("clip_agent", {})
        self.paths = config.get("paths", {})
        self.min_duration = self.clip_config.get("min_clip_duration", 15)
        self.max_duration = self.clip_config.get("max_clip_duration", 60)
        self.max_clips = self.clip_config.get("max_clips", 5)
        self.min_clips = min(
            self.max_clips,
            max(1, int(self.clip_config.get("min_clips", 3))),
        )
        self.strategy = self.clip_config.get("detection_strategy", "combined")
        self.keywords = self.clip_config.get("highlight_keywords", [])
        self.padding_before = self.clip_config.get("padding_before", 1.5)
        self.padding_after = self.clip_config.get("padding_after", 1.0)
        self.completion_padding_after = self.clip_config.get(
            "completion_padding_after", 6.0
        )
        self.transcript_boundary_max_extension = self.clip_config.get(
            "transcript_boundary_max_extension", 2.0
        )
        self.emotion_audio_fusion_bonus = self.clip_config.get(
            "emotion_audio_fusion_bonus", 15.0
        )
        self.topic_relevance_weight = self.clip_config.get(
            "topic_relevance_weight", 0.55
        )
        self.hook_score_weight = self.clip_config.get("hook_score_weight", 0.45)
        self.topic_relevance_threshold = self.clip_config.get(
            "topic_relevance_threshold", 60.0
        )
        self.snap_to_silence = self.clip_config.get("snap_to_silence", True)
        subtitle_config = config.get("subtitle_agent", {})
        self.whisper_model = subtitle_config.get("whisper_model", "small")
        self.whisper_beam_size = subtitle_config.get("beam_size", 5)
        self.whisper_vad_filter = subtitle_config.get("vad_filter", True)
        self.whisper_vad_min_silence_ms = subtitle_config.get(
            "vad_min_silence_ms", 300
        )
        self.whisper_vocabulary_hints = subtitle_config.get(
            "vocabulary_hints", []
        )
        gemini_config = self.clip_config.get("gemini_review", {})
        self.gemini_review_required = bool(gemini_config.get("required", False))
        self.gemini_review_weight = min(
            1.0, max(0.0, float(gemini_config.get("blend_weight", 0.5)))
        )
        self.gemini_video_review_enabled = (
            bool(gemini_config.get("video_review_enabled", True)) and
            (bool(gemini_config.get("enabled", False)) or self.gemini_review_required)
        )
        self.gemini_video_candidate_limit = max(
            self.max_clips,
            int(gemini_config.get("video_candidate_limit", self.max_clips * 2)),
        )
        self.gemini_video_context_seconds = max(
            0.0, float(gemini_config.get("video_context_seconds", 4.0))
        )
        self.gemini_video_min_score = min(
            100.0, max(0.0, float(gemini_config.get("video_min_score", 60.0)))
        )
        self.gemini_video_backfill_min_score = min(
            self.gemini_video_min_score,
            max(0.0, float(gemini_config.get("video_backfill_min_score", 45.0))),
        )
        self.gemini_video_review_weight = min(
            1.0, max(0.0, float(gemini_config.get("video_blend_weight", 0.8)))
        )
        self.gemini_reviewer = GeminiClipReviewer(config)
        self.selection_funnel: Dict[str, Any] = {}
        if self.gemini_review_required:
            self.gemini_reviewer.enabled = True
            self.gemini_video_review_enabled = True
            self.gemini_reviewer.video_review_enabled = True

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main execution: analyze video, find highlights, cut clips.

        input_data: {"video_path": str, "video_name": str (optional),
                     "content_analysis": {...} (optional, from Content Analyzer),
                     "full_transcript": [...] (optional, pre-computed segments)}
        returns: {"clips": [{"path", "start", "end", "duration", "title", "reason"}]}
        """
        video_path = input_data["video_path"]
        video_name = input_data.get("video_name", Path(video_path).stem)
        self.selection_funnel = {
            "target_clips": self.max_clips,
            "minimum_clips": self.min_clips,
        }

        # Check if Content Analyzer already ran
        content_analysis = input_data.get("content_analysis")
        precomputed_segments = input_data.get("full_transcript", [])
        detected_lang = input_data.get("detected_language")

        # Step 1: Get video info (use from content analysis if available)
        if content_analysis and "video_info" not in input_data:
            self.log("Using video info from Content Analyzer...")
            video_info = get_video_info(video_path)
        else:
            self.log("Getting video info...")
            video_info = input_data.get("video_info") or get_video_info(video_path)

        total_duration = video_info["duration"]
        self.log(f"Video: {video_info['width']}x{video_info['height']}, "
                 f"{total_duration:.1f}s, {video_info['fps']}fps, "
                 f"audio: {video_info['has_audio']}")

        if content_analysis:
            topics = content_analysis.get("topics", [])
            key_moments = content_analysis.get("key_moments", [])
            summary = content_analysis.get("summary", "")
            self.log(f"Content analysis available: {len(topics)} topics, "
                     f"{len(key_moments)} key moments")
            self.log(f"Video summary: {summary[:100]}...")

        if not video_info["has_audio"]:
            self.log("WARNING: No audio track found. Using video-only analysis.")

        # Step 2: Extract highlights based on strategy
        highlights = self._find_highlights(video_path, video_info, input_data)

        if not highlights:
            self.log("No highlights detected. Creating evenly-spaced clips as fallback.")
            highlights = self._fallback_even_clips(total_duration)
            self.selection_funnel.update({
                "generated": len(highlights),
                "deduplicated": len(highlights),
                "shortlisted": len(highlights),
                "used_evenly_spaced_fallback": True,
            })

        # Step 3: Protect the payoff/reaction after each detected moment.
        highlights = self._extend_for_completion(
            highlights, total_duration, precomputed_segments
        )

        # Step 4: Snap to silence boundaries if enabled
        if self.snap_to_silence and video_info["has_audio"]:
            self.log("Snapping clip boundaries to silence...")
            highlights = self._snap_to_silence(video_path, highlights)

        # Step 5: Apply output padding once, while enforcing duration limits.
        highlights = self._finalize_clip_boundaries(highlights, total_duration)

        # Step 6: Visually verify the actual candidate videos. Gemini may reject
        # weak candidates or refine their boundaries using the proxy context.
        highlights = self._apply_gemini_video_review(
            video_path, highlights, total_duration
        )
        self.selection_funnel["selected"] = min(
            len(highlights), self.max_clips
        )
        visual_summary = "visual review skipped"
        if "video_reviewed" in self.selection_funnel:
            visual_summary = (
                f"strictly accepted "
                f"{self.selection_funnel.get('strictly_accepted', 0)} + "
                f"backfilled {self.selection_funnel.get('backfilled', 0)}"
            )
        self.log(
            "Candidate funnel: "
            f"generated {self.selection_funnel.get('generated', 0)} -> "
            f"deduplicated {self.selection_funnel.get('deduplicated', 0)} -> "
            f"shortlisted {self.selection_funnel.get('shortlisted', len(highlights))} -> "
            f"{visual_summary} -> selected "
            f"{self.selection_funnel['selected']}/{self.max_clips}"
        )

        # Step 7: Cut the clips
        clips_dir = self.paths.get("clips_dir", "output/clips")
        self.ensure_dir(clips_dir)
        clips = []

        for i, hl in enumerate(highlights[:self.max_clips]):
            clip_filename = f"{video_name}_clip_{i+1:02d}.mp4"
            clip_path = os.path.join(clips_dir, clip_filename)

            self.log(f"Cutting clip {i+1}/{len(highlights)}: "
                     f"{hl['start']:.1f}s - {hl['end']:.1f}s ({hl['reason']})")

            cut_clip(
                video_path=video_path,
                start_time=hl["start"],
                end_time=hl["end"],
                output_path=clip_path,
                padding_before=0,
                padding_after=0,
            )

            actual_duration = hl["end"] - hl["start"]
            clips.append({
                "path": clip_path,
                "filename": clip_filename,
                "start": hl["start"],
                "end": hl["end"],
                "duration": actual_duration,
                "title": hl.get("title", f"Highlight {i+1}"),
                "reason": hl["reason"],
                "transcript_preview": hl.get("text_preview", ""),
                "detected_language": detected_lang or hl.get("detected_language", ""),
                "selection_score": round(hl.get("score", 0), 1),
                "sentiment_intensity": hl.get("sentiment_intensity", 0),
                "audio_energy_score": hl.get("audio_energy_score", 0),
                "audio_energy_db": hl.get("audio_energy_db"),
                "fusion_bonus": hl.get("fusion_bonus", 0),
                "hook_score": hl.get("hook_score", 0),
                "hook_score_breakdown": hl.get("hook_score_breakdown", {}),
                "topic_relevance": hl.get("topic_relevance", 0),
                "topic_relevance_breakdown": hl.get(
                    "topic_relevance_breakdown", {}
                ),
                "matched_topic": hl.get("matched_topic", ""),
                "is_topic_relevant": hl.get("is_topic_relevant", False),
                "llm_reviewed": hl.get("llm_reviewed", False),
                "llm_editorial_score": hl.get("llm_editorial_score"),
                "llm_score_breakdown": hl.get("llm_score_breakdown", {}),
                "llm_reason": hl.get("llm_reason", ""),
                "pre_llm_selection_score": hl.get("pre_llm_selection_score"),
                "gemini_video_reviewed": hl.get("gemini_video_reviewed", False),
                "gemini_video_accepted": hl.get("gemini_video_accepted"),
                "gemini_video_backfilled": hl.get(
                    "gemini_video_backfilled", False
                ),
                "gemini_video_score": hl.get("gemini_video_score"),
                "gemini_video_score_breakdown": hl.get(
                    "gemini_video_score_breakdown", {}
                ),
                "gemini_event_type": hl.get("gemini_event_type", ""),
                "gemini_video_reason": hl.get("gemini_video_reason", ""),
                "pre_gemini_video_start": hl.get("pre_gemini_video_start"),
                "pre_gemini_video_end": hl.get("pre_gemini_video_end"),
            })

        self.log(f"Created {len(clips)} clips from {video_name}")
        self.selection_funnel["created"] = len(clips)
        return {
            "clips": clips,
            "clip_selection_funnel": dict(self.selection_funnel),
            "video_info": video_info,
            "video_name": video_name,
            "content_analysis": content_analysis,
            "detected_language": detected_lang,
        }

    def _find_highlights(self, video_path: str, video_info: Dict,
                        input_data: Dict = None) -> List[Dict]:
        """Find highlight moments using the configured strategy."""
        highlights = []
        input_data = input_data or {}
        content_analysis = input_data.get("content_analysis")
        topics = content_analysis.get("topics", []) if content_analysis else []
        precomputed_segments = input_data.get("full_transcript", [])
        detected_lang = input_data.get("detected_language")

        if self.strategy in ("transcript", "combined") and video_info["has_audio"]:
            highlights.extend(self._find_transcript_highlights(
                video_path, video_info, precomputed_segments, detected_lang
            ))

        if self.strategy in ("audio_energy", "combined") and video_info["has_audio"]:
            highlights.extend(self._find_energy_highlights(video_path, video_info))

        # Use key moments from Content Analyzer as additional highlight candidates
        if content_analysis:
            key_moment_hls = self._find_key_moment_highlights(content_analysis, video_info)
            highlights.extend(key_moment_hls)

        # Combine emotional and audio evidence before overlapping candidates are
        # deduplicated, so corroborating signals reinforce rather than compete.
        highlights = self._fuse_emotion_and_energy(highlights)
        highlights = self._score_hooks(highlights)
        highlights = self._score_topic_relevance(highlights, topics)
        highlights = self._rank_candidates(highlights, has_topics=bool(topics))
        self.selection_funnel["generated"] = len(highlights)

        # Deduplicate overlapping highlights (keep highest scoring)
        highlights = self._deduplicate_highlights(highlights)
        self.selection_funnel["deduplicated"] = len(highlights)
        highlights = self._apply_gemini_review(
            highlights, precomputed_segments, topics, detected_lang
        )
        highlights = self._select_topic_diverse(highlights)
        self.selection_funnel["shortlisted"] = len(highlights)

        if not highlights and self.strategy == "combined":
            # If combined found nothing, try energy-only as last resort
            highlights = self._find_energy_highlights(video_path, video_info)
            highlights = self._score_hooks(highlights)
            highlights = self._score_topic_relevance(highlights, topics)
            highlights = self._rank_candidates(highlights, has_topics=bool(topics))
            highlights = self._deduplicate_highlights(highlights)
            highlights = self._apply_gemini_review(
                highlights, precomputed_segments, topics, detected_lang
            )
            highlights = self._select_topic_diverse(highlights)
            self.selection_funnel.update({
                "generated": len(highlights),
                "deduplicated": len(highlights),
                "shortlisted": len(highlights),
            })

        return highlights

    def _find_transcript_highlights(self, video_path: str, video_info: Dict,
                                    precomputed_segments: List[Dict] = None,
                                    detected_lang: str = None) -> List[Dict]:
        """
        Transcribe the full video and find interesting moments based on:
        - Keyword matches (configurable highlight keywords)
        - Sentence density (lots of content in a short time)
        - Potential sentiment shifts

        If precomputed_segments is provided (from Content Analyzer),
        skips re-transcription and uses those segments directly.
        """
        if precomputed_segments:
            self.log("Using pre-computed transcript from Content Analyzer (skipping re-transcription)...")
            segments = precomputed_segments
        else:
            self.log("Transcribing video for highlight detection...")

            # Extract audio for transcription
            temp_audio = os.path.join(self.paths.get("output_dir", "output"), "_temp_audio.wav")
            extract_audio(video_path, temp_audio)

            # Transcribe
            source_lang = config_get(self.config, "subtitle_agent.source_language")
            segments = transcribe_audio(
                temp_audio,
                self.whisper_model,
                language=source_lang,
                beam_size=self.whisper_beam_size,
                vad_filter=self.whisper_vad_filter,
                vad_min_silence_ms=self.whisper_vad_min_silence_ms,
                vocabulary_hints=self.whisper_vocabulary_hints,
            )

            # Clean up temp audio
            try:
                os.remove(temp_audio)
            except OSError:
                pass

        if not segments:
            return []

        if not detected_lang:
            detected_lang = detect_language_from_segments(segments)
        self.log(f"Language: {detected_lang}, {len(segments)} segments")

        highlights = []
        keyword_lower = [k.lower() for k in self.keywords]

        for seg in segments:
            text_lower = seg["text"].lower()
            score = 0
            matched_keywords = []

            # Keyword matching
            for kw in keyword_lower:
                if kw in text_lower:
                    score += 10
                    matched_keywords.append(kw)

            # Content density: longer sentences may contain more substance
            word_count = len(seg["text"].split())
            if word_count > 15:
                score += 3
            if word_count > 25:
                score += 2

            # Question marks often indicate engaging content
            if "?" in seg["text"]:
                score += 2

            # Exclamation marks indicate excitement
            if "!" in seg["text"]:
                score += 3

            # Numbers often indicate facts/statistics (interesting content)
            import re
            numbers = re.findall(r'\b\d+\b', seg["text"])
            if numbers:
                score += len(numbers) * 1

            if score > 0:
                # Expand segment to meet minimum duration
                start = seg["start"]
                end = seg["end"]
                duration = end - start

                if duration < self.min_duration:
                    # Expand around the segment
                    expand = (self.min_duration - duration) / 2
                    start = max(0, start - expand)
                    end = end + expand

                if duration > self.max_duration:
                    end = start + self.max_duration

                reason_parts = []
                if matched_keywords:
                    reason_parts.append(f"keywords: {', '.join(matched_keywords[:3])}")
                if word_count > 15:
                    reason_parts.append("content-rich")
                if "!" in seg["text"]:
                    reason_parts.append("excitement marker")
                if "?" in seg["text"]:
                    reason_parts.append("question")

                highlights.append({
                    "start": start,
                    "end": min(end, video_info["duration"]),
                    "score": score,
                    "reason": "; ".join(reason_parts) if reason_parts else "content match",
                    "text_preview": seg["text"][:100],
                    "title": self._generate_title(seg["text"]),
                })

        return highlights

    def _find_energy_highlights(self, video_path: str, video_info: Dict) -> List[Dict]:
        """
        Find high-energy audio moments (cheers, laughter, emphasis, music).
        """
        self.log("Analyzing audio energy for highlight detection...")

        segments = get_audio_energy_segments(video_path, segment_duration=2.0)
        if not segments:
            return []

        # Take top energy segments
        total_duration = video_info["duration"]
        top_count = min(self.max_clips * 3, len(segments))

        highlights = []
        selected_segments = segments[:top_count]
        energy_values = [seg["energy"] for seg in selected_segments]
        min_energy = min(energy_values)
        max_energy = max(energy_values)

        for seg in selected_segments:
            # Expand to minimum clip duration
            start = seg["start"]
            end = seg["end"]
            duration = end - start

            if duration < self.min_duration:
                expand = (self.min_duration - duration) / 2
                start = max(0, start - expand)
                end = end + expand

            if end - start > self.max_duration:
                end = start + self.max_duration

            # Normalize energy to a score (higher energy = higher score)
            # Energy is in dB (negative), closer to 0 = louder
            score = max(1, int(50 + seg["energy"]))  # Convert to positive score
            if max_energy == min_energy:
                energy_score = 100.0
            else:
                energy_score = round(
                    50 + 50 * (seg["energy"] - min_energy) /
                    (max_energy - min_energy),
                    1,
                )

            highlights.append({
                "start": start,
                "end": min(end, total_duration),
                "score": score,
                "reason": f"high audio energy ({seg['energy']:.1f}dB)",
                "text_preview": "",
                "title": f"Energy peak at {seg['start']:.0f}s",
                "source": "audio_energy",
                "audio_peak_start": seg["start"],
                "audio_peak_end": seg["end"],
                "audio_energy_db": round(seg["energy"], 1),
                "audio_energy_score": energy_score,
            })

        return highlights

    def _find_key_moment_highlights(self, content_analysis: Dict, video_info: Dict) -> List[Dict]:
        """
        Use key moments identified by the Content Analyzer as highlight candidates.
        These are pre-scored based on content density, sentiment intensity, and
        keyword presence — providing higher-quality highlights than raw transcript
        scanning alone.
        """
        key_moments = content_analysis.get("key_moments", [])
        if not key_moments:
            return []

        self.log(f"Using {len(key_moments)} key moments from Content Analyzer...")
        highlights = []
        total_duration = video_info["duration"]

        for km in key_moments:
            start = km["start"]
            end = km["end"]
            duration = end - start

            # Expand to minimum clip duration
            if duration < self.min_duration:
                expand = (self.min_duration - duration) / 2
                start = max(0, start - expand)
                end = end + expand

            if end - start > self.max_duration:
                end = start + self.max_duration

            # Use the combined score from content analysis
            score = int(km.get("combined_score", 10))

            # Build reason string
            reason_parts = []
            if km.get("density_score", 0) > 30:
                reason_parts.append("high content density")
            if km.get("sentiment_score", 0) > 0:
                reason_parts.append("emotional intensity")
            if km.get("keyword_hits", 0) > 0:
                reason_parts.append(f"{km['keyword_hits']} keyword hits")
            reason = km.get("reason") or (
                "; ".join(reason_parts) if reason_parts else "key moment"
            )

            highlights.append({
                "start": start,
                "end": min(end, total_duration),
                "score": score,
                "reason": reason,
                "text_preview": km.get("text_preview", ""),
                "title": km.get("title") or self._generate_title(
                    km.get("text", "")
                ),
                "source": "content_key_moment",
                "sentiment_intensity": km.get("sentiment_score", 0),
            })

        return highlights

    def _apply_gemini_review(
        self,
        highlights: List[Dict],
        transcript_segments: List[Dict],
        topics: List[Dict],
        detected_language: str = "",
    ) -> List[Dict]:
        """Blend optional Gemini editorial judgement into candidate ranking."""
        if not highlights:
            return highlights
        if not self.gemini_reviewer.available:
            if self.gemini_review_required:
                raise GeminiRequiredError(
                    "Gemini API is required for this run but is unavailable: "
                    f"{self.gemini_reviewer.unavailable_reason}. "
                    "Check GEMINI_API_KEY and the Gemini SDK installation."
                )
            if self.gemini_reviewer.enabled:
                self.log(
                    f"Gemini review skipped: {self.gemini_reviewer.unavailable_reason}",
                    "warning",
                )
            return highlights

        try:
            reviews = self.gemini_reviewer.review_candidates(
                highlights,
                transcript_segments=transcript_segments,
                topics=topics,
                detected_language=detected_language,
            )
        except Exception as exc:
            if self.gemini_review_required:
                raise GeminiRequiredError(
                    "Gemini API is required for this run, but candidate review failed: "
                    f"{exc}"
                ) from exc
            self.log(
                f"Gemini review unavailable; keeping heuristic ranking: {exc}",
                "warning",
            )
            return highlights

        for index, review in reviews.items():
            highlight = highlights[index]
            heuristic_score = float(highlight.get("score", 0))
            editorial_score = float(review["editorial_score"])
            combined_score = (
                heuristic_score * (1.0 - self.gemini_review_weight) +
                editorial_score * self.gemini_review_weight
            )
            highlight["pre_llm_selection_score"] = round(heuristic_score, 1)
            highlight["llm_reviewed"] = True
            highlight["llm_editorial_score"] = round(editorial_score, 1)
            highlight["llm_score_breakdown"] = review["score_breakdown"]
            highlight["llm_reason"] = review["reason"]
            highlight["score"] = round(min(100.0, max(0.0, combined_score)), 1)
            highlight["final_selection_score"] = highlight["score"]

        self.log(
            f"Gemini reviewed {len(reviews)} candidate(s) with {self.gemini_reviewer.model}"
        )
        return highlights

    def _apply_gemini_video_review(
        self,
        video_path: str,
        highlights: List[Dict],
        total_duration: float,
    ) -> List[Dict]:
        """Visually verify candidate proxies and apply safe boundary edits."""
        if not highlights or not self.gemini_video_review_enabled:
            return highlights
        if not self.gemini_reviewer.available:
            if self.gemini_review_required:
                raise GeminiRequiredError(
                    "Gemini API is required for visual clip review but is unavailable: "
                    f"{self.gemini_reviewer.unavailable_reason}."
                )
            self.log(
                f"Gemini visual review skipped: {self.gemini_reviewer.unavailable_reason}",
                "warning",
            )
            return highlights

        candidates = highlights[:self.gemini_video_candidate_limit]
        try:
            output_dir = self.paths.get("output_dir", "output")
            self.ensure_dir(output_dir)
            with tempfile.TemporaryDirectory(
                prefix="gemini_clip_review_", dir=output_dir
            ) as temp_dir:
                review_candidates = []
                for index, highlight in enumerate(candidates):
                    proxy_start = max(
                        0.0,
                        float(highlight["start"]) - self.gemini_video_context_seconds,
                    )
                    proxy_end = min(
                        total_duration,
                        float(highlight["end"]) + self.gemini_video_context_seconds,
                    )
                    proxy_path = os.path.join(temp_dir, f"candidate_{index}.mp4")
                    create_review_proxy(
                        video_path, proxy_start, proxy_end, proxy_path
                    )
                    review_candidate = dict(highlight)
                    review_candidate.update({
                        "proxy_path": proxy_path,
                        "proxy_start": proxy_start,
                        "proxy_end": proxy_end,
                    })
                    review_candidates.append(review_candidate)

                reviews = self.gemini_reviewer.review_candidate_videos(
                    review_candidates
                )
        except Exception as exc:
            if self.gemini_review_required:
                raise GeminiRequiredError(
                    "Gemini API is required for this run, but visual candidate "
                    f"review failed: {exc}"
                ) from exc
            self.log(
                f"Gemini visual review unavailable; keeping transcript ranking: {exc}",
                "warning",
            )
            return highlights

        strictly_accepted = []
        backfill_pool = []
        for index, review in reviews.items():
            highlight = candidates[index]
            visual_score = float(review["visual_editorial_score"])
            is_accepted = bool(review["accepted"]) and (
                visual_score >= self.gemini_video_min_score
            )
            highlight["gemini_video_reviewed"] = True
            highlight["gemini_video_accepted"] = is_accepted
            highlight["gemini_video_score"] = round(visual_score, 1)
            highlight["gemini_video_score_breakdown"] = review["score_breakdown"]
            highlight["gemini_event_type"] = review["event_type"]
            highlight["gemini_video_reason"] = review["reason"]
            prior_score = float(highlight.get("score", 0))
            highlight["score"] = round(
                prior_score * (1.0 - self.gemini_video_review_weight) +
                visual_score * self.gemini_video_review_weight,
                1,
            )
            highlight["final_selection_score"] = highlight["score"]
            highlight["gemini_video_backfilled"] = False
            if is_accepted:
                strictly_accepted.append((index, highlight, review))
            elif visual_score >= self.gemini_video_backfill_min_score:
                backfill_pool.append((index, highlight, review))

        rank_key = lambda item: (
            -float(item[1].get("score", 0)),
            float(item[1].get("start", 0)),
            float(item[1].get("end", 0)),
            str(item[1].get("title", "")),
        )
        strictly_accepted.sort(key=rank_key)
        backfill_pool.sort(key=rank_key)
        target_count = min(self.max_clips, len(candidates))
        selected = strictly_accepted[:target_count]
        needed = target_count - len(selected)
        if needed > 0:
            additions = backfill_pool[:needed]
            for _, highlight, _ in additions:
                highlight["gemini_video_backfilled"] = True
            selected.extend(additions)
        selected.sort(key=rank_key)

        self.selection_funnel.update({
            "video_reviewed": len(reviews),
            "strictly_accepted": len(strictly_accepted),
            "backfill_eligible": len(backfill_pool),
            "backfilled": sum(
                1 for _, highlight, _ in selected
                if highlight.get("gemini_video_backfilled")
            ),
        })

        if not selected:
            message = (
                "Gemini inspected the candidate videos but rejected all of them "
                "as unsuitable or below the backfill quality threshold."
            )
            if self.gemini_review_required:
                raise GeminiRequiredError(message)
            self.log(f"{message} Keeping the transcript-ranked candidates.", "warning")
            return highlights

        finalized = []
        for index, highlight, review in selected:
            review_candidate = review_candidates[index]
            highlight["pre_gemini_video_start"] = round(highlight["start"], 3)
            highlight["pre_gemini_video_end"] = round(highlight["end"], 3)
            suggested_start = (
                review_candidate["proxy_start"] +
                review["recommended_start_offset"]
            )
            suggested_end = (
                review_candidate["proxy_start"] +
                review["recommended_end_offset"]
            )
            highlight["start"], highlight["end"] = self._clamp_reviewed_boundaries(
                suggested_start, suggested_end, total_duration
            )
            finalized.append(highlight)

        if len(finalized) < min(self.min_clips, target_count):
            self.log(
                f"Only {len(finalized)} candidate(s) met the visual quality floor; "
                f"the configured minimum is {min(self.min_clips, target_count)}.",
                "warning",
            )
        self.log(
            f"Gemini selected {len(finalized)}/{len(candidates)} candidate(s): "
            f"{min(len(strictly_accepted), target_count)} strict, "
            f"{self.selection_funnel['backfilled']} quality backfill(s) "
            f"with {self.gemini_reviewer.model}"
        )
        return finalized

    def _clamp_reviewed_boundaries(
        self,
        start: float,
        end: float,
        total_duration: float,
    ) -> Tuple[float, float]:
        """Keep Gemini boundary suggestions valid and within duration limits."""
        start = min(total_duration, max(0.0, float(start)))
        end = min(total_duration, max(start, float(end)))
        if end - start > self.max_duration:
            start = max(0.0, end - self.max_duration)
        if end - start < self.min_duration:
            missing = self.min_duration - (end - start)
            start = max(0.0, start - missing / 2)
            end = min(total_duration, end + missing / 2)
            if end - start < self.min_duration:
                if start <= 0:
                    end = min(total_duration, self.min_duration)
                else:
                    start = max(0.0, end - self.min_duration)
        return round(start, 3), round(end, 3)

    def _fuse_emotion_and_energy(self, highlights: List[Dict]) -> List[Dict]:
        """Reward key moments corroborated by an overlapping audio peak.

        The semantic key-moment candidate is retained so its title, transcript,
        and reason survive deduplication. Its fused score starts from the larger
        of the two source scores, then adds a bounded bonus based on emotional
        intensity and relative audio energy.
        """
        energy_candidates = [
            hl for hl in highlights if hl.get("source") == "audio_energy"
        ]

        for hl in highlights:
            intensity = hl.get("sentiment_intensity", 0)
            if intensity <= 0 or hl.get("source") != "content_key_moment":
                continue

            overlapping_energy = [
                energy for energy in energy_candidates
                if energy.get("audio_peak_start", energy["start"]) < hl["end"]
                and energy.get("audio_peak_end", energy["end"]) > hl["start"]
            ]
            if not overlapping_energy:
                continue

            strongest = max(
                overlapping_energy,
                key=lambda energy: energy.get("audio_energy_score", 0),
            )
            emotion_factor = min(1.0, intensity / 3.0)
            energy_factor = strongest.get("audio_energy_score", 0) / 100.0
            fusion_bonus = round(
                self.emotion_audio_fusion_bonus * emotion_factor * energy_factor,
                1,
            )

            hl["score"] = round(
                max(hl["score"], strongest["score"]) + fusion_bonus, 1
            )
            hl["audio_energy_score"] = strongest.get("audio_energy_score", 0)
            hl["audio_energy_db"] = strongest.get("audio_energy_db")
            hl["fusion_bonus"] = fusion_bonus
            hl["reason"] = f"{hl['reason']}; emotional + audio peak"

        return highlights

    def _score_hooks(self, highlights: List[Dict]) -> List[Dict]:
        """Calculate an explainable 0-100 engagement hook score.

        The score measures the candidate itself and is intentionally separate
        from topic relevance. Step 5 combines the two for final ranking.
        """
        weights = {
            "opening_strength": 0.25,
            "emotional_intensity": 0.20,
            "audio_energy": 0.20,
            "specificity": 0.20,
            "standalone_coherence": 0.15,
        }

        for hl in highlights:
            text = (hl.get("text_preview") or hl.get("title") or "").strip()
            tokens = re.findall(r"[A-Za-z0-9']+|[\u4e00-\u9fff]", text)
            lower_text = text.lower()
            opening_text = lower_text[:80]

            opening = 20.0 if text else 0.0
            if re.search(r"[?!？！]", opening_text):
                opening += 25.0
            hook_cues = (
                "how", "why", "what", "watch", "look", "wait", "never",
                "secret", "imagine", "this", "here", "看", "为什么",
                "竟然", "原来", "没想到",
            )
            if any(cue in opening_text for cue in hook_cues):
                opening += 35.0
            if 4 <= len(tokens) <= 30:
                opening += 20.0
            opening = min(100.0, opening)

            intensity = max(0.0, float(hl.get("sentiment_intensity", 0)))
            emotional = min(100.0, intensity / 3.0 * 100.0)
            audio = min(100.0, max(0.0, float(hl.get("audio_energy_score", 0))))

            stop_words = {
                "a", "an", "and", "are", "at", "but", "for", "from",
                "in", "is", "it", "of", "on", "or", "that", "the",
                "this", "to", "was", "with", "you",
            }
            content_tokens = {
                token.lower() for token in tokens
                if token.lower() not in stop_words
            }
            specificity = min(60.0, len(content_tokens) * 7.5)
            if re.search(r"\d", text):
                specificity += 20.0
            proper_nouns = re.findall(r"\b[A-Z][a-z]{2,}\b", text)
            if len(proper_nouns) >= 2 or re.search(r"[\u4e00-\u9fff]{2,}", text):
                specificity += 20.0
            specificity = min(100.0, specificity)

            coherence = 20.0 if text else 0.0
            if 4 <= len(tokens) <= 40:
                coherence += 40.0
            if re.search(r"[.!?。！？][\"']?$", text):
                coherence += 25.0
            if text and not text.startswith(("...", "…")) and \
                    not text.endswith(("...", "…")):
                coherence += 15.0
            coherence = min(100.0, coherence)

            components = {
                "opening_strength": round(opening, 1),
                "emotional_intensity": round(emotional, 1),
                "audio_energy": round(audio, 1),
                "specificity": round(specificity, 1),
                "standalone_coherence": round(coherence, 1),
            }
            hook_score = sum(
                components[name] * weight for name, weight in weights.items()
            )
            hl["hook_score"] = round(min(100.0, max(0.0, hook_score)), 1)
            hl["hook_score_breakdown"] = components

        return highlights

    def _score_topic_relevance(self, highlights: List[Dict],
                               topics: List[Dict]) -> List[Dict]:
        """Score how strongly each candidate matches Content Analyzer topics."""
        uninformative_terms = {
            "he", "her", "hers", "him", "his", "i", "it", "its", "me",
            "mine", "my", "our", "ours", "she", "their", "theirs", "them",
            "they", "us", "we", "you", "your", "yours",
        }
        valid_topics = [
            topic for topic in topics
            if str(topic.get("term", "")).strip().lower() not in uninformative_terms
        ]
        max_frequency = max(
            (float(topic.get("frequency", 0)) for topic in valid_topics),
            default=0,
        )

        for hl in highlights:
            text = (hl.get("text_preview") or hl.get("title") or "").strip()
            best_score = 0.0
            best_topic = ""
            best_breakdown = {
                "lexical_match": 0.0,
                "time_overlap": 0.0,
                "topic_prominence": 0.0,
            }

            for topic in valid_topics:
                term = str(topic.get("term", "")).strip()
                if not term:
                    continue

                if re.search(r"[\u4e00-\u9fff]", term):
                    lexical_match = term in text
                else:
                    lexical_match = bool(re.search(
                        rf"(?<!\w){re.escape(term)}(?!\w)", text,
                        flags=re.IGNORECASE,
                    ))

                time_overlap = any(
                    len(time_range) >= 2 and
                    float(time_range[0]) < hl["end"] and
                    float(time_range[1]) > hl["start"]
                    for time_range in topic.get("time_ranges", [])
                )
                prominence = (
                    float(topic.get("frequency", 0)) / max_frequency
                    if max_frequency > 0 else 0
                )
                breakdown = {
                    "lexical_match": 55.0 if lexical_match else 0.0,
                    "time_overlap": 35.0 if time_overlap else 0.0,
                    "topic_prominence": round(10.0 * prominence, 1),
                }
                score = min(100.0, sum(breakdown.values()))

                if score > best_score:
                    best_score = score
                    best_topic = term
                    best_breakdown = breakdown

            hl["topic_relevance"] = round(best_score, 1)
            hl["topic_relevance_breakdown"] = best_breakdown
            hl["matched_topic"] = best_topic
            hl["is_topic_relevant"] = best_score >= self.topic_relevance_threshold

        return highlights

    def _rank_candidates(self, highlights: List[Dict],
                         has_topics: bool) -> List[Dict]:
        """Combine topic relevance and hook potential into the final rank."""
        total_weight = self.topic_relevance_weight + self.hook_score_weight
        if total_weight <= 0:
            topic_weight, hook_weight = 0.55, 0.45
        else:
            topic_weight = self.topic_relevance_weight / total_weight
            hook_weight = self.hook_score_weight / total_weight

        for hl in highlights:
            hl["candidate_score"] = round(float(hl.get("score", 0)), 1)
            if has_topics:
                final_score = (
                    float(hl.get("topic_relevance", 0)) * topic_weight +
                    float(hl.get("hook_score", 0)) * hook_weight
                )
            else:
                final_score = float(hl.get("hook_score", 0))
            hl["score"] = round(min(100.0, max(0.0, final_score)), 1)
            hl["final_selection_score"] = hl["score"]

        return highlights

    def _select_topic_diverse(self, highlights: List[Dict]) -> List[Dict]:
        """Prefer relevant, distinct topics before weaker or repeated matches."""
        ranked = sorted(highlights, key=lambda hl: hl["score"], reverse=True)
        selected = []
        seen_topics = set()
        selection_limit = (
            self.gemini_video_candidate_limit
            if self.gemini_video_review_enabled else self.max_clips
        )

        for hl in ranked:
            topic = hl.get("matched_topic", "").lower()
            if hl.get("is_topic_relevant") and topic and topic not in seen_topics:
                selected.append(hl)
                seen_topics.add(topic)
                if len(selected) >= selection_limit:
                    return selected

        # After topic coverage, keep other relevant clips ahead of candidates
        # that only happen to overlap a topic's timestamp.
        if len(selected) < selection_limit:
            selected_ids = {id(hl) for hl in selected}
            for hl in ranked:
                if id(hl) in selected_ids or not hl.get("is_topic_relevant"):
                    continue
                selected.append(hl)
                if len(selected) >= selection_limit:
                    return selected

        if len(selected) < selection_limit:
            selected_ids = {id(hl) for hl in selected}
            for hl in ranked:
                if id(hl) in selected_ids:
                    continue
                selected.append(hl)
                if len(selected) >= selection_limit:
                    break

        return selected

    def _fallback_even_clips(self, total_duration: float) -> List[Dict]:
        """If no highlights found, create evenly-spaced clips."""
        num_clips = min(self.max_clips, max(1, int(total_duration / 60)))
        clip_duration = min(self.max_duration, total_duration / num_clips)
        spacing = (total_duration - clip_duration * num_clips) / (num_clips + 1)

        highlights = []
        for i in range(num_clips):
            start = spacing * (i + 1) + clip_duration * i
            end = start + clip_duration
            highlights.append({
                "start": start,
                "end": end,
                "score": 1,
                "reason": "evenly spaced fallback",
                "text_preview": "",
                "title": f"Segment {i+1}",
            })

        return highlights

    def _extend_for_completion(self, highlights: List[Dict],
                               total_duration: float,
                               transcript_segments: List[Dict] = None) -> List[Dict]:
        """Reserve time after a highlight for its payoff and reaction.

        When the target endpoint falls inside a transcript segment, extend to
        the end of that segment so commentary is not cut mid-sentence. If the
        result exceeds the maximum duration, move the start forward rather than
        removing the protected ending.
        """
        transcript_segments = sorted(
            transcript_segments or [], key=lambda seg: seg.get("start", 0)
        )

        for hl in highlights:
            if hl.get("reason") == "evenly spaced fallback":
                continue

            desired_end = min(
                total_duration, hl["end"] + self.completion_padding_after
            )

            for seg in transcript_segments:
                seg_start = seg.get("start", 0)
                seg_end = seg.get("end", seg_start)
                if seg_start <= desired_end < seg_end:
                    if (seg_end - desired_end <=
                            self.transcript_boundary_max_extension):
                        desired_end = min(total_duration, seg_end)
                    break

            hl["end"] = desired_end
            if hl["end"] - hl["start"] > self.max_duration:
                hl["start"] = max(0, hl["end"] - self.max_duration)

        return highlights

    def _finalize_clip_boundaries(self, highlights: List[Dict],
                                  total_duration: float) -> List[Dict]:
        """Apply configured padding and keep final metadata within limits."""
        for hl in highlights:
            start = max(0, hl["start"] - self.padding_before)
            end = min(total_duration, hl["end"] + self.padding_after)

            # Preserve the protected payoff when padding pushes past the limit.
            if end - start > self.max_duration:
                start = max(0, end - self.max_duration)

            # Near the end of a source video, recover minimum duration from the
            # beginning side when enough source material is available.
            if end - start < self.min_duration:
                start = max(0, end - self.min_duration)

            hl["start"] = start
            hl["end"] = end

        return highlights

    def _snap_to_silence(self, video_path: str, highlights: List[Dict]) -> List[Dict]:
        """Expand highlight boundaries to nearby silence for clean cuts.

        Starts may only move backward and ends may only move forward. This
        prevents silence snapping from removing either buildup or payoff.
        """
        try:
            silences = detect_silence(video_path, min_silence_duration=0.3)
        except Exception:
            return highlights

        if not silences:
            return highlights

        for hl in highlights:
            # Snap start to nearest silence end before current start
            best_start = hl["start"]
            best_dist = float("inf")
            for s_start, s_end in silences:
                dist = abs(s_end - hl["start"])
                if (dist < best_dist and dist < 2.0 and
                        s_end <= hl["start"] and
                        hl["end"] - s_end <= self.max_duration):
                    best_dist = dist
                    best_start = s_end
            hl["start"] = best_start

            # Snap end to nearest silence start after current end
            best_end = hl["end"]
            best_dist = float("inf")
            for s_start, s_end in silences:
                dist = abs(s_start - hl["end"])
                if (dist < best_dist and dist < 2.0 and
                        s_start >= hl["end"] and
                        s_start - hl["start"] <= self.max_duration):
                    best_dist = dist
                    best_end = s_start
            hl["end"] = best_end

        return highlights

    def _deduplicate_highlights(self, highlights: List[Dict]) -> List[Dict]:
        """Remove overlapping highlights, keeping the highest scoring ones."""
        if not highlights:
            return []

        # Sort by score descending
        highlights.sort(key=lambda x: x["score"], reverse=True)

        kept = []
        for hl in highlights:
            overlaps = False
            for k in kept:
                # Check if this highlight overlaps with a kept one
                overlap_start = max(hl["start"], k["start"])
                overlap_end = min(hl["end"], k["end"])
                if overlap_end > overlap_start:
                    overlap_ratio = (overlap_end - overlap_start) / \
                                    max(hl["end"] - hl["start"], k["end"] - k["start"])
                    if overlap_ratio > 0.3:  # More than 30% overlap
                        overlaps = True
                        break
            if not overlaps:
                kept.append(hl)

        return kept

    def _generate_title(self, text: str) -> str:
        """Generate a short title from segment text."""
        # Take first few words, capitalize
        words = text.split()[:6]
        title = " ".join(words)
        if len(words) == 6 and len(text.split()) > 6:
            title += "..."
        return title.title()[:60]


def config_get(config: Dict, key_path: str):
    """Get nested config value by dot-separated key path."""
    keys = key_path.split(".")
    value = config
    for k in keys:
        if isinstance(value, dict):
            value = value.get(k)
        else:
            return None
    return value
