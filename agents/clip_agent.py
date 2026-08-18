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
from pathlib import Path
from typing import Any, Dict, List, Tuple

from agents.base_agent import BaseAgent
from utils.video_utils import (
    get_video_info,
    cut_clip,
    extract_audio,
    detect_silence,
    get_audio_energy_segments,
)
from utils.subtitle_utils import transcribe_audio, detect_language_from_segments


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
        self.strategy = self.clip_config.get("detection_strategy", "combined")
        self.keywords = self.clip_config.get("highlight_keywords", [])
        self.padding_before = self.clip_config.get("padding_before", 1.5)
        self.padding_after = self.clip_config.get("padding_after", 1.0)
        self.snap_to_silence = self.clip_config.get("snap_to_silence", True)
        self.whisper_model = config.get("subtitle_agent", {}).get("whisper_model", "base")

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

        # Step 3: Snap to silence boundaries if enabled
        if self.snap_to_silence and video_info["has_audio"]:
            self.log("Snapping clip boundaries to silence...")
            highlights = self._snap_to_silence(video_path, highlights)

        # Step 4: Cut the clips
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
                padding_before=self.padding_before,
                padding_after=self.padding_after,
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
            })

        self.log(f"Created {len(clips)} clips from {video_name}")
        return {
            "clips": clips,
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

        # Deduplicate overlapping highlights (keep highest scoring)
        highlights = self._deduplicate_highlights(highlights)

        if not highlights and self.strategy == "combined":
            # If combined found nothing, try energy-only as last resort
            highlights = self._find_energy_highlights(video_path, video_info)
            highlights = self._deduplicate_highlights(highlights)

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
            segments = transcribe_audio(temp_audio, self.whisper_model, language=source_lang)

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
        for seg in segments[:top_count]:
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

            highlights.append({
                "start": start,
                "end": min(end, total_duration),
                "score": score,
                "reason": f"high audio energy ({seg['energy']:.1f}dB)",
                "text_preview": "",
                "title": f"Energy peak at {seg['start']:.0f}s",
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
            reason = "; ".join(reason_parts) if reason_parts else "key moment"

            highlights.append({
                "start": start,
                "end": min(end, total_duration),
                "score": score,
                "reason": reason,
                "text_preview": km.get("text_preview", ""),
                "title": self._generate_title(km.get("text", "")),
            })

        return highlights

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

    def _snap_to_silence(self, video_path: str, highlights: List[Dict]) -> List[Dict]:
        """Adjust highlight boundaries to nearby silence for clean cuts."""
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
                if dist < best_dist and dist < 2.0 and s_end < hl["end"]:
                    best_dist = dist
                    best_start = s_end
            hl["start"] = best_start

            # Snap end to nearest silence start after current end
            best_end = hl["end"]
            best_dist = float("inf")
            for s_start, s_end in silences:
                dist = abs(s_start - hl["end"])
                if dist < best_dist and dist < 2.0 and s_start > hl["start"]:
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
