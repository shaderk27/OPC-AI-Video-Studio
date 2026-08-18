"""
Content Analyzer Agent
=======================
Role: Video Content Intelligence Analyst

This agent runs as a PRE-PROCESSING step before the Clip Expert.
It transcribes the full video and performs deep content analysis:

1. Topic detection — extracts key themes via keyword frequency
2. Sentiment analysis — tracks emotional arc across the video
3. Content density scoring — rates information richness per segment
4. Narrative structure — identifies intro, body, conclusion, transitions
5. Chapter detection — groups segments into thematic chapters
6. Key moment detection — finds the most quotable, impactful moments
7. Summary generation — creates a brief text summary of the video

The analysis is passed downstream to the Clip Expert (smarter highlight
selection), Social Media Copywriter (better captions), and other agents.

Pipeline position: content_analysis -> clip -> subtitle -> audio -> ...
"""

import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agents.base_agent import BaseAgent
from utils.video_utils import get_video_info, extract_audio
from utils.subtitle_utils import transcribe_audio, detect_language_from_segments


# ─────────────────────────────────────────────────────────────────────────────
# Sentiment lexicons (simple, no external dependencies)
# ─────────────────────────────────────────────────────────────────────────────
POSITIVE_WORDS = {
    "amazing", "incredible", "awesome", "fantastic", "great", "wonderful",
    "excellent", "perfect", "beautiful", "love", "best", "brilliant",
    "exciting", "excited", "happy", "glad", "delighted", "stunning",
    "magnificent", "spectacular", "unbelievable", "remarkable", "superb",
    "outstanding", "marvelous", "fabulous", "terrific", "phenomenal",
    "breathtaking", "gorgeous", "impressive", "inspiring", "thrilling",
    "fascinating", "captivating", "memorable", "unforgettable", "epic",
    "win", "winner", "victory", "success", "succeed", "triumph", "glory",
    "breakthrough", "innovation", "revolutionary", "game-changer",
}

NEGATIVE_WORDS = {
    "terrible", "awful", "bad", "horrible", "worst", "hate", "disgusting",
    "disappointing", "disappointed", "frustrating", "frustrated", "angry",
    "sad", "depressing", "boring", "dull", "poor", "fail", "failure",
    "failed", "mistake", "wrong", "problem", "issue", "difficult", "hard",
    "struggle", "struggling", "painful", "pain", "hurt", "broken",
    "crash", "disaster", "catastrophe", "nightmare", "tragedy", "loss",
    "lost", "defeat", "rejected", "denied", "blocked", "stopped",
}

# Stopwords for topic detection (English + common Chinese particles)
STOPWORDS = {
    # English
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "must", "can", "this", "that",
    "these", "those", "i", "you", "he", "she", "it", "we", "they",
    "what", "which", "who", "when", "where", "why", "how", "all",
    "each", "every", "some", "any", "few", "more", "most", "other",
    "into", "through", "during", "before", "after", "above", "below",
    "up", "down", "out", "off", "over", "under", "again", "further",
    "then", "once", "here", "there", "so", "than", "too", "very",
    "just", "also", "only", "own", "same", "such", "no", "not", "nor",
    "now", "about", "get", "got", "go", "going", "gone", "like",
    "really", "actually", "basically", "literally", "kind", "sort",
    "thing", "things", "stuff", "way", "ways", "lot", "lots", "bit",
    "yeah", "yes", "ok", "okay", "well", "right", "let", "lets",
    # Common Chinese particles (single char)
    "的", "了", "是", "在", "我", "你", "他", "她", "它", "们",
    "这", "那", "个", "们", "和", "与", "或", "但", "不", "没",
    "有", "就", "都", "也", "还", "又", "才", "只", "能", "会",
    "要", "想", "说", "做", "看", "去", "来", "到", "给", "对",
}

# Transition phrases that signal narrative structure changes
INTRO_PHRASES = [
    "today", "in this video", "let's", "welcome", "hello", "hi everyone",
    "we're going to", "i'm going to", "let me", "first", "to start",
    "introduction", "begin", "getting started",
]
CONCLUSION_PHRASES = [
    "finally", "in conclusion", "to sum up", "summary", "wrap up",
    "that's it", "thanks for", "thank you", "see you", "until next",
    "that's all", "conclusion", "overall", "in the end",
]
TRANSITION_PHRASES = [
    "now", "next", "then", "moving on", "let's talk about",
    "another", "also", "additionally", "furthermore", "however",
    "but first", "on the other hand", "meanwhile", "let's move",
]


class ContentAnalyzerAgent(BaseAgent):
    """Pre-processes video content to provide intelligence for downstream agents."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(
            name="ContentAnalyzer",
            role="Video Content Intelligence Analyst",
            config=config,
        )
        self.ca_config = config.get("content_analyzer", {})
        self.paths = config.get("paths", {})
        self.whisper_model = config.get("subtitle_agent", {}).get("whisper_model", "base")
        self.source_lang = config.get("subtitle_agent", {}).get("source_language")
        self.max_topics = self.ca_config.get("max_topics", 8)
        self.chapter_min_duration = self.ca_config.get("chapter_min_duration", 30.0)
        self.key_moment_count = self.ca_config.get("key_moment_count", 10)
        self.summary_sentence_count = self.ca_config.get("summary_sentences", 3)
        self.save_report = self.ca_config.get("save_report", True)

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze full video content and produce intelligence report.

        input_data: {"video_path": str, "video_name": str}
        returns: {"content_analysis": {...}, "segments": [...], ...}
        """
        video_path = input_data["video_path"]
        video_name = input_data.get("video_name", Path(video_path).stem)

        # Step 1: Get video info
        self.log("Getting video info...")
        video_info = get_video_info(video_path)
        total_duration = video_info["duration"]

        if not video_info["has_audio"]:
            self.log("WARNING: No audio track. Content analysis limited to metadata only.")
            return self._minimal_analysis(video_info, video_name)

        # Step 2: Extract audio and transcribe the FULL video
        self.log("Extracting audio for full transcription...")
        temp_audio = os.path.join(
            self.paths.get("output_dir", "output"),
            f"_{video_name}_full_audio.wav"
        )
        extract_audio(video_path, temp_audio)

        self.log(f"Transcribing full video ({total_duration:.0f}s) with Whisper "
                 f"({self.whisper_model})...")
        segments = transcribe_audio(temp_audio, self.whisper_model, language=self.source_lang)

        # Clean up temp audio
        try:
            os.remove(temp_audio)
        except OSError:
            pass

        if not segments:
            self.log("WARNING: Transcription returned no segments.")
            return self._minimal_analysis(video_info, video_name)

        detected_lang = detect_language_from_segments(segments)
        self.log(f"Transcribed {len(segments)} segments, detected language: {detected_lang}")

        # Step 3: Run all analysis modules
        self.log("Running topic detection...")
        topics = self._detect_topics(segments)

        self.log("Running sentiment analysis...")
        sentiment = self._analyze_sentiment(segments)

        self.log("Scoring content density...")
        density = self._score_content_density(segments)

        self.log("Detecting narrative structure...")
        narrative = self._detect_narrative_structure(segments, total_duration)

        self.log("Detecting chapters...")
        chapters = self._detect_chapters(segments, topics, total_duration)

        self.log("Identifying key moments...")
        key_moments = self._find_key_moments(segments, sentiment, density)

        self.log("Generating summary...")
        summary = self._generate_summary(segments, topics, key_moments)

        # Step 4: Assemble the analysis report
        analysis = {
            "video_name": video_name,
            "video_duration": total_duration,
            "detected_language": detected_lang,
            "total_segments": len(segments),
            "total_words": sum(len(s["text"].split()) for s in segments),
            "topics": topics,
            "sentiment": sentiment,
            "content_density": density,
            "narrative_structure": narrative,
            "chapters": chapters,
            "key_moments": key_moments,
            "summary": summary,
            "segments": segments,  # Full transcript for downstream agents
        }

        # Save report
        if self.save_report:
            report_path = os.path.join(
                self.paths.get("output_dir", "output"),
                "content_analysis.json"
            )
            self.ensure_dir(os.path.dirname(report_path))
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(analysis, f, indent=2, ensure_ascii=False)
            self.log(f"Content analysis report saved to: {report_path}")

        # Print summary
        self.log(f"Analysis complete: {len(topics)} topics, "
                 f"{len(chapters)} chapters, {len(key_moments)} key moments")
        self.log(f"Summary: {summary[:120]}...")

        return {
            "content_analysis": analysis,
            "video_info": video_info,
            "video_name": video_name,
            "full_transcript": segments,
            "detected_language": detected_lang,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Topic Detection
    # ─────────────────────────────────────────────────────────────────────────
    def _detect_topics(self, segments: List[Dict]) -> List[Dict]:
        """
        Extract key topics via word frequency analysis.
        Returns topics sorted by frequency with associated time ranges.
        """
        word_freq = Counter()
        word_times: Dict[str, List[Tuple[float, float]]] = {}

        for seg in segments:
            words = self._tokenize(seg["text"])
            for word in words:
                if len(word) < 2 and not self._is_cjk(word):
                    continue
                if word.lower() in STOPWORDS:
                    continue
                word_freq[word] += 1
                if word not in word_times:
                    word_times[word] = []
                word_times[word].append((seg["start"], seg["end"]))

        # Get top topics
        topics = []
        for word, count in word_freq.most_common(self.max_topics * 3):
            times = word_times[word]
            # Calculate time spread
            first_seen = times[0][0]
            last_seen = times[-1][1]
            time_spread = last_seen - first_seen

            topics.append({
                "term": word,
                "frequency": count,
                "first_seen": round(first_seen, 1),
                "last_seen": round(last_seen, 1),
                "time_spread": round(time_spread, 1),
                "occurrences": len(times),
                "time_ranges": [(round(s, 1), round(e, 1)) for s, e in times[:10]],
            })

            if len(topics) >= self.max_topics:
                break

        return topics

    # ─────────────────────────────────────────────────────────────────────────
    # Sentiment Analysis
    # ─────────────────────────────────────────────────────────────────────────
    def _analyze_sentiment(self, segments: List[Dict]) -> Dict:
        """
        Track sentiment across the video using lexicon-based scoring.
        Returns overall sentiment + per-segment scores + emotional arc.
        """
        segment_scores = []
        positive_count = 0
        negative_count = 0
        total_score = 0

        for seg in segments:
            words = self._tokenize(seg["text"])
            if not words:
                segment_scores.append({
                    "start": seg["start"],
                    "end": seg["end"],
                    "score": 0,
                    "label": "neutral",
                })
                continue

            pos_hits = sum(1 for w in words if w.lower() in POSITIVE_WORDS)
            neg_hits = sum(1 for w in words if w.lower() in NEGATIVE_WORDS)
            score = pos_hits - neg_hits

            if score > 0:
                label = "positive"
                positive_count += 1
            elif score < 0:
                label = "negative"
                negative_count += 1
            else:
                label = "neutral"

            total_score += score
            segment_scores.append({
                "start": seg["start"],
                "end": seg["end"],
                "score": score,
                "positive_hits": pos_hits,
                "negative_hits": neg_hits,
                "label": label,
            })

        # Detect sentiment shifts (transitions from pos to neg or vice versa)
        shifts = []
        for i in range(1, len(segment_scores)):
            prev = segment_scores[i - 1]["label"]
            curr = segment_scores[i]["label"]
            if (prev == "positive" and curr == "negative") or \
               (prev == "negative" and curr == "positive"):
                shifts.append({
                    "time": segment_scores[i]["start"],
                    "from": prev,
                    "to": curr,
                })

        # Overall sentiment
        if total_score > 0:
            overall = "positive"
        elif total_score < 0:
            overall = "negative"
        else:
            overall = "neutral"

        # Emotional arc: aggregate into 10 buckets
        num_buckets = min(10, len(segment_scores))
        arc = []
        if num_buckets > 0:
            bucket_size = len(segment_scores) / num_buckets
            for i in range(num_buckets):
                start_idx = int(i * bucket_size)
                end_idx = int((i + 1) * bucket_size)
                bucket = segment_scores[start_idx:end_idx]
                avg_score = sum(s["score"] for s in bucket) / len(bucket) if bucket else 0
                arc.append({
                    "bucket": i + 1,
                    "start_time": bucket[0]["start"] if bucket else 0,
                    "end_time": bucket[-1]["end"] if bucket else 0,
                    "avg_score": round(avg_score, 2),
                    "label": "positive" if avg_score > 0 else
                             ("negative" if avg_score < 0 else "neutral"),
                })

        return {
            "overall": overall,
            "total_score": total_score,
            "positive_segments": positive_count,
            "negative_segments": negative_count,
            "neutral_segments": len(segment_scores) - positive_count - negative_count,
            "sentiment_shifts": shifts,
            "emotional_arc": arc,
            "segment_scores": segment_scores,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Content Density Scoring
    # ─────────────────────────────────────────────────────────────────────────
    def _score_content_density(self, segments: List[Dict]) -> Dict:
        """
        Rate each segment by information richness.
        Factors: words per second, unique word ratio, number density.
        """
        scored_segments = []
        all_scores = []

        for seg in segments:
            text = seg["text"]
            duration = seg["end"] - seg["start"]
            if duration <= 0:
                continue

            words = self._tokenize(text)
            word_count = len(words)
            unique_words = len(set(w.lower() for w in words))

            # Words per second (speaking rate)
            wps = word_count / duration

            # Unique word ratio (vocabulary diversity)
            uniqueness = unique_words / word_count if word_count > 0 else 0

            # Number density (facts/statistics)
            numbers = re.findall(r'\b\d+(?:\.\d+)?\b', text)
            number_density = len(numbers) / max(word_count, 1)

            # Content words ratio (non-stopword)
            content_words = sum(1 for w in words if w.lower() not in STOPWORDS)
            content_ratio = content_words / max(word_count, 1)

            # Combined density score (0-100 scale)
            density_score = min(100, (
                wps * 8 +           # Speaking rate (max ~40)
                uniqueness * 25 +    # Vocabulary diversity (max ~25)
                number_density * 100 + # Number density (max ~20)
                content_ratio * 25   # Content word ratio (max ~25)
            ))

            scored_segments.append({
                "start": seg["start"],
                "end": seg["end"],
                "word_count": word_count,
                "words_per_second": round(wps, 2),
                "uniqueness": round(uniqueness, 3),
                "number_count": len(numbers),
                "content_ratio": round(content_ratio, 3),
                "density_score": round(density_score, 1),
            })
            all_scores.append(density_score)

        avg_density = sum(all_scores) / len(all_scores) if all_scores else 0
        max_density = max(all_scores) if all_scores else 0
        min_density = min(all_scores) if all_scores else 0

        # Find densest segments (top 5)
        top_segments = sorted(scored_segments, key=lambda x: x["density_score"], reverse=True)[:5]

        return {
            "average_density": round(avg_density, 1),
            "max_density": round(max_density, 1),
            "min_density": round(min_density, 1),
            "densest_segments": top_segments,
            "all_segments": scored_segments,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Narrative Structure Detection
    # ─────────────────────────────────────────────────────────────────────────
    def _detect_narrative_structure(self, segments: List[Dict], total_duration: float) -> Dict:
        """
        Identify narrative sections: intro, body, conclusion.
        Detects transition points using phrase matching and time position.
        """
        sections = {
            "intro": {"start": 0, "end": total_duration * 0.1, "segments": []},
            "body": {"start": total_duration * 0.1, "end": total_duration * 0.9, "segments": []},
            "conclusion": {"start": total_duration * 0.9, "end": total_duration, "segments": []},
        }

        intro_score = 0
        conclusion_score = 0
        transition_points = []

        for seg in segments:
            text_lower = seg["text"].lower()

            # Check for intro phrases
            for phrase in INTRO_PHRASES:
                if phrase in text_lower:
                    intro_score += 1
                    break

            # Check for conclusion phrases
            for phrase in CONCLUSION_PHRASES:
                if phrase in text_lower:
                    conclusion_score += 1
                    break

            # Check for transition phrases
            for phrase in TRANSITION_PHRASES:
                if phrase in text_lower:
                    transition_points.append({
                        "time": seg["start"],
                        "phrase": phrase,
                        "text_preview": seg["text"][:80],
                    })
                    break

            # Assign to section based on time
            if seg["start"] < total_duration * 0.1:
                sections["intro"]["segments"].append(seg["text"])
            elif seg["start"] > total_duration * 0.9:
                sections["conclusion"]["segments"].append(seg["text"])
            else:
                sections["body"]["segments"].append(seg["text"])

        return {
            "sections": {
                "intro": {
                    "start": sections["intro"]["start"],
                    "end": sections["intro"]["end"],
                    "segment_count": len(sections["intro"]["segments"]),
                    "text": " ".join(sections["intro"]["segments"])[:500],
                    "confidence": intro_score,
                },
                "body": {
                    "start": sections["body"]["start"],
                    "end": sections["body"]["end"],
                    "segment_count": len(sections["body"]["segments"]),
                },
                "conclusion": {
                    "start": sections["conclusion"]["start"],
                    "end": sections["conclusion"]["end"],
                    "segment_count": len(sections["conclusion"]["segments"]),
                    "text": " ".join(sections["conclusion"]["segments"])[:500],
                    "confidence": conclusion_score,
                },
            },
            "transition_points": transition_points[:20],
            "intro_confidence": intro_score,
            "conclusion_confidence": conclusion_score,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Chapter Detection
    # ─────────────────────────────────────────────────────────────────────────
    def _detect_chapters(self, segments: List[Dict], topics: List[Dict],
                         total_duration: float) -> List[Dict]:
        """
        Group segments into thematic chapters based on topic shifts
        and transition phrases. Each chapter covers a coherent topic.
        """
        if not segments:
            return []

        # Build a topic map: word -> topic index
        topic_map = {}
        for i, topic in enumerate(topics):
            topic_map[topic["term"].lower()] = i

        # Assign dominant topic to each segment
        seg_topics = []
        for seg in segments:
            words = self._tokenize(seg["text"])
            topic_hits = Counter()
            for word in words:
                if word.lower() in topic_map:
                    topic_hits[topic_map[word.lower()]] += 1

            dominant_topic = topics[topic_hits.most_common(1)[0][0]]["term"] \
                if topic_hits else None
            seg_topics.append({
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
                "dominant_topic": dominant_topic,
                "topic_count": sum(topic_hits.values()),
            })

        # Group consecutive segments with same topic into chapters
        chapters = []
        current_chapter = {
            "start": seg_topics[0]["start"],
            "end": seg_topics[0]["end"],
            "topic": seg_topics[0]["dominant_topic"],
            "segment_count": 1,
            "texts": [seg_topics[0]["text"]],
        }

        for seg in seg_topics[1:]:
            same_topic = seg["dominant_topic"] == current_chapter["topic"]
            chapter_duration = current_chapter["end"] - current_chapter["start"]

            # Start new chapter if topic changed or chapter is long enough
            if not same_topic and chapter_duration >= self.chapter_min_duration:
                chapters.append(self._finalize_chapter(current_chapter))
                current_chapter = {
                    "start": seg["start"],
                    "end": seg["end"],
                    "topic": seg["dominant_topic"],
                    "segment_count": 1,
                    "texts": [seg["text"]],
                }
            else:
                current_chapter["end"] = seg["end"]
                current_chapter["segment_count"] += 1
                current_chapter["texts"].append(seg["text"])

        # Add final chapter
        chapters.append(self._finalize_chapter(current_chapter))

        # Merge very short chapters into neighbors
        chapters = self._merge_short_chapters(chapters)

        return chapters

    def _finalize_chapter(self, chapter: Dict) -> Dict:
        """Clean up a chapter dict for output."""
        full_text = " ".join(chapter["texts"])
        return {
            "start": round(chapter["start"], 1),
            "end": round(chapter["end"], 1),
            "duration": round(chapter["end"] - chapter["start"], 1),
            "topic": chapter["topic"] or "general",
            "segment_count": chapter["segment_count"],
            "text_preview": full_text[:200],
        }

    def _merge_short_chapters(self, chapters: List[Dict]) -> List[Dict]:
        """Merge chapters shorter than chapter_min_duration into neighbors."""
        if len(chapters) <= 1:
            return chapters

        merged = [chapters[0]]
        for chapter in chapters[1:]:
            if chapter["duration"] < self.chapter_min_duration:
                # Merge into previous
                prev = merged[-1]
                prev["end"] = chapter["end"]
                prev["duration"] = round(prev["end"] - prev["start"], 1)
                prev["segment_count"] += chapter["segment_count"]
                prev["text_preview"] = (prev["text_preview"] + " " + chapter["text_preview"])[:200]
            else:
                merged.append(chapter)

        return merged

    # ─────────────────────────────────────────────────────────────────────────
    # Key Moment Detection
    # ─────────────────────────────────────────────────────────────────────────
    def _find_key_moments(self, segments: List[Dict],
                          sentiment: Dict, density: Dict) -> List[Dict]:
        """
        Identify the most quotable, impactful moments using a combined score
        of sentiment intensity + content density + keyword presence.
        """
        # Build density lookup
        density_map = {round(s["start"], 1): s for s in density.get("all_segments", [])}

        # Build sentiment lookup
        sentiment_map = {round(s["start"], 1): s for s in sentiment.get("segment_scores", [])}

        scored_moments = []

        for seg in segments:
            start_key = round(seg["start"], 1)
            d_score = density_map.get(start_key, {}).get("density_score", 0)
            s_score = abs(sentiment_map.get(start_key, {}).get("score", 0))

            # Keyword bonus
            words = self._tokenize(seg["text"])
            keyword_hits = sum(1 for w in words if w.lower() in POSITIVE_WORDS)

            # Length penalty for very short or very long segments
            duration = seg["end"] - seg["start"]
            length_factor = 1.0
            if duration < 2:
                length_factor = 0.5
            elif duration > 30:
                length_factor = 0.7

            combined_score = (d_score + s_score * 15 + keyword_hits * 5) * length_factor

            scored_moments.append({
                "start": round(seg["start"], 1),
                "end": round(seg["end"], 1),
                "duration": round(duration, 1),
                "text": seg["text"],
                "text_preview": seg["text"][:120],
                "density_score": d_score,
                "sentiment_score": s_score,
                "keyword_hits": keyword_hits,
                "combined_score": round(combined_score, 1),
            })

        # Sort by combined score and take top N
        scored_moments.sort(key=lambda x: x["combined_score"], reverse=True)
        top_moments = scored_moments[:self.key_moment_count]

        # Sort by time for output
        top_moments.sort(key=lambda x: x["start"])

        return top_moments

    # ─────────────────────────────────────────────────────────────────────────
    # Summary Generation
    # ─────────────────────────────────────────────────────────────────────────
    def _generate_summary(self, segments: List[Dict], topics: List[Dict],
                          key_moments: List[Dict]) -> str:
        """
        Generate a brief text summary using extractive summarization.
        Picks the most representative sentences based on topic coverage
        and key moment scores.
        """
        if not segments:
            return "No content to summarize."

        # Build sentence list with scores
        sentences = []
        topic_terms = set(t["term"].lower() for t in topics[:5])

        for seg in segments:
            # Split segment into sentences
            seg_sentences = re.split(r'[.!?。！？]', seg["text"])
            for sentence in seg_sentences:
                sentence = sentence.strip()
                if len(sentence) < 10:
                    continue

                words = self._tokenize(sentence)
                topic_coverage = sum(1 for w in words if w.lower() in topic_terms)
                word_count = len(words)

                # Score: topic coverage + length appropriateness
                length_bonus = 1.0 if 10 <= word_count <= 30 else 0.5
                score = (topic_coverage * 3 + length_bonus) * word_count

                sentences.append({
                    "text": sentence,
                    "score": score,
                    "time": seg["start"],
                })

        if not sentences:
            # Fallback: use first segment text
            return segments[0]["text"][:200]

        # Sort by score and take top N
        sentences.sort(key=lambda x: x["score"], reverse=True)
        top_sentences = sentences[:self.summary_sentence_count]

        # Sort selected sentences by time for coherent flow
        top_sentences.sort(key=lambda x: x["time"])

        summary = ". ".join(s["text"] for s in top_sentences)
        if not summary.endswith("."):
            summary += "."

        return summary

    # ─────────────────────────────────────────────────────────────────────────
    # Helper Methods
    # ─────────────────────────────────────────────────────────────────────────
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words. Handles both English (space-split)
        and Chinese (character-level) text.
        """
        # For Chinese text, split into individual characters
        cjk_chars = [c for c in text if '\u4e00' <= c <= '\u9fff']
        if len(cjk_chars) > len(text) * 0.3:
            # Mostly Chinese — use character + bigram tokenization
            tokens = []
            # Individual meaningful characters
            for c in text:
                if '\u4e00' <= c <= '\u9fff' and c not in STOPWORDS:
                    tokens.append(c)
            # Also extract any English words mixed in
            en_words = re.findall(r'[a-zA-Z]+', text)
            tokens.extend(en_words)
            return tokens
        else:
            # Mostly English — split on non-alphanumeric
            words = re.findall(r'[a-zA-Z]+(?:\'[a-zA-Z]+)?', text)
            return words

    def _is_cjk(self, text: str) -> bool:
        """Check if text contains CJK characters."""
        return any('\u4e00' <= c <= '\u9fff' for c in text)

    def _minimal_analysis(self, video_info: Dict, video_name: str) -> Dict:
        """Return minimal analysis when no audio is available."""
        return {
            "content_analysis": {
                "video_name": video_name,
                "video_duration": video_info["duration"],
                "detected_language": "unknown",
                "total_segments": 0,
                "total_words": 0,
                "topics": [],
                "sentiment": {"overall": "unknown", "segment_scores": []},
                "content_density": {"average_density": 0, "all_segments": []},
                "narrative_structure": {"sections": {}, "transition_points": []},
                "chapters": [],
                "key_moments": [],
                "summary": "No audio content available for analysis.",
                "segments": [],
            },
            "video_info": video_info,
            "video_name": video_name,
            "full_transcript": [],
            "detected_language": "unknown",
        }
