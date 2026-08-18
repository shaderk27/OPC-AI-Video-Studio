"""
Social Media Agent (Enhancement)
=================================
Role: Social Media Copywriter & Content Strategist

Writes platform-specific captions, hashtags, and descriptions for each clip.
Generates tailored copy for TikTok, YouTube Shorts, Instagram Reels, and X.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List
from collections import Counter

from agents.base_agent import BaseAgent
from deep_translator import GoogleTranslator


class SocialMediaAgent(BaseAgent):
    """Generates social media copy for clips."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(
            name="SocialMediaCopywriter",
            role="Social Media Copywriter & Content Strategist",
            config=config,
        )
        self.sm_config = config.get("social_media_agent", {})
        self.paths = config.get("paths", {})
        self.platforms = self.sm_config.get("platforms", [])
        self.tone = self.sm_config.get("tone", "energetic")

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate social media copy for each clip."""
        clips = input_data.get("clips", [])
        social_dir = self.paths.get("social_dir", "output/social")
        self.ensure_dir(social_dir)

        updated_clips = []

        for i, clip in enumerate(clips):
            clip_name = Path(clip["filename"]).stem
            self.log(f"Writing social media copy for {clip_name}")

            # Extract keywords from transcript
            transcript = self._get_transcript_text(clip)
            keywords = self._extract_keywords(transcript)

            # Generate copy for each platform
            platform_copy = {}
            for platform in self.platforms:
                pname = platform["name"]
                caption = self._generate_caption(
                    clip, transcript, keywords, platform
                )
                hashtags = self._generate_hashtags(keywords, platform, clip)

                platform_copy[pname] = {
                    "caption": caption,
                    "hashtags": hashtags,
                    "full_post": f"{caption}\n\n{' '.join(hashtags)}",
                }

            # Save to JSON file
            copy_filename = f"{clip_name}_social.json"
            copy_path = Path(social_dir) / copy_filename
            with open(copy_path, "w", encoding="utf-8") as f:
                json.dump(platform_copy, f, indent=2, ensure_ascii=False)

            clip["social_media"] = {
                "platforms": platform_copy,
                "keywords": keywords,
                "copy_file": str(copy_path),
            }
            updated_clips.append(clip)

        self.log(f"Generated social media copy for {len(updated_clips)} clips")
        return {"clips": updated_clips, **{k: v for k, v in input_data.items() if k != "clips"}}

    def _get_transcript_text(self, clip: Dict) -> str:
        """Get transcript text from clip subtitles."""
        subtitles = clip.get("subtitles", {})
        for lang_data in subtitles.values():
            segments = lang_data.get("segments", [])
            if segments:
                return " ".join(seg["text"] for seg in segments)
        return clip.get("transcript_preview", clip.get("title", ""))

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text (simple frequency-based)."""
        if not text:
            return []

        # Remove common stop words
        stop_words = {
            "the", "a", "an", "and", "or", "but", "is", "are", "was", "were",
            "be", "been", "being", "have", "has", "had", "do", "does", "did",
            "will", "would", "could", "should", "may", "might", "must", "can",
            "this", "that", "these", "those", "i", "you", "he", "she", "it",
            "we", "they", "what", "which", "who", "when", "where", "why", "how",
            "all", "each", "every", "both", "few", "more", "most", "other",
            "some", "such", "no", "nor", "not", "only", "own", "same", "so",
            "than", "too", "very", "just", "of", "to", "in", "for", "on", "with",
            "at", "by", "from", "up", "about", "into", "through", "during",
            "before", "after", "above", "below", "between", "my", "your",
        }

        words = re.findall(r'\b[A-Za-z]{3,}\b', text.lower())
        words = [w for w in words if w not in stop_words]

        counter = Counter(words)
        return [word for word, _ in counter.most_common(10)]

    def _generate_caption(self, clip: Dict, transcript: str,
                          keywords: List[str], platform: Dict) -> str:
        """Generate a caption for a specific platform."""
        title = clip.get("title", "Amazing moment")
        max_len = platform.get("max_caption_length", 200)

        # Build caption based on tone
        if self.tone == "energetic":
            hooks = [
                "You won't believe what happened! ",
                "This is absolutely incredible! ",
                "Wait for it... ",
                "This changed everything! ",
                "The moment everyone was waiting for! ",
            ]
        elif self.tone == "educational":
            hooks = [
                "Learn this key insight: ",
                "Here's what you need to know: ",
                "Important lesson: ",
                "Key takeaway: ",
            ]
        elif self.tone == "professional":
            hooks = [
                "Highlight: ",
                "Key moment: ",
                "Notable segment: ",
            ]
        else:
            hooks = [""]

        # Pick hook based on keywords hash for consistency
        hook = hooks[0] if not keywords else hooks[len(keywords[0]) % len(hooks)]

        # Build the caption
        keyword_str = ", ".join(keywords[:3]) if keywords else "highlights"
        caption = f"{hook}{title} - {keyword_str}"

        # Add call to action
        ctas = {
            "TikTok": " Follow for more! ",
            "YouTube Shorts": " Subscribe for more! ",
            "Instagram Reels": " Save this for later! ",
            "X (Twitter)": " RT if you enjoyed! ",
        }
        cta = ctas.get(platform["name"], "")
        caption = f"{caption}{cta}"

        # Truncate to platform limit
        if len(caption) > max_len:
            caption = caption[:max_len - 3] + "..."

        return caption

    def _generate_hashtags(self, keywords: List[str], platform: Dict,
                           clip: Dict) -> List[str]:
        """Generate hashtags for a platform."""
        hashtag_count = platform.get("hashtag_count", 5)

        # Base hashtags
        base_tags = {
            "TikTok": ["#fyp", "#foryou", "#viral", "#trending"],
            "YouTube Shorts": ["#shorts", "#youtube", "#viral", "#trending"],
            "Instagram Reels": ["#reels", "#instagram", "#viral", "#explore"],
            "X (Twitter)": ["#viral", "#trending"],
        }

        tags = list(base_tags.get(platform["name"], ["#viral"]))

        # Add keyword-based hashtags
        for kw in keywords:
            if len(tags) >= hashtag_count:
                break
            tag = "#" + kw.replace(" ", "").replace("-", "")
            if tag not in tags and len(tag) > 2:
                tags.append(tag.lower())

        return tags[:hashtag_count]
