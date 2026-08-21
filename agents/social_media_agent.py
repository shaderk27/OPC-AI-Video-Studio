"""
Social Media Agent (Enhancement)
=================================
Role: Social Media Copywriter & Content Strategist

Writes platform-specific captions, hashtags, and descriptions for each clip.
Generates tailored copy for TikTok, YouTube Shorts, Instagram Reels, X,
Douyin, and Xiaohongshu.
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
        detected_language = str(clip.get("detected_language", "")).lower()
        preferred_languages = []
        if detected_language:
            preferred_languages.append(detected_language)
            if detected_language.startswith("zh") or detected_language == "chinese":
                preferred_languages.extend(["zh", "zh-cn", "chinese"])

        for language in preferred_languages:
            lang_data = subtitles.get(language, {})
            segments = lang_data.get("segments", [])
            if segments:
                return " ".join(seg["text"] for seg in segments)

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
        english_keywords = [word for word, _ in counter.most_common(10)]
        chinese_keywords = self._extract_chinese_keywords(text)
        return (chinese_keywords + english_keywords)[:10]

    def _extract_chinese_keywords(self, text: str) -> List[str]:
        """Extract useful Chinese phrases without requiring online NLP tools."""
        sequences = re.findall(r"[\u4e00-\u9fff]+", text)
        if not sequences:
            return []

        stop_phrases = {
            "一个", "一些", "这个", "那个", "我们", "你们", "他们",
            "今天", "现在", "然后", "就是", "真的", "可以", "因为",
            "所以", "还是", "没有", "什么", "非常", "已经", "自己",
        }
        leading_particles = set(
            "的了是在我你他她它们这那有就也又才只会要想说做看去来给对到湾山街路"
        )
        useful_suffixes = (
            "公园", "景区", "地铁站", "海滩", "海边", "古城", "商场",
            "博物馆", "美术馆", "咖啡店", "餐厅", "夜市", "大厦",
            "广场", "酒店", "攻略", "日落", "湾", "山", "街", "路",
        )

        counts = Counter()
        first_position = {}
        position = 0
        for sequence in sequences:
            for length in range(2, min(4, len(sequence)) + 1):
                for start in range(len(sequence) - length + 1):
                    phrase = sequence[start:start + length]
                    if phrase in stop_phrases or phrase[0] in leading_particles:
                        continue
                    counts[phrase] += 1
                    first_position.setdefault(phrase, position)
                    position += 1

        ranked = sorted(
            counts,
            key=lambda phrase: (
                -(counts[phrase] * 10 +
                  (5 if phrase.endswith(useful_suffixes) else 0) +
                  len(phrase) / 10),
                first_position[phrase],
            ),
        )

        selected = []
        for phrase in ranked:
            high_confidence = (
                counts[phrase] >= 2 or phrase.endswith(useful_suffixes)
            )
            if not high_confidence:
                continue
            if any(phrase in existing or existing in phrase for existing in selected):
                continue
            selected.append(phrase)
            if len(selected) >= 10:
                break
        return selected

    def _generate_caption(self, clip: Dict, transcript: str,
                          keywords: List[str], platform: Dict) -> str:
        """Generate a caption for a specific platform."""
        title = clip.get("title", "Amazing moment")
        max_len = platform.get("max_caption_length", 200)
        platform_name = platform["name"]

        if platform_name == "Douyin":
            caption = self._generate_douyin_caption(clip, title, keywords)
            return self._truncate_caption(caption, max_len)
        if platform_name == "Xiaohongshu":
            caption = self._generate_xiaohongshu_caption(clip, title, keywords)
            return self._truncate_caption(caption, max_len)

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

        return self._truncate_caption(caption, max_len)

    def _generate_douyin_caption(self, clip: Dict, title: str,
                                 keywords: List[str]) -> str:
        """Short, immediate, conversational copy for Douyin."""
        hook_score = float(clip.get("hook_score", 0))
        if hook_score >= 70:
            hook = "没想到，最精彩的是这一刻！"
        elif hook_score >= 50:
            hook = "这一幕值得看到最后！"
        else:
            hook = "现场这一刻，真的很难忘。"
        focus = "、".join(keywords[:2])
        detail = f"重点是{focus}。" if focus else "把这一刻记录下来了。"
        return f"{hook}{title}\n{detail}你会来打卡吗？"

    def _generate_xiaohongshu_caption(self, clip: Dict, title: str,
                                      keywords: List[str]) -> str:
        """Searchable, experience-led copy for Xiaohongshu."""
        location = next(
            (keyword for keyword in keywords
             if keyword.endswith(("湾", "公园", "景区", "街", "路", "山"))),
            keywords[0] if keywords else "旅行记录",
        )
        focus = "、".join(keywords[:3]) if keywords else "现场体验"
        return (
            f"📍{location}｜{title}\n"
            f"这段最值得记录的是{focus}。现场的氛围比镜头里更有感觉，"
            "建议把这段行程留出一点时间慢慢体验。\n"
            "如果你也准备去，可以先收藏这条。"
        )

    @staticmethod
    def _truncate_caption(caption: str, max_len: int) -> str:
        if len(caption) > max_len:
            return caption[:max_len - 3].rstrip() + "..."
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
            "Douyin": ["#旅行", "#城市漫游", "#旅行vlog"],
            "Xiaohongshu": ["#旅行攻略", "#周末去哪儿", "#城市漫游"],
        }

        platform_name = platform["name"]
        tags = [] if platform_name in ("Douyin", "Xiaohongshu") else \
            list(base_tags.get(platform_name, ["#viral"]))

        # Add keyword-based hashtags
        for kw in keywords:
            if len(tags) >= hashtag_count:
                break
            cleaned = re.sub(r"[^A-Za-z0-9\u4e00-\u9fff]", "", kw)
            if len(cleaned) < 2:
                continue
            tag = "#" + cleaned.lower()
            if tag not in tags:
                tags.append(tag)

        for tag in base_tags.get(platform_name, []):
            if len(tags) >= hashtag_count:
                break
            if tag not in tags:
                tags.append(tag)

        return tags[:hashtag_count]
