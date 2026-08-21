import unittest

from agents.social_media_agent import SocialMediaAgent


class SocialMediaChinesePlatformTests(unittest.TestCase):
    def setUp(self):
        self.agent = SocialMediaAgent({})
        self.clip = {
            "title": "深圳湾日落太美了",
            "hook_score": 72,
            "detected_language": "zh",
        }
        self.text = "今天来到深圳湾看日落，深圳湾真的很美。人才公园也很适合拍照。"

    def test_extracts_meaningful_chinese_phrases(self):
        keywords = self.agent._extract_keywords(self.text)

        self.assertIn("深圳湾", keywords)
        self.assertIn("人才公园", keywords)
        self.assertIn("日落", keywords)
        self.assertNotIn("圳湾", keywords)
        self.assertNotIn("湾看日落", keywords)
        self.assertNotIn("今天来到", keywords)

    def test_prefers_detected_chinese_subtitle(self):
        clip = {
            "detected_language": "zh",
            "subtitles": {
                "en": {"segments": [{"text": "English translation"}]},
                "zh": {"segments": [{"text": "深圳湾日落"}]},
            },
        }

        self.assertEqual(self.agent._get_transcript_text(clip), "深圳湾日落")

    def test_douyin_and_xiaohongshu_have_distinct_caption_styles(self):
        keywords = ["深圳湾", "人才公园", "日落"]
        douyin = self.agent._generate_caption(
            self.clip, self.text, keywords,
            {"name": "Douyin", "max_caption_length": 150},
        )
        xiaohongshu = self.agent._generate_caption(
            self.clip, self.text, keywords,
            {"name": "Xiaohongshu", "max_caption_length": 1000},
        )

        self.assertIn("你会来打卡吗", douyin)
        self.assertIn("📍深圳湾", xiaohongshu)
        self.assertIn("收藏", xiaohongshu)
        self.assertNotEqual(douyin, xiaohongshu)

    def test_chinese_platform_hashtags_prioritize_content_keywords(self):
        tags = self.agent._generate_hashtags(
            ["深圳湾", "人才公园", "日落"],
            {"name": "Xiaohongshu", "hashtag_count": 5},
            self.clip,
        )

        self.assertEqual(tags[:3], ["#深圳湾", "#人才公园", "#日落"])
        self.assertEqual(len(tags), 5)


if __name__ == "__main__":
    unittest.main()
