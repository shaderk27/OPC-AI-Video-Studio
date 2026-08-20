# -*- coding: utf-8 -*-
"""【占位 STUB · 待 M2/队友替换】Content Analyzer Agent。

仅用于让 main.py 在队友代码到位前能独立跑通。真实实现应由内容分析 Agent 提供
（话题/章节/关键帧/情感等）。这里只给出 main.py 需要的最小字段。
"""
class ContentAnalyzerAgent:
    def __init__(self, config: dict = None):
        self.config = config or {}

    def run(self, pipeline_data: dict) -> dict:
        src = self.config.get("source_language", "en")
        pipeline_data["detected_language"] = src
        pipeline_data["full_transcript"] = ""
        pipeline_data["video_info"] = {"duration": 0, "width": 0, "height": 0}
        ca = {
            "total_segments": 0,
            "topics": [],
            "chapters": [],
            "key_moments": [],
            "summary": "（stub）待内容分析 Agent 实现",
            "sentiment": {"overall": "neutral"},
        }
        pipeline_data["content_analysis"] = ca
        return {
            "content_analysis": ca,
            "full_transcript": "",
            "detected_language": src,
            "video_info": pipeline_data["video_info"],
        }
