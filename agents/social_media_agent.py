# -*- coding: utf-8 -*-
"""【占位 STUB · 待 M2/队友替换】Social Media Copywriter Agent。

仅用于让 main.py 在队友代码到位前能独立跑通。真实实现应由社媒文案 Agent 提供。
"""
class SocialMediaAgent:
    def __init__(self, config: dict = None):
        self.config = config or {}

    def run(self, pipeline_data: dict) -> dict:
        for clip in pipeline_data.get("clips", []):
            clip["social_media"] = {"copy_file": f"social_{clip.get('filename', 'clip')}.txt"}
        return {"clips": pipeline_data.get("clips", [])}
