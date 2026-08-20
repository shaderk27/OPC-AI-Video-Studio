# -*- coding: utf-8 -*-
"""【占位 STUB · 待 M5 替换】Quality Assurance Reviewer Agent。

仅用于让 main.py 在队友代码到位前能独立跑通。真实实现应由 QA Agent 提供
（WER、hook score、时长合规等度量）。
"""
class QualityAgent:
    def __init__(self, config: dict = None):
        self.config = config or {}

    def run(self, pipeline_data: dict) -> dict:
        clips = pipeline_data.get("clips", [])
        return {"quality_report": {
            "summary": {
                "clips_passed": len(clips),
                "clips_failed": 0,
                "total_issues": 0,
            }
        }}
