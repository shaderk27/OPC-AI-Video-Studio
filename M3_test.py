# -*- coding: utf-8 -*-
"""M3 独立自测：不依赖队友，单独验证 4 个媒体 Agent（接口与 main.py 一致）。

用法（在 team_project 目录下，终端里）:
    py M3_test.py

会用 input/ 下的 .mp4（最多 3 条）跑完整媒体流水线（字幕→配音→封面→格式），
并打印每个 clip 的产物，验证 M3 负责的 3 条硬指标。
"""
import os
import sys
import time
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.media_utils import get_duration
from agents.subtitle_agent import SubtitleAgent
from agents.audio_agent import AudioAgent
from agents.thumbnail_agent import ThumbnailAgent
from agents.format_agent import FormatAgent

INPUT_DIR = "input"
MAX_CLIPS = 3

CONFIG = {
    "subtitle_agent": {"whisper_model": "base", "target_languages": ["en", "zh"]},
    "audio_agent": {"target_languages": ["en", "zh"]},
    "format_agent": {"platforms": {"tiktok": "9:16", "youtube": "16:9", "instagram": "1:1"}},
    "paths": {"output_dir": "output"},
    "source_language": "en",
}


def main():
    vids = sorted(glob.glob(os.path.join(INPUT_DIR, "*.mp4")))[:MAX_CLIPS]
    if not vids:
        print(f"[M3] 未找到测试视频，请把 .mp4 放到 {INPUT_DIR}/ 目录")
        return

    # 构造与 main.py 一致的 pipeline_data（单视频模拟多 clip）
    clips = []
    for i, v in enumerate(vids, 1):
        clips.append({
            "path": os.path.abspath(v),
            "filename": f"clip_{i:02d}",
            "start": 0, "end": 0, "duration": 0,
            "title": f"Clip {i}",
            "detected_language": "en",
        })
    pd = {"clips": clips, "detected_language": "en", "video_name": "test"}

    t = time.time()
    print("[M3] 字幕...");  pd["clips"] = SubtitleAgent(CONFIG).run(pd)["clips"]
    print("[M3] 配音...");  pd["clips"] = AudioAgent(CONFIG).run(pd)["clips"]
    print("[M3] 封面...");  pd["clips"] = ThumbnailAgent(CONFIG).run(pd)["clips"]
    print("[M3] 格式...");  pd["clips"] = FormatAgent(CONFIG).run(pd)["clips"]
    cost = time.time() - t

    print("\n==== 产物清单 ====")
    all_ok = True
    for c in pd["clips"]:
        dur = get_duration(c["path"])
        has_sub = bool(c.get("subtitles"))
        in_range = 15 <= dur <= 60
        all_ok = all_ok and has_sub and in_range
        print(f"  {c['filename']}:")
        print(f"    字幕语言 = {list(c.get('subtitles', {}).keys())}")
        print(f"    配音语言 = {list(c.get('audio_translations', {}).keys())}")
        print(f"    封面数   = {len(c.get('thumbnails', []))}")
        print(f"    格式     = {list(c.get('platform_versions', {}).keys())}")
        print(f"    成片时长 = {dur:.1f}s  (15-60区间:{in_range})")

    print(f"\n[M3] 总耗时 {cost:.1f}s（3 条需 ≤180s）")
    print(f"[M3] 验收: {'全部通过 ✅' if all_ok and cost <= 180 else '存在不达标项 ❌'}")


if __name__ == "__main__":
    main()
