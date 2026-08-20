# -*- coding: utf-8 -*-
"""【占位 STUB · 待 M2 替换】Clip / Highlight Cutter Agent。

仅用于让 main.py 在队友代码到位前能独立跑通。真实实现应由剪辑 Agent 提供
（基于内容分析的亮点检测）。这里把输入视频切成若干 15-60s 的等长片段，
以满足 M3 的"时长 15-60s"指标要求。
"""
import os
from utils.media_utils import (
    run, ensure_dir, get_duration, get_dimensions, get_out_dir, FFMPEG
)


class ClipAgent:
    def __init__(self, config: dict = None):
        self.config = config or {}

    def run(self, pipeline_data: dict) -> dict:
        video = pipeline_data["video_path"]
        max_clips = self.config.get("clip_agent", {}).get("max_clips", 3)
        dur = get_duration(video)
        try:
            w, h = get_dimensions(video)
        except Exception:
            w, h = 0, 0

        # 每段目标长度：至少 15s、不超过 60s、尽量均分
        seg = max(15.0, min(60.0, dur / max(1, max_clips)))
        n = min(max_clips, max(1, int(dur // seg)))

        out_dir = os.path.join(get_out_dir(self.config), "clips")
        ensure_dir(out_dir)
        clips = []
        for i in range(n):
            start = i * seg
            length = min(seg, dur - start)
            if length < 5:
                break
            # 用 f-string 直接嵌进去，不再依赖 fname 变量
            out = os.path.abspath(os.path.join(out_dir, f"clip_{i + 1:02d}.mp4"))
            ensure_dir(out)  # 保险，确保父目录存在

            # 用 input-seek（-ss 在 -i 前）+ copy；start=0 时不加 -ss
            try:
                if start > 0:
                    run([FFMPEG, "-y", "-ss", f"{start:.3f}", "-i", video,
                         "-t", f"{length:.3f}", "-c", "copy", out])
                else:
                    run([FFMPEG, "-y", "-i", video, "-t", f"{length:.3f}",
                         "-c", "copy", out])
            except Exception:
                # fallback: 重编码（保证成功，唯一慢一点）
                if start > 0:
                    run([FFMPEG, "-y", "-ss", f"{start:.3f}", "-i", video,
                         "-t", f"{length:.3f}",
                         "-c:v", "libx264", "-preset", "fast", "-c:a", "aac",
                         "-movflags", "+faststart", out])
                else:
                    run([FFMPEG, "-y", "-i", video, "-t", f"{length:.3f}",
                         "-c:v", "libx264", "-preset", "fast", "-c:a", "aac",
                         "-movflags", "+faststart", out])
            clips.append({
                "path": out,
                "filename": f"clip_{i + 1:02d}",
                "start": start,
                "end": start + length,
                "duration": length,
                "title": f"Clip {i + 1}",
                "reason": "(stub) equal-length split",
                "detected_language": pipeline_data.get("detected_language", "en"),
            })

        video_info = {
            "duration": dur,
            "width": w,
            "height": h,
            "source": os.path.abspath(video),
        }
        pipeline_data["clips"] = clips
        pipeline_data["video_info"] = video_info
        return {"clips": clips, "video_info": video_info}
