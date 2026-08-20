# -*- coding: utf-8 -*-
"""M3 · 封面 Agent（接入 main.py 流水线）。

接口: ThumbnailAgent(config).run(pipeline_data)
  - 从每个 clip 抽一帧做封面（默认取 30% 处，画面较稳）
  - 写回 clip["thumbnails"] = [{"path": ...}]
"""
import os
import datetime
from utils.media_utils import run, ensure_dir, get_duration, FFMPEG, get_out_dir


class ThumbnailAgent:
    def __init__(self, config: dict = None):
        self.config = config or {}

    def extract(self, video_path, timestamp=None, output_path=None):
        """抽取视频某一帧作为封面图。"""
        if timestamp is None:
            dur = get_duration(video_path)
            timestamp = max(0.0, dur * 0.3)
        if output_path is None:
            output_path = os.path.splitext(video_path)[0] + "_thumb.jpg"
        ensure_dir(output_path)
        ts = str(datetime.timedelta(seconds=timestamp))
        run([FFMPEG, "-y", "-ss", ts, "-i", video_path, "-frames:v", "1", "-q:v", "2", output_path])
        return output_path

    def run(self, pipeline_data: dict) -> dict:
        clips = pipeline_data.get("clips", [])
        out_root = get_out_dir(self.config)
        for clip in clips:
            base = clip.get("filename", os.path.splitext(os.path.basename(clip["path"]))[0])
            cdir = os.path.join(out_root, "clips", base, "thumbnail")
            ensure_dir(cdir)
            out = os.path.join(cdir, base + "_cover.jpg")
            self.extract(clip["path"], output_path=out)
            clip.setdefault("thumbnails", []).append({"path": out})
        return {"clips": clips}
