# -*- coding: utf-8 -*-
"""M3 · 格式 Agent（接入 main.py 流水线）。

接口: FormatAgent(config).run(pipeline_data)
  - 把每个 clip 裁切为各平台比例（默认 9:16 / 16:9 / 1:1，居中裁切无黑边）
  - 写回 clip["platform_versions"] = {platform: {"path": ...}}

v2 关键修复：_parse_aspect() 智能解析各种 aspect 写法
  - 字符串 "9:16" / "9x16" / "9/16"  → (9, 16)
  - 整数 916 (PyYAML 把 "9:16" 拼一起解析) → (9, 16)
  - 整数 556 (PyYAML 把 "9:16" 当 sexagesimal 解析) → (9, 16)
  - tuple (9, 16)                        → (9, 16)
"""
import os
from utils.media_utils import run, ensure_dir, get_dimensions, FFMPEG, get_out_dir

DEFAULT_PLATFORMS = {"tiktok": "9:16", "youtube": "16:9", "instagram": "1:1"}


def _parse_aspect(aspect):
    """把任何写法解析成 (width, height) 整数元组。"""
    # 1) tuple/list
    if isinstance(aspect, (tuple, list)) and len(aspect) == 2:
        return int(aspect[0]), int(aspect[1])
    # 2) 字符串
    if isinstance(aspect, str):
        s = aspect.replace("x", ":").replace("/", ":").strip()
        # 处理被 PyYAML 解析后再 stringify 的情况（"916"）
        if ":" not in s and s.isdigit() and len(s) in (3, 4):
            mid = len(s) // 2
            return int(s[:mid]), int(s[mid:])
        tw, th = s.split(":")
        return int(tw), int(th)
    # 3) 整数（PyYAML 把 "9:16" 解析成 916 或 556）
    if isinstance(aspect, (int, float)):
        a = int(aspect)
        # PyYAML sexagesimal: 9:16 → 9*60+16 = 556（如果 y < 60）
        if a % 100 < 60 and a // 100 < 60 and a > 100:
            return a // 100, a % 100
        # 简单拼接: 916 = 9*100 + 16 或 9*16
        s = str(a)
        if len(s) == 4:
            # 优先尝试 4 位拆成 2+2
            return int(s[:2]), int(s[2:])
        if len(s) == 3:
            # 916 → 9, 16
            return int(s[0]), int(s[1:])
        # 实在解析不出来，强行拆
        return a // 100, a % 100
    raise ValueError(f"无法解析 aspect: {aspect!r}")


class FormatAgent:
    def __init__(self, config: dict = None):
        self.config = config or {}
        fmt_cfg = self.config.get("format_agent", {})
        self.platforms = fmt_cfg.get("platforms", DEFAULT_PLATFORMS)

    def convert(self, video_path, aspect, output_path=None):
        """用 ffmpeg 把视频裁切为目标比例（居中裁切）。"""
        tw, th = _parse_aspect(aspect)
        target_ratio = tw / th
        iw, ih = get_dimensions(video_path)
        video_ratio = iw / ih
        if video_ratio > target_ratio:
            cw, ch = int(ih * target_ratio), ih
        else:
            cw, ch = iw, int(iw / target_ratio)
        x = (iw - cw) // 2
        y = (ih - ch) // 2
        # 保证宽高都是偶数（libx264 要求 yuv420p）
        cw = cw - (cw % 2)
        ch = ch - (ch % 2)
        if output_path is None:
            output_path = os.path.splitext(video_path)[0] + f"_{tw}x{th}.mp4"
        ensure_dir(output_path)
        run([FFMPEG, "-y", "-i", video_path,
             "-vf", f"crop={cw}:{ch}:{x}:{y}", "-c:a", "copy", output_path])
        return output_path

    def run(self, pipeline_data: dict) -> dict:
        clips = pipeline_data.get("clips", [])
        out_root = get_out_dir(self.config)
        for clip in clips:
            base = clip.get("filename", os.path.splitext(os.path.basename(clip["path"]))[0])
            cdir = os.path.join(out_root, "clips", base, "format")
            ensure_dir(cdir)
            pvs = {}
            for p, aspect in self.platforms.items():
                out = os.path.join(cdir, f"{base}_{p}.mp4")
                try:
                    self.convert(clip["path"], aspect, out)
                    pvs[p] = {"path": out}
                except Exception as e:
                    print(f"[format_agent] convert {p} aspect={aspect!r} failed: {e}")
                    pvs[p] = {"path": None, "error": str(e)}
            clip["platform_versions"] = pvs
        return {"clips": clips}
