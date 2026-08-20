# -*- coding: utf-8 -*-
"""【占位 STUB · 待 M1/M2 替换】Subtitle Agent。

Whisper 转录 + 写 SRT/VTT + 烧字幕到视频。配合 utils/whisper_patch.py 使用。

关键修复（v3）：
  * 永远用 os.makedirs(..., exist_ok=True) 创建父目录（不依赖 ensure_dir import）
  * 所有路径转 os.path.abspath 绝对路径
  * burn_in 失败不阻塞其他 clip
  * subs[lang] 返 dict 格式给 main.py
"""
import os
import shutil

import whisper
from utils.media_utils import (
    run, ensure_dir, get_out_dir, FFMPEG
)


def _srt_time(t: float) -> str:
    if t is None or t < 0:
        t = 0
    hours = int(t // 3600)
    minutes = int((t % 3600) // 60)
    seconds = int(t % 60)
    millis = int((t - int(t)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


class SubtitleAgent:
    def __init__(self, config: dict = None):
        self.config = config or {}
        sa = self.config.get("subtitle_agent", {})
        self.model_size = sa.get("model_size", "base")
        self.languages = sa.get("languages", ["en"])
        self._model = None
        self._loaded_size = None

    def _load_model(self):
        if self._model is None or self._loaded_size != self.model_size:
            self._model = whisper.load_model(self.model_size)
            self._loaded_size = self.model_size
        return self._model

    def transcribe(self, clip_path: str, language=None):
        model = self._load_model()
        return model.transcribe(clip_path, language=language, verbose=False)

    @staticmethod
    def _to_srt(segments) -> str:
        lines = []
        for i, seg in enumerate(segments, 1):
            start = _srt_time(seg.get("start", 0))
            end = _srt_time(seg.get("end", 0))
            text = (seg.get("text") or "").strip()
            if not text:
                continue
            lines.append(str(i))
            lines.append(f"{start} --> {end}")
            lines.append(text)
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def burn_in(clip_path: str, srt_path: str, output_path: str):
        clip_path = os.path.abspath(clip_path)
        srt_path = os.path.abspath(srt_path)
        output_path = os.path.abspath(output_path)

        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)

        tmp_srt = os.path.join(output_dir, "tmp_subs.srt")
        shutil.copy(srt_path, tmp_srt)

        import subprocess
        # 关键修复：显式指定 ffmpeg 子进程的 cwd=output_dir，
        # 这样 -vf subtitles=tmp_subs.srt 用相对文件名就能找到 srt 文件
        proc = subprocess.run(
            [
                FFMPEG, "-y",
                "-i", clip_path,
                "-vf", f"subtitles={os.path.basename(tmp_srt)}",
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-c:a", "copy",
                output_path,
            ],
            cwd=output_dir,
            capture_output=True,
        )
        if proc.returncode != 0:
            err = (proc.stderr or b"").decode("utf-8", errors="replace")
            print(f"[burn_in] ffmpeg stderr: {err[-2000:]}")
            raise subprocess.CalledProcessError(proc.returncode, "ffmpeg burn_in", output=None, stderr=err.encode("utf-8") if isinstance(err, str) else err)


        try:
            os.remove(tmp_srt)
        except Exception:
            pass

    def run(self, pipeline_data: dict) -> dict:
        clips = pipeline_data.get("clips", [])
        if not clips:
            pipeline_data["subtitles"] = {}
            return {"subtitles": {}}

        out_dir = get_out_dir(self.config)
        languages = self.languages
        detected = pipeline_data.get("detected_language", "en")
        src_lang = detected if detected in languages else (languages[0] if languages else "auto")

        subs = {}
        for clip in clips:
            clip_path = clip["path"]
            clip_id = clip["filename"]
            clip_name = os.path.splitext(os.path.basename(clip_path))[0]

            clip_sub_dir = os.path.abspath(os.path.join(out_dir, "clips", clip_id, "subtitle"))
            os.makedirs(clip_sub_dir, exist_ok=True)
            try:
                ensure_dir(clip_sub_dir)
            except Exception:
                pass

            try:
                tr = self.transcribe(clip_path, language=src_lang if src_lang != "auto" else None)
                segments = tr.get("segments", []) if isinstance(tr, dict) else []
            except Exception as e:
                print(f"[subtitle_agent] transcribe failed for {clip_id}: {e}")
                segments = []

            srt_path = os.path.abspath(os.path.join(clip_sub_dir, f"{clip_name}.srt"))
            srt_content = self._to_srt(segments)
            with open(srt_path, "w", encoding="utf-8") as f:
                f.write(srt_content)

            vtt_path = os.path.abspath(os.path.join(clip_sub_dir, f"{clip_name}.vtt"))
            vtt_content = "WEBVTT\n\n" + srt_content.replace(",", ".")
            with open(vtt_path, "w", encoding="utf-8") as f:
                f.write(vtt_content)

            burned_path = os.path.abspath(os.path.join(clip_sub_dir, f"{clip_name}_burned.mp4"))
            try:
                self.burn_in(clip_path, srt_path, burned_path)
            except Exception as e:
                print(f"[subtitle_agent] burn_in failed for {clip_id}: {e}")
                burned_path = None

            subs[src_lang] = {
                "srt": {"path": srt_path, "content": srt_content, "language": src_lang},
                "vtt": {"path": vtt_path},
                "burned": {"path": burned_path},
            }

        pipeline_data["subtitles"] = subs
        pipeline_data["clips"] = clips  # forward 给下游，主调用需要 result["clips"]
        return {"subtitles": subs, "clips": clips}
