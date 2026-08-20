# -*- coding: utf-8 -*-
"""【补丁 v2】完整接管 whisper.audio.load_audio。

彻底绕开 Whisper 内部脆弱的 ffmpeg 子进程调用链：
  1. 用 imageio_ffmpeg 的 ffmpeg.exe（绝对路径）把任意音频转 wav
  2. 用 Python 标准库 `wave` 读 wav 到 numpy 数组
  3. 返回给 Whisper，Whisper 完全不感知
"""
import sys
import os
import subprocess
import tempfile
import wave

import numpy as np

import whisper
import whisper.audio


def _patched_load_audio(file, sr=16000):
    if not os.path.exists(file):
        raise FileNotFoundError(f"audio file not found: {file}")

    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        import shutil
        ffmpeg_exe = shutil.which("ffmpeg")
        if not ffmpeg_exe:
            raise FileNotFoundError(
                "ffmpeg not found. 请安装 imageio_ffmpeg，或在 PATH 里放 ffmpeg.exe。"
            )

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        wav_path = tf.name

    try:
        result = subprocess.run(
            [
                ffmpeg_exe, "-y", "-loglevel", "warning",
                "-i", file,
                "-ar", str(sr),
                "-ac", "1",
                "-f", "wav",
                wav_path,
            ],
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            err = (result.stderr or b"").decode("utf-8", errors="replace")
            sys.stderr.write(
                f"\n[whisper_patch] ffmpeg 转 wav 失败, exit={result.returncode}\n{err[:2000]}\n"
            )
            sys.stderr.flush()
            raise RuntimeError(f"ffmpeg 转 wav 失败: {err}")

        with wave.open(wav_path, "rb") as wf:
            n_frames = wf.getnframes()
            audio_bytes = wf.readframes(n_frames)
            audio = np.frombuffer(audio_bytes, dtype=np.int16).flatten().astype(np.float32) / 32768.0
        return audio
    finally:
        try:
            os.remove(wav_path)
        except Exception:
            pass


whisper.audio.load_audio = _patched_load_audio
whisper.load_audio = _patched_load_audio
