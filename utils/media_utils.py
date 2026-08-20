# -*- coding: utf-8 -*-
"""M3 共享媒体工具：自动检测 ffmpeg（系统 PATH / imageio-ffmpeg 内置）。

无需手动安装 ffmpeg、配置 PATH。只要 `pip install imageio-ffmpeg` 即可。
中国大陆网络：建议用清华 PyPI 镜像 `pip install -i https://pypi.tuna.tsinghua.edu.cn/simple ...`
"""
import os
import re
import shutil
import subprocess
import logging

logger = logging.getLogger(__name__)


# 智能查找 ffmpeg
def _find_ffmpeg() -> str:
    """优先系统 PATH 上的 ffmpeg；否则退回 imageio-ffmpeg 自带二进制。"""
    path = shutil.which("ffmpeg")
    if path:
        return path
    try:
        import imageio_ffmpeg  # 随 pip 安装自带的 ffmpeg 二进制
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.exists(exe):
            return exe
    except ImportError:
        pass
    raise RuntimeError(
        "ffmpeg 不可用。请运行: pip install imageio-ffmpeg  （推荐加 -i 国内 PyPI 镜像）"
    )


try:
    FFMPEG = _find_ffmpeg()
    # imageio-ffmpeg 不带 ffprobe；统一用 ffmpeg -i 解析元数据
    FFPROBE = FFMPEG
except RuntimeError as _e:
    logger.warning("[M3 utils] %s", _e)
    FFMPEG = "ffmpeg"
    FFPROBE = "ffmpeg"


def ensure_dir(path: str) -> str:
    """确保文件所在目录存在，返回原路径。"""
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    return path


def run(cmd):
    """执行命令（cmd 为列表），失败抛异常。失败时把 stderr 也输出。"""
    import sys
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=os.getcwd())
    if proc.returncode != 0:
        sys.stderr.write(f"[ffmpeg cmd failed] rc={proc.returncode}\n")
        sys.stderr.write("cmd: " + " ".join(cmd) + "\n")
        if proc.stderr:
            try:
                sys.stderr.write(proc.stderr.decode("utf-8", errors="replace"))
            except Exception:
                sys.stderr.write(str(proc.stderr))
            if not proc.stderr.endswith(b"\n"):
                sys.stderr.write("\n")
        raise subprocess.CalledProcessError(proc.returncode, cmd)


def get_out_dir(config: dict) -> str:
    """从配置取输出根目录。"""
    return config.get("paths", {}).get("output_dir", "output")


def translate_text(text: str, src: str, tgt: str):
    """简单翻译（deep-translator / MyMemory）。失败返回 None（调用方自行降级）。"""
    if not text or not src or not tgt or src == tgt:
        return None
    try:
        from deep_translator import MyMemoryTranslator
        return MyMemoryTranslator(source=src, target=tgt).translate(text[:5000])
    except Exception:
        return None


def _ffmpeg_metadata(path: str) -> dict:
    """用 ffmpeg -i 解析元数据（无 ffprobe 时的替代方案）。"""
    proc = subprocess.run(
        [FFMPEG, "-hide_banner", "-i", path],
        capture_output=True, text=True
    )
    err = proc.stderr or ""

    # Duration: HH:MM:SS.xx
    duration = 0.0
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.?\d*)", err)
    if m:
        duration = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))

    # 第一个视频流: "Stream #0:0 ... Video: ... 1920x1080 ..."
    width = height = 0
    m = re.search(r"Stream #\d+:\d+.*?Video:.*?(\d{2,5})x(\d{2,5})", err)
    if m:
        width, height = int(m.group(1)), int(m.group(2))

    return {"duration": duration, "width": width, "height": height}


def get_duration(path: str) -> float:
    return _ffmpeg_metadata(path)["duration"]


def get_dimensions(path: str):
    """返回 (width, height)。"""
    meta = _ffmpeg_metadata(path)
    if meta["width"] == 0 or meta["height"] == 0:
        raise RuntimeError("未找到视频流: " + path)
    return meta["width"], meta["height"]


def _esc_sub_path(srt_path: str) -> str:
    """跨平台 ffmpeg subtitles 滤镜路径转义。"""
    p = srt_path.replace("\\", "/")
    p = p.replace(":", "\\:").replace("'", "\\'")
    return f"subtitles='{p}'"
