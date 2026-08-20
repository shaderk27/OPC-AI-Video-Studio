# -*- coding: utf-8 -*-
"""M3 · Day 1 必做：在中国网络下预下载 Whisper 模型，避免演示时首跑联网卡死。

IMPROVEMENT_PLAN.md 第 7 节风险项：
    "Whisper model slow on first run → Pre-download model before demo"

用法（在 team_project 目录下，Windows PowerShell 也可）：
    python scripts/setup_whisper_model.py            # 默认下载 base
    python scripts/setup_whisper_model.py small      # 指定型号

模型会被缓存到：
    Windows: C:\\Users\\<你的用户名>\\.cache\\whisper\\
    Linux:   ~/.cache/whisper/
之后 subtitle_agent.transcribe() 会直接复用缓存，不再联网。
"""
import os
import sys

# 中国大陆网络：让 torch/whisper 从镜像拉取权重
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


def main():
    size = sys.argv[1] if len(sys.argv) > 1 else "base"
    print(f"[M3] 开始预下载 Whisper 模型: {size} (走 hf-mirror 镜像)")
    try:
        import whisper
        whisper.load_model(size)
        print(f"[M3] 模型 {size} 已就绪，缓存在 {whisper._MODELS_DIR}")
    except Exception as e:
        print("[M3] 自动下载失败，请手动方案：")
        print("  1) 到镜像站下载对应 .pt 文件（如 base.pt）")
        print("     镜像示例: https://hf-mirror.com/ 搜索 openai/whisper 资源，")
        print("     或直接百度 'whisper base.pt 国内镜像' 获取下载链接")
        print("  2) 把 .pt 放进缓存目录:")
        print("     Windows: C:\\Users\\<用户名>\\.cache\\whisper\\")
        print("     Linux:   ~/.cache/whisper/")
        print(f"  原始错误: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
