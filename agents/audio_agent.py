# -*- coding: utf-8 -*-
"""M3 · 配音 Agent（接入 main.py 流水线）。

接口: AudioAgent(config).run(pipeline_data)
  - 读取每个 clip 的转写文本（subtitle 阶段缓存的 clip["transcript"]）
  - Edge TTS 生成中文/英文等多语言配音，混入（替换原音轨）视频
  - 写回 clip["audio_translations"][lang] = {"video_path", "voiceover_path"}
  - 更新 clip["path"] 为配音版本（与烧录字幕同语言，保证一致）
"""
import os
import asyncio
from utils.media_utils import run, ensure_dir, FFMPEG, get_out_dir, translate_text

# Edge TTS 免费音色（无需密钥）
EDGE_VOICES = {
    "zh": "zh-CN-XiaoxiaoNeural",
    "en": "en-US-AriaNeural",
    "es": "es-ES-ElviraNeural",
    "ja": "ja-JP-NanamiNeural",
    "fr": "fr-FR-DeniseNeural",
    "de": "de-DE-KatjaNeural",
}


class AudioAgent:
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.target_languages = self.config.get("audio_agent", {}).get("target_languages", ["en"])

    # ---------- 底层能力（可单独测试） ----------
    def synthesize(self, text, lang="zh", voice=None, output_path=None):
        """Edge TTS 文本转语音 → mp3。"""
        import edge_tts
        if voice is None:
            voice = EDGE_VOICES.get(lang, EDGE_VOICES["en"])
        if output_path is None:
            output_path = os.path.join("output", f"voiceover_{lang}.mp3")
        ensure_dir(output_path)

        async def _go():
            comm = edge_tts.Communicate(text, voice)
            await comm.save(output_path)

        asyncio.new_event_loop().run_until_complete(_go())
        return output_path

    def mix(self, video_path, audio_path, output_path=None, replace_original=True):
        """把配音混入视频。replace_original=True 用配音替换原音轨。"""
        if output_path is None:
            output_path = os.path.splitext(video_path)[0] + "_vo.mp4"
        ensure_dir(output_path)
        if replace_original:
            # apad 把配音补静音到视频全长，避免视频被截断
            run([FFMPEG, "-y", "-i", video_path, "-i", audio_path,
                 "-filter_complex", "[1:a]apad[a]",
                 "-map", "0:v:0", "-map", "[a]",
                 "-c:v", "copy", "-shortest", output_path])
        else:
            filt = "[1:a]volume=0.8[a1];[0:a][a1]amix=inputs=2[a]"
            run([FFMPEG, "-y", "-i", video_path, "-i", audio_path,
                 "-filter_complex", filt, "-map", "0:v", "-map", "[a]", "-c:v", "copy", output_path])
        return output_path

    # ---------- 流水线入口（main.py 调用） ----------
    def run(self, pipeline_data: dict) -> dict:
        clips = pipeline_data.get("clips", [])
        src_lang = pipeline_data.get("detected_language", self.target_languages[0])
        out_root = get_out_dir(self.config)
        for clip in clips:
            clip_path = clip["path"]
            base = clip.get("filename", os.path.splitext(os.path.basename(clip_path))[0])
            cdir = os.path.join(out_root, "clips", base, "audio")
            ensure_dir(cdir)

            text = clip.get("transcript", "")
            audio_trans = {}
            for lang in self.target_languages:
                voice_text = text
                if lang != src_lang and text:
                    voice_text = translate_text(text, src_lang, lang) or text
                if not voice_text:
                    continue
                mp3 = self.synthesize(voice_text, lang=lang,
                                      output_path=os.path.join(cdir, f"{base}_{lang}.mp3"))
                mixed = self.mix(clip_path, mp3,
                                 output_path=os.path.join(cdir, f"{base}_{lang}_vo.mp4"),
                                 replace_original=True)
                audio_trans[lang] = {"video_path": mixed, "voiceover_path": mp3}

            if audio_trans:
                # clip["path"] 指向源语言配音版（与烧录字幕语言一致）
                clip["path"] = audio_trans.get(src_lang, list(audio_trans.values())[0])["video_path"]
            clip["audio_translations"] = audio_trans
        return {"clips": clips}
