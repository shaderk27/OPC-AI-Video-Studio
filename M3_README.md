# M3 · Media Agent Engineer — 交付说明（已接入 main.py）

**负责人**：Member 3（你）
**职责**：4 个媒体 Agent（字幕 / 配音 / 封面 / 格式），接入 M1 的 `main.py` 8-agent 流水线。
**硬指标（你背锅的 3 条，见 IM 第 6 节）**：① 100% 成片带字幕；② 100% 时长 15–60s；③ 3 条片子渲染 ≤ 3 分钟。

---

## 一、本目录即「团队工程」结构

```
team_project/
├── main.py                      # M1 的编排器（已就绪，你不用改）
├── config.yaml                  # 全局配置（你关心的字段在下方）
├── requirements.txt             # 团队依赖（含 M3 的 whisper/edge-tts/...）
├── agents/
│   ├── subtitle_agent.py        # 【M3 真实】Whisper 转写 + 烧录 + 多语言
│   ├── audio_agent.py           # 【M3 真实】Edge TTS 配音 + 混音
│   ├── thumbnail_agent.py       # 【M3 真实】抽帧封面
│   ├── format_agent.py          # 【M3 真实】9:16 / 16:9 / 1:1 裁切
│   ├── content_analyzer_agent.py# 【STUB 占位】待 M2/队友替换
│   ├── clip_agent.py            # 【STUB 占位】切成 15–60s 片段
│   ├── social_media_agent.py    # 【STUB 占位】待队友替换
│   └── quality_agent.py         # 【STUB 占位】待 M5 替换
├── utils/
│   └── media_utils.py           # M3 共享工具（ffmpeg 自动检测 + 元数据 + 翻译）
├── scripts/setup_whisper_model.py  # 中国网络预下载 Whisper 模型
├── input/                      # 放测试视频 *.mp4
└── output/                     # 流水线产物
```

> ⚠️ 4 个 STUB 只是**让你现在就能独立跑通** `python main.py`。等队友把正式 Agent 发来后，
> **用他们的覆盖掉这 4 个 stub 文件即可**；你的 4 个真实 M3 agent 不动。

---

## 二、接口约定（你与 M1 的对齐结果）

每个 Agent 都长这样，由 `main.py` 统一调用：

```python
agent = SubtitleAgent(config)        # config = 整个 config.yaml 加载后的 dict
result = agent.run(pipeline_data)    # pipeline_data["clips"] 是 clip 字典列表
pipeline_data["clips"] = result["clips"]
```

Agent 之间通过 `pipeline_data["clips"]` 互传；M3 负责往每个 clip 上挂这些数据：

| Agent | 写入字段 | 含义 |
|-------|---------|------|
| subtitle | `clip["subtitles"][lang]` = `{"formats":{srt,vtt,burned}, "segments":[...]}` | 字幕（≥1 语言） |
|          | `clip["transcript"]` | 转写全文，供 Audio 复用 |
|          | `clip["path"]` | 更新为烧录后视频 |
| audio    | `clip["audio_translations"][lang]` = `{"video_path","voiceover_path"}` | 多语言配音 |
| thumbnail| `clip["thumbnails"]` = `[{"path":...}]` | 封面图 |
| format   | `clip["platform_versions"]` = `{tiktok/youtube/instagram:{"path":...}}` | 多平台格式 |

---

## 三、你（M3）负责的 3 条验收硬指标

| 指标 | 判定位置 | 说明 |
|---|---|---|
| 100% 成片带字幕 | `clip["subtitles"]` 非空 | subtitle 阶段保证每个 clip 至少 1 语言 |
| 时长 100% 在 15–60s | `clip["duration"]` | 由 clip_agent 切 15–60s 片段保证（见 config `clip_agent.max_clips`） |
| 3 条渲染 ≤ 3 分钟 | `M3_test.py` 计时 | whisper 用 base/tiny + 复用预下载模型提速 |

---

## 四、你的 config.yaml 字段（可改）

```yaml
subtitle_agent:
  whisper_model: base          # 演示追求速度 → 改 tiny
  target_languages: [en, zh]   # 字幕语言
audio_agent:
  target_languages: [en, zh]   # 配音语言
format_agent:
  platforms: {tiktok: 9:16, youtube: 16:9, instagram: 1:1}
paths:
  output_dir: output
source_language: en            # 中文素材改 "zh"
```

---

## 五、本地运行（不依赖队友）

```powershell
# 1) 用 VS Code 打开本文件夹 → 终端（已自动在目录里）
# 2) 装依赖（含 ffmpeg 内置包，无需手动装 ffmpeg）
py -m pip install -r requirements.txt

# 3) 预下载 Whisper 模型（防演示首跑联网卡死，中国网络用镜像）
py scripts/setup_whisper_model.py

# 4) 跑完整流水线（input/ 放一个测试视频）
py main.py input/test.mp4

# 或者只测 M3 的 4 个 agent（更快、更聚焦）
py M3_test.py
```

---

## 六、与队友的对接点

- **M1（编排）**：已用 `agent = XxxAgent(config); agent.run(pipeline_data)` 标准接口接入，你无需改动。
- **M2（剪辑）**：`clip_agent` 负责把长视频切成 15–60s 短片；我这边用 STUB 占位，正式版由 M2 提供。
- **M4（前端）/ M5（QA）**：`run()` 返回的 clip 数据（subtitles / audio_translations / thumbnails / platform_versions）直接喂前端展示与验收。
