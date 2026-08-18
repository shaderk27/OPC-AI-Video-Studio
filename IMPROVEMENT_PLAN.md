# One-Person AI Video Studio — Improvement Plan & Team Assignment

> Base version: a working Python CLI pipeline (`main.py`) with 8 agents running sequentially.
> Goal: turn it into an impressive, demo-ready multi-agent studio with a web frontend in 4 days.
> Evaluation criteria: System Design · Creativity · Demo Quality · Presentation

---

## 1. Current State Assessment

| Aspect | Status | Gap |
|--------|--------|-----|
| Agent pipeline | 8 agents work (Content Analyzer → Clip → Subtitle → Audio → Thumbnail → Social → Format → Quality) | Sequential only; no orchestrator/manager agent |
| Execution | Python CLI (`python main.py video.mp4`) | No web UI; not demo-friendly |
| Multi-agent platform | None — plain Python classes | Brief recommends WorkBuddy/Coze; not used yet |
| Human control | None — fully automated | No review/approve gate; "one person" can't steer |
| Error handling | Stop on core failure, skip others | No retry, no fallback, no escalation |
| Quality metrics | Quality agent outputs a report | No success criteria; "works perfectly" undefined |
| Output | Files in `output/` | No gallery/preview UI |

---

## 2. Suggested Improvements (Prioritized)

### P0 — Must-have for a strong demo (Days 1–2)

**A. Web frontend (thin wrapper)** — *the demo surface*
- Stack: **FastAPI** backend (wraps the existing Python pipeline) + **vanilla HTML/CSS/JS** or a lightweight React app (no build step needed if vanilla).
- Flow: Upload/select video → choose languages & clip count → watch agent progress (live status) → review generated clips in a gallery → approve/edit captions → export.
- Why: directly boosts **Demo Quality** and embodies the "one person, one interface" vision.

**B. Human-in-the-loop review gate** — *the "one person" is the boss*
- After the Clip agent produces highlight candidates, pause the pipeline and show them in the web UI.
- The human can: approve all, drop weak clips, edit clip boundaries, or trigger re-cut.
- This is the single biggest **System Design** upgrade — it shows you understand that the human directs the AI team.

**C. Manager/Orchestrator agent** — *true multi-agent coordination*
- Add a `ManagerAgent` that: receives the request, decides which agents to run, routes work, handles failures (retry once, then escalate to human), and aggregates the final output.
- Map each existing agent to a WorkBuddy expert/skill so the "AI team" is visible and the architecture matches the brief.

### P1 — Strong differentiators (Days 2–3)

**D. Emotion-aware highlight selection** — *the creativity angle*
- Use the Content Analyzer's sentiment data + audio energy to score "emotional peaks" and prefer clips with high engagement potential. Many teams will just keyword-match; this is visibly smarter.
- Add a simple "hook score" (0–100) shown in the review UI.

**E. Failure handling & retry**
- If an agent fails, the Manager retries once with adjusted params; if it fails again, it flags the clip as "needs attention" and continues — no full pipeline crash.

**F. Measurable success metrics**
- Define: clip relevance (does it match a detected topic?), subtitle WER (word error rate proxy via re-transcription check), hook score, output length compliance (15–60s), processing time.
- Show these in the web UI as a "studio dashboard."

### P2 — Polish (Day 4)

**G. Demo resilience**
- Pre-render a full run's output as backup in case the live demo hits network/timeout issues.
- Keep a short (~3 min) test video for fast live demo, not the full 10+ min.

**H. Pitch deck & storytelling**
- Frame the story: "From 10 minutes of raw footage to 5 ready-to-post clips in one click — here's the AI team doing it."

---

## 3. Web Frontend Approach — Confirmed: Thin Wrapper

```
┌───────────────────────────────────┐
│   Web frontend (browser)          │   ← M4 builds this
│   upload · progress · review ·    │
│   gallery · export                │
└───────────────┬───────────────────┘
                │ HTTP / SSE
┌───────────────▼───────────────────┐
│   FastAPI layer (api/server.py)   │   ← M1 + M4
│   /upload  /run  /status  /clips  │
│   /approve  /export               │
└───────────────┬───────────────────┘
                │ calls
┌───────────────▼───────────────────┐
│   Manager agent + pipeline        │   ← M1 owns, M2/M3 build agents
│   (existing agents + new ones)    │
└───────────────────────────────────┘
```

**Why this is the right call:**
- The Python pipeline already works — we add value on top, not rebuild.
- FastAPI is ~50 lines to wrap `run_studio()`.
- The web UI makes the demo feel like a product, not a script.
- Time-boxed: frontend is ~1.5 days of effort; the rest goes to agent depth.

**Tech choices (keep it simple):**
- Backend: FastAPI (already Python; async; built-in docs).
- Real-time progress: Server-Sent Events (SSE) — simpler than WebSockets.
- Frontend: single `index.html` + minimal JS (fetch + render). No build step. If a team member knows React, a Vite scaffold is fine too.
- File serving: FastAPI `StaticFiles` for the `output/` gallery.

---

## 4. Six-Member Assignment

Every member owns a concrete deliverable track and builds something real.

### Member 1 — System Architect & Orchestrator
**Owns:** Manager agent, pipeline integration, FastAPI skeleton, the end-to-end "it works" demo.
- Day 1: Write `ManagerAgent` design; create `api/server.py` (FastAPI) with `/upload`, `/run`, `/status` endpoints; map each agent to a WorkBuddy expert/skill entry for the architecture diagram.
- Day 2: Wire Manager to call the existing agents with retry logic; connect FastAPI to the pipeline; stream progress via SSE.
- Day 3: End-to-end integration test (upload → run → review → export); fix routing bugs.
- Day 4: Final tuning; stand by the live demo; prepare the architecture slide.

### Member 2 — Content Agent Engineer
**Owns:** Scriptwriter/Clip logic + Captioner (Social Media) agent prompts & behavior.
- Day 1: Improve `clip_agent.py` highlight selection using Content Analyzer topics; draft the "hook score" formula; refine `social_media_agent.py` captions for Douyin/Xiaohongshu tone.
- Day 2: Add emotion-aware scoring (sentiment + audio energy); test on real Shenzhen travel footage.
- Day 3: Iterate prompts with M5's test data; tune caption tone/hashtags; ensure clip relevance > 80%.
- Day 4: Finalize; prepare the exact demo input video + expected outputs.

### Member 3 — Media Agent Engineer
**Owns:** Subtitle, Audio (voiceover), Thumbnail, Format agents — the media pipeline.
- Day 1: Audit `subtitle_agent.py` (Whisper) and `audio_agent.py` (Edge TTS) on CN network; verify `format_agent.py` outputs 9:16/16:9/1:1; check `thumbnail_agent.py`.
- Day 2: Ensure subtitle burn-in works; verify multi-language TTS (zh/en); optimize render speed for demo.
- Day 3: Tune clip assembly; fix any sync issues; make sure 3 clips render in < 3 min for the live demo.
- Day 4: Final render quality check; stage the demo output files.

### Member 4 — Frontend Developer
**Owns:** The web control panel (upload → progress → review → gallery → export).
- Day 1: Scaffold `web/index.html` + `app.js`; design the 4 screens (upload, running, review, gallery); plan API contract with M1.
- Day 2: Build upload + live progress view (SSE); connect to FastAPI; show agent-by-agent status.
- Day 3: Build the **human review gate** UI (approve/drop/edit clips before final render) + the gallery + export buttons.
- Day 4: Polish UI; make it demo-ready; add the "studio dashboard" with M5's metrics.

### Member 5 — Data & QA Engineer
**Owns:** Test data, quality metrics, the iterate/validate loop, the metrics dashboard.
- Day 1: Source 3–5 Shenzhen travel vlogs (or use sample footage); define success metrics (relevance, hook score, WER proxy, length compliance, time); write a test script.
- Day 2: Run the first end-to-end test on real video; collect failures; build the test harness.
- Day 3: Run iterations 2–3; document metrics before/after each change; file bugs to M1/M2/M3.
- Day 4: Final validation report; feed metrics into M4's dashboard; prepare the "results" slide.

### Member 6 — Creative & Demo Lead
**Owns:** The differentiation feature, pitch deck, demo script, backup video, live presentation.
- Day 1: Define the differentiation angle (emotion-aware clips + hook score); draft the 5-minute demo script; outline the pitch deck.
- Day 2: Build the differentiation feature with M2 (the emotion scoring + a "trend-aware caption" twist); start the pitch deck slides.
- Day 3: Finish pitch deck; rehearse the live demo end-to-end; record a backup video of a full run.
- Day 4: Final rehearsal; present. Have the backup video + pre-rendered outputs ready.

---

## 5. Four-Day Schedule (Milestones)

| Day | Theme | End-of-day milestone |
|-----|-------|---------------------|
| 1 | Design & foundation | Architecture agreed; FastAPI skeleton up; agents mapped; test videos sourced; demo script drafted |
| 2 | Build core | All agents integrated via Manager; web upload + progress working; first end-to-end run on real video |
| 3 | Integrate & iterate | Human review gate live; emotion scoring done; 2–3 tuning iterations; pitch deck drafted; backup video recorded |
| 4 | Polish & present | Final demo run; metrics dashboard live; pitch deck final; rehearsal; **present** |

**Daily sync:** 15-min standup at 9:00 (M1 leads). Integration test at 16:00 (M1 + M5). Demo rehearsal at 17:30 (M6 leads, Days 3–4).

---

## 6. Success Metrics (define "works perfectly")

| Metric | Target | Measured by |
|--------|--------|------------|
| Clip relevance | ≥ 80% of clips match a detected topic | M5 |
| Hook score | avg ≥ 60/100 | M2 + M5 |
| Output length | 100% of clips in 15–60s | M3 |
| Subtitle presence | 100% of clips have ≥ 1 language subtitle | M3 |
| Pipeline reliability | ≤ 1 unrecoverable failure per run | M1 |
| Demo render time | ≤ 3 min for 3 clips | M3 |
| Human approval | review gate functional in web UI | M4 |

---

## 7. Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Live demo network failure | Pre-rendered backup video + cached outputs |
| Whisper model slow on first run | Pre-download model (`scripts/setup_whisper_model.py`) before demo |
| CN network blocks services | Already handled (hf-mirror, MyMemory, Edge TTS) — verify on Day 1 |
| Frontend takes too long | Vanilla HTML/JS (no build step); M1 backs up M4 if needed |
| Agent integration bugs | Manager retry logic + M5's test harness catches them by Day 2 |
| Scope creep | P0 only by Day 2 end; P1 on Day 3 only if P0 is solid |

---

## 8. Deliverables Checklist

- [ ] Web frontend (upload → progress → review → gallery → export)
- [ ] Manager agent with retry/fallback
- [ ] Human-in-the-loop review gate
- [ ] Emotion-aware highlight scoring (differentiator)
- [ ] Metrics dashboard
- [ ] Pitch deck (slides)
- [ ] Backup demo video
- [ ] Architecture diagram (for presentation)
- [ ] Test data (3–5 Shenzhen travel videos)
- [ ] Final validation report
