/* ═══════════════════════════════════════════════════════════════════
   OPC AI Video Studio — Control Panel (M4)
   Vanilla JS · no build step · fetch + EventSource + render
   Falls back to Demo Mode when the backend (/api/config) is unreachable.
   ═══════════════════════════════════════════════════════════════════ */
"use strict";

/* ───────────────────────── helpers ───────────────────────── */
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const el = (tag, cls, html) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (html !== undefined) n.innerHTML = html;
  return n;
};
const fmtTime = (s) => {
  s = Math.max(0, Math.round(s || 0));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
};
const fmtMB = (mb) => (mb >= 1024 ? `${(mb / 1024).toFixed(1)} GB` : `${mb} MB`);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
}[c]));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/* ───────────────────────── state ───────────────────────── */
const state = {
  demo: false,
  config: null,
  video: null,          // {name, path, size_mb, modified}
  subLangs: new Set(),
  audioLangs: new Set(),
  subtitleMode: "separate",
  formats: new Set(),
  stages: null,         // Set of active stage ids
  advanced: {},
  runId: null,
  run: { status: "idle", stages: {}, elapsed: 0, video: "" },
  review: [],
  reviewActive: false,
  manifest: null,
  quality: null,
  downloads: null,
  pollTimer: null,
  clockTimer: null,
  runStartTs: 0,
};

/* ───────────────────────── demo data ───────────────────────── */
const DEMO_LANGUAGES = [
  { code: "en", name: "English" }, { code: "zh", name: "Chinese" },
  { code: "es", name: "Spanish" }, { code: "fr", name: "French" },
  { code: "de", name: "German" }, { code: "ja", name: "Japanese" },
  { code: "ko", name: "Korean" }, { code: "ar", name: "Arabic" },
  { code: "hi", name: "Hindi" }, { code: "pt", name: "Portuguese" },
  { code: "ru", name: "Russian" }, { code: "it", name: "Italian" },
];
const DEMO_STAGES = [
  { id: "content_analysis", name: "Content Analyzer", role: "Video Content Intelligence Analyst" },
  { id: "clip", name: "Clip Expert", role: "Video Highlight Cutter" },
  { id: "subtitle", name: "Subtitle Expert", role: "Multi-Language Subtitle Generator" },
  { id: "audio", name: "Audio Expert", role: "Audio Translation & Voiceover" },
  { id: "thumbnail", name: "Thumbnail Generator", role: "Thumbnail Creator" },
  { id: "social_media", name: "Social Media Copywriter", role: "Content Strategist" },
  { id: "format", name: "Format Adapter", role: "Multi-Platform Formatter" },
  { id: "quality", name: "Quality Reviewer", role: "QA Specialist" },
];
const DEMO_CONFIG = {
  project: "OPC AI Video Studio",
  languages: DEMO_LANGUAGES,
  formats: [
    { platform: "TikTok", width: 1080, height: 1920, aspect_ratio: "9:16" },
    { platform: "YouTube Shorts", width: 1080, height: 1920, aspect_ratio: "9:16" },
    { platform: "Instagram Reels", width: 1080, height: 1920, aspect_ratio: "9:16" },
    { platform: "YouTube", width: 1920, height: 1080, aspect_ratio: "16:9" },
    { platform: "Instagram Feed", width: 1080, height: 1080, aspect_ratio: "1:1" },
  ],
  whisper_models: ["tiny", "base", "small", "medium", "large-v3"],
  detection_strategies: ["combined", "transcript", "audio_energy", "silence"],
  tts_engines: ["edge", "gtts"],
  tones: ["energetic", "casual", "professional", "educational"],
  stages: DEMO_STAGES,
  defaults: {
    subtitle_languages: ["en", "zh"],
    subtitle_mode: "separate",
    subtitle_formats: ["srt", "vtt"],
    audio_languages: ["zh"],
    tts_engine: "edge",
    formats: ["TikTok", "YouTube Shorts", "Instagram Reels", "YouTube", "Instagram Feed"],
    stages: DEMO_STAGES.map((s) => s.id),
    whisper_model: "base",
    max_clips: 5, min_clip_duration: 15, max_clip_duration: 60,
    detection_strategy: "combined",
    padding_before: 1.5, padding_after: 1.0, snap_to_silence: true,
    gemini_required: false,
    source_language: null,
    original_audio_volume: 0.15, voiceover_volume: 1.0, tts_speed: 1.0,
    thumbnail_enabled: true,
    social_tone: "energetic",
    quality_checks: { av_sync: true, subtitle_sync: true, silence_gaps: true, min_audio_level: -30 },
  },
};
const DEMO_VIDEOS = [
  { name: "Kayaking.mp4", path: "input/Kayaking.mp4", size_mb: 2964, modified: "2026-08-18 13:41" },
];

/* ───────────────────────── api client ───────────────────────── */
async function api(path, opts = {}) {
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), opts.timeout || 20000);
  try {
    const res = await fetch(path, { ...opts, signal: ctrl.signal });
    if (!res.ok) {
      let msg = `HTTP ${res.status}`;
      try { msg = (await res.json()).detail || msg; } catch (e) {}
      throw new Error(msg);
    }
    return await res.json();
  } finally { clearTimeout(t); }
}

/* ───────────────────────── boot ───────────────────────── */
async function boot() {
  try {
    state.config = await api("/api/config", { timeout: 6000 });
    state.demo = false;
    setBadge("live");
  } catch (e) {
    state.demo = true;
    state.config = DEMO_CONFIG;
    setBadge("demo");
    $("#demo-banner").classList.remove("hidden");
    log("Demo Mode: backend not reachable — using simulated pipeline.", "warn");
  }
  renderSetup();
  loadVideos();
}

function setBadge(kind) {
  const b = $("#api-badge");
  b.className = "badge badge-" + kind;
  b.textContent = kind === "live" ? "● LIVE API" : "● DEMO MODE";
}

/* ───────────────────────── setup screen ───────────────────────── */
function renderSetup() {
  const cfg = state.config;
  const d = cfg.defaults;

  // languages — start empty so the user explicitly chooses
  buildChips($("#sub-langs"), cfg.languages, new Set(), (code, on) => {
    on ? state.subLangs.add(code) : state.subLangs.delete(code);
    updateRunSummary();
  });
  buildChips($("#audio-langs"), cfg.languages, new Set(), (code, on) => {
    on ? state.audioLangs.add(code) : state.audioLangs.delete(code);
    updateRunSummary();
  });

  // subtitle mode
  state.subtitleMode = d.subtitle_mode || "separate";
  syncSeg();
  $$("#subtitle-mode .seg-btn").forEach((b) => b.addEventListener("click", () => {
    state.subtitleMode = b.dataset.mode;
    syncSeg();
  }));

  // subtitle formats
  const sf = new Set(d.subtitle_formats || ["srt", "vtt"]);
  $("#sub-fmt-srt").checked = sf.has("srt");
  $("#sub-fmt-vtt").checked = sf.has("vtt");

  // tts engine
  fillSelect($("#tts-engine"), cfg.tts_engines, d.tts_engine, (v) => state.advanced.tts_engine = v);

  // formats grouped by aspect ratio — start unticked so the user explicitly chooses
  renderFormats(cfg.formats, []);

  // advanced
  fillSelect($("#adv-whisper"), cfg.whisper_models, d.whisper_model, (v) => state.advanced.whisper_model = v);
  fillSelect($("#adv-strategy"), cfg.detection_strategies, d.detection_strategy, (v) => state.advanced.detection_strategy = v);
  fillSelect($("#adv-tone"), cfg.tones, d.social_tone, (v) => state.advanced.social_tone = v);
  fillSelect($("#adv-source-lang"),
    [{ code: "", name: "Auto-detect" }].concat(cfg.languages), d.source_language || "",
    (v) => state.advanced.source_language = v || null, "code");

  setNum($("#adv-max-clips"), d.max_clips, (v) => state.advanced.max_clips = v);
  setNum($("#adv-min-dur"), d.min_clip_duration, (v) => state.advanced.min_clip_duration = v);
  setNum($("#adv-max-dur"), d.max_clip_duration, (v) => state.advanced.max_clip_duration = v);
  setNum($("#adv-pad-before"), d.padding_before, (v) => state.advanced.padding_before = v);
  setNum($("#adv-pad-after"), d.padding_after, (v) => state.advanced.padding_after = v);
  setNum($("#adv-orig-vol"), d.original_audio_volume, (v) => state.advanced.original_audio_volume = v);
  setNum($("#adv-vo-vol"), d.voiceover_volume, (v) => state.advanced.voiceover_volume = v);
  setNum($("#adv-tts-speed"), d.tts_speed, (v) => state.advanced.tts_speed = v);

  $("#adv-snap").checked = !!d.snap_to_silence;
  $("#adv-gemini-required").checked = !!d.gemini_required;
  $("#adv-gemini-required").addEventListener("change", () => {
    if ($("#adv-gemini-required").checked) {
      state.stages.add("clip");
      const clipToggle = $('#adv-stages input[data-stage="clip"]');
      if (clipToggle) clipToggle.checked = true;
    }
    updateRunSummary();
  });
  $("#adv-thumb").checked = d.thumbnail_enabled !== false;
  $("#adv-check-av").checked = !!d.quality_checks?.av_sync;
  $("#adv-check-sub").checked = !!d.quality_checks?.subtitle_sync;
  $("#adv-check-sil").checked = !!d.quality_checks?.silence_gaps;

  // stage toggles
  state.stages = new Set(d.stages);
  const sc = $("#adv-stages");
  sc.innerHTML = "";
  cfg.stages.forEach((s) => {
    const lb = el("label", "check");
    const cb = el("input");
    cb.type = "checkbox";
    cb.dataset.stage = s.id;
    cb.checked = state.stages.has(s.id);
    cb.addEventListener("change", () => {
      cb.checked ? state.stages.add(s.id) : state.stages.delete(s.id);
      updateRunSummary();
    });
    lb.appendChild(cb);
    lb.appendChild(el("span", "", esc(s.name)));
    sc.appendChild(lb);
  });

  // advanced toggle
  const at = $("#advanced-toggle");
  at.addEventListener("click", () => {
    const open = $("#advanced-body").classList.toggle("hidden");
    at.classList.toggle("open", !open);
  });

  // dropzone + browse
  setupDropzone();

  // run button
  $("#run-btn").addEventListener("click", startRun);
  updateRunSummary();
}

function buildChips(container, items, selected, onToggle) {
  container.innerHTML = "";
  items.forEach((it) => {
    const c = el("button", "chip" + (selected.has(it.code) ? " on" : ""),
      `${esc(it.name)} <small>${it.code}</small>`);
    c.addEventListener("click", () => {
      const on = !c.classList.contains("on");
      c.classList.toggle("on", on);
      onToggle(it.code, on);
    });
    container.appendChild(c);
  });
}

function fillSelect(sel, items, value, onPick, key = "code") {
  sel.innerHTML = "";
  items.forEach((it) => {
    const v = typeof it === "string" ? it : it[key];
    const label = typeof it === "string" ? it : (it.name || v);
    sel.appendChild(el("option", "", esc(label))).value = v;
  });
  sel.value = value;
  sel.addEventListener("change", () => onPick(sel.value));
}

function setNum(inp, val, onPick) {
  inp.value = val;
  inp.addEventListener("input", () => onPick(parseFloat(inp.value)));
}

function renderFormats(formats, defaults) {
  const groups = {};
  formats.forEach((f) => (groups[f.aspect_ratio] = groups[f.aspect_ratio] || []).push(f));
  const wrap = $("#format-options");
  wrap.innerHTML = "";
  Object.entries(groups).forEach(([ratio, fmts]) => {
    const g = el("div", "fmt-group");
    g.appendChild(el("div", "fmt-group-head", `<span class="ratio-badge">${esc(ratio)}</span>${esc(ratio)} — one shared render`));
    const opts = el("div", "fmt-options");
    fmts.forEach((f) => {
      const lb = el("label", "check");
      const cb = el("input");
      cb.type = "checkbox";
      cb.checked = defaults.includes(f.platform);
      if (cb.checked) state.formats.add(f.platform);
      cb.addEventListener("change", () => {
        cb.checked ? state.formats.add(f.platform) : state.formats.delete(f.platform);
        updateRunSummary();
      });
      const size = el("small", "", ` ${f.width}×${f.height}`);
      lb.appendChild(cb);
      lb.appendChild(el("span", "", `${esc(f.platform)}`));
      lb.appendChild(size);
      opts.appendChild(lb);
    });
    g.appendChild(opts);
    wrap.appendChild(g);
  });
}

function syncSeg() {
  $$("#subtitle-mode .seg-btn").forEach((b) => b.classList.toggle("active", b.dataset.mode === state.subtitleMode));
  $("#burn-row").classList.toggle("hidden", !["burn", "both"].includes(state.subtitleMode));
  // file formats only matter when we are producing separate subtitle files
  $("#sub-fmts-row").classList.toggle("hidden", state.subtitleMode === "burn");
}

/* ───────────────────────── videos ───────────────────────── */
async function loadVideos() {
  const list = $("#video-list");
  try {
    const { videos } = state.demo ? { videos: DEMO_VIDEOS } : await api("/api/videos");
    $("#video-count").textContent = `${videos.length} found`;
    if (!videos.length) { list.innerHTML = `<div class="muted">No videos in <code>input/</code> yet — upload one above.</div>`; return; }
    list.innerHTML = "";
    videos.forEach((v) => {
      const item = el("div", "video-item");
      item.innerHTML = `
        <div class="vi-name">🎞 ${esc(v.name)}</div>
        <div class="vi-meta">${fmtMB(v.size_mb)} · ${esc(v.modified)}</div>`;
      item.addEventListener("click", () => {
        $$(".video-item", list).forEach((x) => x.classList.remove("selected"));
        item.classList.add("selected");
        state.video = v;
        $("#selected-video").classList.remove("hidden");
        $("#sv-name").textContent = v.name;
        $("#sv-size").textContent = fmtMB(v.size_mb);
        $("#sv-time").textContent = v.modified;
        updateRunSummary();
      });
      list.appendChild(item);
    });
  } catch (e) {
    list.innerHTML = `<div class="muted">Could not list videos: ${esc(e.message)}</div>`;
  }
}

function setupDropzone() {
  const dz = $("#dropzone");
  const inp = $("#file-input");
  const browse = () => inp.click();
  dz.addEventListener("click", (e) => { if (e.target.id !== "dz-browse") browse(); });
  $("#dz-browse").addEventListener("click", (e) => { e.stopPropagation(); browse(); });
  inp.addEventListener("change", () => { if (inp.files.length) uploadVideo(inp.files[0]); });
  ["dragover", "dragenter"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("over"); }));
  ["dragleave", "drop"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("over"); }));
  dz.addEventListener("drop", (e) => { if (e.dataTransfer.files.length) uploadVideo(e.dataTransfer.files[0]); });
}

async function uploadVideo(file) {
  const dz = $("#dropzone");
  dz.innerHTML = `<p><strong>Uploading ${esc(file.name)}…</strong></p>`;
  try {
    if (state.demo) {
      await sleep(1200);
      state.video = { name: file.name, path: `input/${file.name}`, size_mb: +(file.size / 1048576).toFixed(1), modified: "just now" };
      $("#selected-video").classList.remove("hidden");
      $("#sv-name").textContent = file.name;
      $("#sv-size").textContent = fmtMB(state.video.size_mb);
      $("#sv-time").textContent = "just now";
    } else {
      const fd = new FormData();
      fd.append("file", file);
      const res = await api("/api/upload", { method: "POST", body: fd, timeout: 0 });
      state.video = res;
      await loadVideos();
      $$(".video-item").forEach((x) => { const n = x.querySelector(".vi-name").textContent.trim().endsWith(file.name); if (n) x.click(); });
    }
  } catch (e) {
    dz.innerHTML = `<p><strong>Upload failed:</strong> ${esc(e.message)}</p>`;
  }
  updateRunSummary();
}

/* ───────────────────────── run summary ───────────────────────── */
function updateRunSummary() {
  const btn = $("#run-btn");
  if (!state.video) { $("#run-summary").textContent = "Select a video to start."; btn.disabled = true; return; }
  if (!state.subLangs.size) { $("#run-summary").textContent = "Pick at least one subtitle language."; btn.disabled = true; return; }
  if (!state.formats.size) { $("#run-summary").textContent = "Pick at least one output format."; btn.disabled = true; return; }
  const parts = [
    `${state.video.name}`,
    `subs: ${[...state.subLangs].join("+")}${state.subtitleMode === "burn" ? " (burned)" : state.subtitleMode === "both" ? " (+burned)" : ""}`,
    state.audioLangs.size ? `dub: ${[...state.audioLangs].join("+")}` : "no dubbing",
    `formats: ${state.formats.size}`,
    `clips ≤ ${$("#adv-max-clips").value || state.config.defaults.max_clips}`,
    $("#adv-gemini-required").checked ? "Gemini required" : "Gemini optional",
  ];
  $("#run-summary").textContent = parts.join("  ·  ");
  btn.disabled = false;
}

/* ───────────────────────── run payload ───────────────────────── */
function buildPayload() {
  const d = state.config.defaults;
  return {
    video: state.video.path,
    subtitle_languages: [...state.subLangs],
    subtitle_mode: state.subtitleMode,
    subtitle_formats: [($("#sub-fmt-srt").checked ? "srt" : null), ($("#sub-fmt-vtt").checked ? "vtt" : null)].filter(Boolean),
    audio_languages: [...state.audioLangs],
    tts_engine: state.advanced.tts_engine || d.tts_engine,
    formats: [...state.formats],
    stages: [...state.stages],
    whisper_model: state.advanced.whisper_model || d.whisper_model,
    max_clips: parseInt($("#adv-max-clips").value) || d.max_clips,
    min_clip_duration: parseInt($("#adv-min-dur").value) || d.min_clip_duration,
    max_clip_duration: parseInt($("#adv-max-dur").value) || d.max_clip_duration,
    detection_strategy: state.advanced.detection_strategy || d.detection_strategy,
    padding_before: parseFloat($("#adv-pad-before").value) ?? d.padding_before,
    padding_after: parseFloat($("#adv-pad-after").value) ?? d.padding_after,
    snap_to_silence: $("#adv-snap").checked,
    gemini_required: $("#adv-gemini-required").checked,
    source_language: state.advanced.source_language || null,
    original_audio_volume: parseFloat($("#adv-orig-vol").value) ?? d.original_audio_volume,
    voiceover_volume: parseFloat($("#adv-vo-vol").value) ?? d.voiceover_volume,
    tts_speed: parseFloat($("#adv-tts-speed").value) ?? d.tts_speed,
    thumbnail_enabled: $("#adv-thumb").checked,
    social_tone: state.advanced.social_tone || d.social_tone,
    quality_checks: {
      av_sync: $("#adv-check-av").checked,
      subtitle_sync: $("#adv-check-sub").checked,
      silence_gaps: $("#adv-check-sil").checked,
      min_audio_level: d.quality_checks?.min_audio_level ?? -30,
    },
  };
}

/* ───────────────────────── start run ───────────────────────── */
async function startRun() {
  const payload = buildPayload();
  const errBox = $("#run-error");
  errBox.textContent = "";

  if (state.demo) {
    resetRunState();
    showView("progress");
    enableStep("review", false); enableStep("gallery", false);
    $("#progress-video").textContent = payload.video.split("/").pop();
    startClock();
    state.runId = "demo-" + Date.now();
    logRun("run_started", { video: payload.video.split("/").pop(), stages: payload.stages });
    simulateRun(payload);
    return;
  }
  try {
    const { run_id } = await api("/api/run", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
    resetRunState();
    showView("progress");
    enableStep("review", false); enableStep("gallery", false);
    $("#progress-video").textContent = payload.video.split("/").pop();
    startClock();
    state.runId = run_id;
    connectSSE(run_id);
    startPolling(run_id);
  } catch (e) {
    errBox.textContent = "Failed to start run: " + e.message;
    state.run = { status: null, stages: {}, elapsed: 0, video: "" };
  }
}

function resetRunState() {
  state.run = { status: "running", stages: {}, elapsed: 0, video: "" };
  state.review = [];
  state.reviewActive = false;
  state.manifest = state.quality = state.downloads = null;
  state.runStartTs = Date.now();
  $("#stage-rail").innerHTML = "";
  $("#progress-error").textContent = "";
  $("#progress-error").classList.add("hidden");
  clearLog();
  if (state.pollTimer) clearInterval(state.pollTimer);
  if (state.clockTimer) clearInterval(state.clockTimer);
  state.clockTimer = setInterval(() => {
    $("#progress-timer").textContent = fmtTime((Date.now() - state.runStartTs) / 1000);
  }, 500);
}

function startClock() {
  if (state.clockTimer) clearInterval(state.clockTimer);
  state.clockTimer = setInterval(() => {
    $("#progress-timer").textContent = fmtTime((Date.now() - state.runStartTs) / 1000);
  }, 500);
}

function buildRail(stages) {
  const rail = $("#stage-rail");
  rail.innerHTML = "";
  stages.forEach((id, i) => {
    const meta = state.config.stages.find((s) => s.id === id) || { name: id, role: "" };
    const node = el("div", "stage-node", `
      <div class="sn-name">${esc(meta.name)}</div>
      <div class="sn-role">${esc(meta.role)}</div>
      <div class="sn-state">pending</div>`);
    node.dataset.stage = id;
    rail.appendChild(node);
  });
}

function setStage(id, status, extra = "") {
  let node = $(`[data-stage="${id}"]`, $("#stage-rail"));
  if (!node) return;
  node.className = "stage-node " + (status === "active" ? "active" : status === "done" ? "done" : status === "failed" ? "failed" : "");
  const st = $(".sn-state", node);
  st.textContent = status === "active" ? "running…" : status === "done" ? "done" : status === "failed" ? "failed" : "pending";
  if (extra) st.textContent += ` ${extra}`;
}

/* ───────────────────────── SSE + polling ───────────────────────── */
function connectSSE(runId) {
  const es = new EventSource(`/api/runs/${runId}/events`);
  const names = ["run_started", "stage_started", "stage_log", "stage_completed", "stage_failed", "review_ready", "review_completed", "run_completed", "run_failed", "ping"];
  names.forEach((n) => es.addEventListener(n, (ev) => {
    let d = {};
    try { d = JSON.parse(ev.data); } catch (e) {}
    logRun(n, d);
  }));
  es.onerror = () => es.close(); // polling fallback covers it
}

function startPolling(runId) {
  if (state.pollTimer) clearInterval(state.pollTimer);
  state.pollTimer = setInterval(async () => {
    try {
      const snap = await api(`/api/runs/${runId}`);
      if (snap.status === "completed") { clearInterval(state.pollTimer); handleRunCompleted(); }
      else if (snap.status === "failed") { clearInterval(state.pollTimer); handleRunFailed(snap.error || "pipeline failed (see log)"); }
      else if (snap.status === "awaiting_review" && !state.reviewActive) { handleReviewReady(); }
      else {
        Object.entries(snap.stages || {}).forEach(([id, st]) => {
          if (st.status !== "pending") setStage(id, st.status);
        });
      }
    } catch (e) { /* server restarting? */ }
  }, 2000);
}

/* ───────────────────────── run events ───────────────────────── */
function logRun(event, d) {
  switch (event) {
    case "run_started":
      buildRail(d.stages || []);
      log(`Run started — video: ${d.video || ""}`, "ok");
      break;
    case "stage_started":
      setStage(d.stage, "active");
      log(`▶ ${d.name || d.stage} (${d.index}/${d.total})`, "info", d.stage);
      break;
    case "stage_log":
      log(d.message || "", "info", d.stage);
      break;
    case "stage_completed":
      setStage(d.stage, "done");
      log(`✓ ${d.stage} done${d.message?.clips ? ` — ${d.message.clips} clips` : ""}`, "ok", d.stage);
      break;
    case "stage_failed":
      setStage(d.stage, "failed");
      log(`✗ ${d.stage} failed: ${d.error || ""}`, "err", d.stage);
      break;
    case "review_ready":
      log(`${d.count || 0} clips ready — waiting for your approval.`, "info", "clip");
      handleReviewReady();
      break;
    case "review_completed":
      log(`Review complete — ${d.approved || 0} approved, ${d.rejected || 0} dropped.`, "ok");
      state.reviewActive = false;
      showView("progress");
      break;
    case "run_completed":
      log(`Run completed in ${fmtTime(d.elapsed)} — ${d.clips || 0} clips.`, "ok");
      stopClock();
      handleRunCompleted();
      break;
    case "run_failed":
      log(`Run FAILED: ${d.error || ""}`, "err");
      stopClock();
      handleRunFailed(d.error);
      break;
  }
}

function stopClock() {
  if (state.clockTimer) { clearInterval(state.clockTimer); state.clockTimer = null; }
}

/* ───────────────────────── log console ───────────────────────── */
function log(msg, kind = "info", stage = "") {
  const box = $("#log-console");
  if (box.firstChild && box.firstChild.className === "muted") box.innerHTML = "";
  const line = el("div", "log-line" + (kind === "ok" ? " ok" : kind === "err" ? " err" : kind === "warn" ? " warn" : ""));
  const ts = el("span", "ts", new Date().toLocaleTimeString());
  line.appendChild(ts);
  if (stage) line.appendChild(el("span", "stage-tag", esc(stage)));
  line.appendChild(document.createTextNode(msg));
  box.appendChild(line);
  box.scrollTop = box.scrollHeight;
}
const clearLog = () => { $("#log-console").innerHTML = '<div class="muted">Waiting for pipeline…</div>'; };
$("#log-clear").addEventListener("click", clearLog);

/* ───────────────────────── demo simulation ───────────────────────── */
async function simulateRun(payload) {
  const steps = payload.stages || state.config.defaults.stages;
  buildRail(steps);
  const demoLogs = {
    content_analysis: ["transcribing full video…", "detected language: en", "8 topics · 12 chapters · 9 key moments"],
    clip: ["scoring emotional peaks…", "snapping to silence boundaries…"],
    subtitle: [`generating subtitles (${payload.subtitle_languages.join(", ")})…`],
    audio: [`synthesizing voiceover (${payload.audio_languages.join(", ")})…`],
    thumbnail: ["extracting 3 candidates per clip…"],
    social_media: ["writing captions + hashtags…"],
    format: ["rendering 9:16 / 16:9 / 1:1 (deduplicated)…"],
    quality: ["checking A/V sync, subtitle timing, silence gaps…"],
  };
  for (let i = 0; i < steps.length; i++) {
    const s = steps[i];
    setStage(s, "active");
    log(`▶ ${(state.config.stages.find((x) => x.id === s) || {}).name || s} (${i + 1}/${steps.length})`, "info", s);
    for (const line of (demoLogs[s] || [])) {
      await sleep(700 + Math.random() * 600);
      log(line, "info", s);
    }
    await sleep(500);
    setStage(s, "done");
    log(`✓ ${s} done`, "ok", s);
  }
  log("Run completed (simulated) — 3 clips.", "ok");
  stopClock();
  makeDemoResults(payload);
  handleRunCompleted();
}

function makeDemoResults(payload) {
  const subs = payload.subtitle_languages || ["en"];
  const auds = payload.audio_languages || [];
  const fmts = payload.formats || [];
  const mk = (i, title, start, end, reason, hook) => ({
    index: i, filename: `Kayaking_clip_0${i}.mp4`, path: "",
    url: "", duration: +(end - start + 2.5).toFixed(2), start, end, title, reason,
    hook_score: hook, subtitle_langs: subs, thumbnails: [],
    caption: `${title} — the moment you can't miss! #kayaking #outdoor`,
    hashtags: ["#fyp", "#kayaking", "#adventure"],
    decision: "pending",
  });
  state.manifest = {
    video: "Kayaking.mp4", processing_time_seconds: 187,
    config: { target_languages: subs, max_clips: 3, whisper_model: payload.whisper_model },
    content_analysis: { detected_language: "en", total_segments: 214, topics: ["kayaking", "river", "safety"], chapters: 12, key_moments: 9, summary: "A kayaking adventure down a scenic river with several close calls and a big finale.", overall_sentiment: "positive" },
    clips: [
      mk(1, "Rapid Entry", 12.4, 28.7, "High speech density + keyword 'look at this'", 74),
      mk(2, "The Near Flip", 141.0, 156.0, "Sentiment spike + audio energy peak", 88),
      mk(3, "Calm Aftermath", 402.5, 417.5, "Topic change + reflective moment", 61),
    ],
  };
  state.quality = {
    summary: { total_clips: 3, clips_passed: 3, clips_failed: 0, total_issues: 1, pass_rate: 100 },
    issues: [{ severity: "warning", message: "Clip 2: background wind noise above threshold" }],
  };
  state.downloads = { clips: [], subtitles: [], audio: [], formatted: [], burned: [], thumbnails: [], social: [] };
  state.review = state.manifest.clips.map((c, i) => ({ ...mk(i + 1, c.title, c.start, c.end, c.reason, c.hook_score) }));
}

/* ───────────────────────── completed → review ───────────────────────── */
async function handleRunCompleted() {
  if (state.pollTimer) clearInterval(state.pollTimer);
  if (!state.demo && state.review.length > 0) {
    await showGallery();
    return;
  }
  enableStep("review", true);
  if (state.demo) { await loadReviewDemo(); }
  else {
    try {
      const { clips } = await api(`/api/runs/${state.runId}/clips`);
      state.review = clips;
      if (state.review.length === 0) { log("No clips found in manifest.", "warn"); }
    } catch (e) { log("Could not load review clips: " + e.message, "err"); }
  }
  renderReview();
  showView("review");
}

async function handleReviewReady() {
  if (state.reviewActive) return;
  state.reviewActive = true;
  enableStep("review", true);
  try {
    const { clips } = await api(`/api/runs/${state.runId}/clips`);
    state.review = clips;
    renderReview();
    showView("review");
  } catch (e) {
    state.reviewActive = false;
    log("Could not load clips for review: " + e.message, "err");
  }
}

async function loadReviewDemo() {
  state.review = (state.manifest?.clips || []).map((c, i) => ({
    index: i + 1, filename: c.filename, url: "", duration: c.duration, start: c.start, end: c.end,
    title: c.title, reason: c.reason, hook_score: [74, 88, 61][i], subtitle_langs: state.manifest.config.target_languages,
    thumbnails: [], caption: state.review?.[i]?.caption || `#${i + 1} — ${c.title}`, hashtags: ["#fyp", "#kayaking"],
    decision: "pending",
  }));
}

function handleRunFailed(msg) {
  const message = msg || "Unknown pipeline error";
  log("Pipeline failed — " + message, "err");
  $("#progress-error").textContent = message;
  $("#progress-error").classList.remove("hidden");
  enableStep("review", true); // still let them see whatever was produced
}

/* ───────────────────────── review screen ───────────────────────── */
function renderReview() {
  const grid = $("#review-grid");
  grid.innerHTML = "";
  const preview = (clip) => clip.url
    ? `<video src="${esc(clip.url)}" controls preload="metadata"></video>`
    : `<div class="rc-placeholder" style="height:100%;display:flex;align-items:center;justify-content:center;font-size:44px;background:linear-gradient(135deg,#1a2030,#0b0e14)">🎬</div>`;

  state.review.forEach((clip) => {
    const card = el("div", "review-card" + (clip.decision === "dropped" ? " dropped" : ""));
    const score = clip.gemini_video_score != null
      ? `<span class="rc-hook">Gemini ${clip.gemini_video_score}</span>`
      : (clip.hook_score != null ? `<span class="rc-hook">hook ${clip.hook_score}</span>` : "");
    card.innerHTML = `
      <div class="rc-video">
        ${preview(clip)}
        <span class="rc-badge">${clip.index} · ${fmtTime(clip.duration)}</span>${score}
      </div>
      <div class="rc-body">
        <div class="rc-title">${esc(clip.title)}</div>
        <div class="rc-meta">
          <span>subs: ${esc((clip.subtitle_langs || []).join(", ") || "—")}</span>
          <span>${fmtTime(clip.start)} → ${fmtTime(clip.end)}</span>
        </div>
        <div class="rc-reason">${esc(clip.gemini_video_reason || clip.reason || "")}</div>
        ${clip.gemini_event_type ? `<div class="rc-reason">Event: ${esc(clip.gemini_event_type)}</div>` : ""}
      </div>
      <div class="rc-actions">
        <button class="btn btn-approve ${clip.decision === "approved" ? "on" : ""}" data-act="approve">✓ Approve</button>
        <button class="btn btn-drop ${clip.decision === "dropped" ? "on" : ""}" data-act="drop">✕ Drop</button>
      </div>`;
    card.querySelector('[data-act="approve"]').addEventListener("click", () => setDecision(clip.index, "approved"));
    card.querySelector('[data-act="drop"]').addEventListener("click", () => setDecision(clip.index, "dropped"));
    grid.appendChild(card);
  });

  const anyPending = state.review.some((c) => c.decision === "pending");
  $("#review-next").disabled = anyPending;
  $("#review-next").textContent = anyPending ? "Review every clip to continue" : "Continue processing →";
}

async function setDecision(index, decision) {
  const clip = state.review.find((c) => c.index === index);
  if (!clip) return;
  clip.decision = decision;
  renderReview();
  if (!state.demo) {
    try { await api(`/api/runs/${state.runId}/clips/${index}/decision`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ decision }) }); }
    catch (e) { log("Decision save failed: " + e.message, "err"); }
  }
}

let captionTimer = null;
function saveCaption(index, caption, hashtags) {
  clearTimeout(captionTimer);
  captionTimer = setTimeout(async () => {
    const clip = state.review.find((c) => c.index === index);
    if (!clip) return;
    if (caption != null) clip.caption = caption;
    if (hashtags) clip.hashtags = hashtags;
    if (state.demo) return;
    try {
      await api(`/api/runs/${state.runId}/clips/${index}/caption`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ caption: clip.caption, hashtags: clip.hashtags }) });
    } catch (e) { log("Caption save failed: " + e.message, "err"); }
  }, 600);
}

$("#review-approve-all").addEventListener("click", () => state.review.forEach((c) => setDecision(c.index, "approved")));
$("#review-drop-all").addEventListener("click", () => state.review.forEach((c) => setDecision(c.index, "dropped")));
$("#review-next").addEventListener("click", async () => {
  if (state.demo) { showGallery(); return; }
  const approvedIndices = state.review
    .filter((clip) => clip.decision === "approved")
    .map((clip) => clip.index);
  $("#review-next").disabled = true;
  try {
    await api(`/api/runs/${state.runId}/review`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ approved_indices: approvedIndices }),
    });
    state.reviewActive = false;
    showView("progress");
  } catch (e) {
    $("#review-next").disabled = false;
    log("Could not continue after review: " + e.message, "err");
  }
});

/* ───────────────────────── gallery ───────────────────────── */
async function showGallery() {
  enableStep("gallery", true);
  showView("gallery");
  if (!state.demo) {
    try {
      const [manifest, quality, downloads] = await Promise.all([
        api(`/api/runs/${state.runId}/manifest`),
        api(`/api/runs/${state.runId}/quality`),
        api(`/api/runs/${state.runId}/downloads`),
      ]);
      state.manifest = manifest; state.quality = quality; state.downloads = downloads;
    } catch (e) { log("Could not load gallery data: " + e.message, "err"); }
  }
  renderGallery();
}

function renderGallery() {
  const m = state.manifest || { clips: [], content_analysis: {} };
  const q = state.quality || {};
  const approved = state.review.filter((c) => c.decision !== "dropped").map((c) => c.index);
  const clips = state.demo
    ? (m.clips || []).filter((_, i) => approved.includes(i + 1))
    : (m.clips || []);
  const ca = m.content_analysis || {};

  $("#gallery-subtitle").textContent = m.video ? `Video: ${m.video} · processed in ${fmtTime(m.processing_time_seconds || 0)}` : "";

  // metrics strip
  const metrics = [
    { label: "Clips", value: clips.length, cls: "accent" },
    { label: "Subtitle langs", value: new Set(clips.flatMap((c) => Object.keys(c.subtitles || {}))).size || "—" },
    { label: "Formats", value: m.config?.formats || new Set(clips.flatMap((c) => Object.keys(c.platform_versions || {}))).size },
    { label: "Topics", value: (ca.topics || []).length || "—" },
    { label: "Sentiment", value: ca.overall_sentiment || "—" },
    { label: "Pass rate", value: q.summary ? `${q.summary.pass_rate}%` : "—", cls: q.summary?.pass_rate === 100 ? "ok" : "" },
  ];
  const strip = $("#metrics-strip");
  strip.innerHTML = "";
  metrics.forEach((mt) => {
    const d = el("div", "metric");
    d.innerHTML = `<div class="m-value ${mt.cls || ""}">${esc(mt.value)}</div><div class="m-label">${esc(mt.label)}</div>`;
    strip.appendChild(d);
  });

  // quality card
  const qb = $("#quality-badge");
  if (q.summary) {
    qb.className = "badge " + (q.summary.pass_rate === 100 ? "badge-live" : "badge-demo");
    qb.textContent = `${q.summary.clips_passed}/${q.summary.total_clips} passed · ${q.summary.total_issues} issues`;
    const body = $("#quality-body");
    body.innerHTML = "";
    (q.issues || []).forEach((iss) => {
      const it = el("div", `q-item ${iss.severity === "error" ? "fail" : iss.severity === "warning" ? "warn" : "pass"}`);
      it.innerHTML = `<span class="q-dot"></span><span>[${esc(iss.severity)}] ${esc(iss.message)}</span>`;
      body.appendChild(it);
    });
    if (!(q.issues || []).length) body.innerHTML = `<div class="q-item pass"><span class="q-dot"></span>All checks passed.</div>`;
  } else {
    qb.className = "badge badge-pending"; qb.textContent = "no report";
  }

  // clips
  const grid = $("#gallery-grid");
  grid.innerHTML = "";
  // group burned outputs by source clip stem + language: filename like <stem>_burn_<lang>.mp4
  const burnedByStem = {};
  (state.downloads?.burned || []).forEach((b) => {
    const m = b.filename.match(/^(.+)_burn_([a-z]+)\.mp4$/i);
    if (!m) return;
    const [, stem, lang] = m;
    (burnedByStem[stem] = burnedByStem[stem] || []).push({ lang, url: b.url });
  });

  clips.forEach((clip, ci) => {
    const card = el("div", "g-clip");
    const rev = state.review.find((r) => r.index === ci + 1) || {};
    card.innerHTML = `
      <div class="g-head"><h3>${esc(rev.title || clip.title || clip.filename)}</h3><span class="g-dur">${fmtTime(clip.duration)}</span></div>
      <div class="g-tabs"></div>
      <div class="g-body"><div class="g-video"></div><div class="g-content"></div></div>`;
    grid.appendChild(card);
    const tabsEl = $(".g-tabs", card);
    const videoBox = $(".g-video", card);
    const contentBox = $(".g-content", card);

    const views = [];
    const pth = (v) => (typeof v === "string" ? v : (v && v.path) || "");
    if (clip.path) views.push({ id: "src", label: "Source", url: "/" + clip.path.replace(/\\/g, "/").replace(/^\//, "") });
    Object.entries(clip.platform_versions || {}).forEach(([plat, pv]) => {
      const pp = pth(pv);
      if (pp) views.push({ id: "f_" + plat, label: plat + (typeof pv === "object" && pv.aspect_ratio ? ` (${pv.aspect_ratio})` : ""), url: "/" + pp.replace(/\\/g, "/").replace(/^\//, "") });
    });
    const stem = (clip.path || clip.filename || "").split(/[\\/]/).pop().replace(/\.mp4$/, "");
    (burnedByStem[stem] || []).forEach(({ lang, url: bUrl }) => {
      views.push({ id: `burned_${lang}`, label: `Burned ${lang}`, url: bUrl });
    });

    const links = [];
    Object.entries(clip.subtitles || {}).forEach(([lang, ld]) => Object.entries(ld.formats || {}).forEach(([ext, p]) => {
      const sp = pth(p);
      if (sp) links.push({ label: `.${ext} ${lang}`, url: "/" + sp.replace(/\\/g, "/").replace(/^\//, "") });
    }));
    Object.entries(clip.audio_translations || {}).forEach(([lang, ld]) => {
      if (ld && ld.video_path) links.push({ label: `dub ${lang}`, url: "/" + ld.video_path.replace(/\\/g, "/").replace(/^\//, "") });
    });
    (clip.thumbnails || []).forEach((t, i) => { const tp = pth(t); if (tp) links.push({ label: `thumb ${i + 1}`, url: "/" + tp.replace(/\\/g, "/").replace(/^\//, "") }); });

    if (!views.length) {
      videoBox.innerHTML = `<div style="height:100%;display:flex;align-items:center;justify-content:center;color:var(--text-faint)">no preview in demo mode</div>`;
    } else {
      const pick = (v) => {
        tabsEl.querySelectorAll(".g-tab").forEach((t) => t.classList.toggle("on", t.dataset.id === v.id));
        videoBox.innerHTML = `<video src="${esc(v.url)}" controls preload="metadata"></video>`;
      };
      views.forEach((v) => {
        const t = el("button", "g-tab" + (v.id === "src" ? " on" : ""), esc(v.label));
        t.dataset.id = v.id;
        t.addEventListener("click", () => pick(v));
        tabsEl.appendChild(t);
      });
      pick(views[0]);
    }

    if (links.length) {
      contentBox.appendChild(el("div", "g-links", links.map((l) => `<a class="g-link" href="${esc(l.url)}" download>⬇ ${esc(l.label)}</a>`).join("")));
    }
    const caption = rev.caption || (clip.social_media ? "social copy saved — see output/social/" : "");
    if (caption) {
      const c = el("div", "g-caption");
      c.textContent = caption + (rev.hashtags?.length ? "\n" + rev.hashtags.join(" ") : "");
      contentBox.appendChild(c);
    }
  });
  if (!clips.length) grid.innerHTML = `<div class="muted">No approved clips to show.</div>`;
}

/* ───────────────────────── navigation ───────────────────────── */
function showView(name) {
  $$(".view").forEach((v) => v.classList.add("hidden"));
  $("#view-" + name).classList.remove("hidden");
  $$(".step").forEach((s) => s.classList.toggle("active", s.dataset.view === name));
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function enableStep(name, on) {
  const s = $(`.step[data-view="${name}"]`);
  s.disabled = !on;
  s.classList.toggle("done", on);
}

$$(".step").forEach((s) => s.addEventListener("click", () => {
  if (s.disabled) return;
  const v = s.dataset.view;
  if (v === "gallery" && !state.manifest) return;
  showView(v);
}));
$("#new-run-btn").addEventListener("click", () => {
  if (state.pollTimer) clearInterval(state.pollTimer);
  $("#run-error").textContent = "";
  showView("setup");
});

/* ───────────────────────── go ───────────────────────── */
boot();
