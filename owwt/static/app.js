// Wake Word Trainer web UI. Plain ES module, no build step.

const app = document.getElementById("app");
let system = null;
let job = {};
const jobListeners = new Set();
let cleanups = [];
let lastProjectId = null;

// ---------------------------------------------------------------------------------------------
// Helpers
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const pct = (v, digits = 0) => (v == null ? "n/a" : `${(v * 100).toFixed(digits)}%`);
const lines = (t) => t.split("\n").map((s) => s.trim()).filter(Boolean);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function fmtBytes(b) {
  if (!b) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let i = 0;
  while (b >= 1024 && i < units.length - 1) { b /= 1024; i++; }
  return `${b.toFixed(b >= 10 || i === 0 ? 0 : 1)} ${units[i]}`;
}
function fmtAgo(ts) {
  const s = Date.now() / 1000 - ts;
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.round(s / 60)} min ago`;
  if (s < 86400) return `${Math.round(s / 3600)} h ago`;
  return new Date(ts * 1000).toLocaleDateString();
}
function fmtDuration(sec) {
  sec = Math.max(0, Math.round(sec || 0));
  const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = sec % 60;
  return h ? `${h}h ${m}m` : m ? `${m}m ${s}s` : `${s}s`;
}
function toast(message, kind = "info", ms = 4500) {
  const el = document.createElement("div");
  el.className = `toast ${kind}`;
  el.textContent = message;
  $("#toasts").appendChild(el);
  setTimeout(() => el.remove(), ms);
}
function every(ms, fn) {
  const id = setInterval(fn, ms);
  cleanups.push(() => clearInterval(id));
}
async function api(path, opts = {}) {
  const init = { method: opts.method || (opts.body !== undefined ? "POST" : "GET"), headers: {} };
  if (opts.body instanceof FormData) init.body = opts.body;
  else if (opts.body !== undefined) {
    init.body = JSON.stringify(opts.body);
    init.headers["Content-Type"] = "application/json";
  }
  const res = await fetch(path, init);
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try { msg = (await res.json()).detail || msg; } catch {}
    throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
  }
  return (res.headers.get("content-type") || "").includes("json") ? res.json() : res;
}

const svg = (d) => `<svg class="i" viewBox="0 0 24 24" aria-hidden="true">${d}</svg>`;
const icons = {
  play: svg('<path d="M8 5v14l11-7z"/>'),
  stop: svg('<rect x="6" y="6" width="12" height="12" rx="2"/>'),
  mic: svg('<path d="M12 14a3 3 0 0 0 3-3V5a3 3 0 0 0-6 0v6a3 3 0 0 0 3 3zm5-3a5 5 0 0 1-10 0H5a7 7 0 0 0 6 6.92V21h2v-3.08A7 7 0 0 0 19 11z"/>'),
  trash: svg('<path d="M9 3h6l1 2h4v2H4V5h4zM6 9h12l-1 12H7z"/>'),
  download: svg('<path d="M11 4h2v8.2l3.3-3.3 1.4 1.4L12 16l-5.7-5.7 1.4-1.4 3.3 3.3zM5 18h14v2H5z"/>'),
  check: svg('<path d="M9 16.2 4.8 12l-1.4 1.4L9 19 21 7l-1.4-1.4z"/>'),
  upload: svg('<path d="M11 16h2V7.8l3.3 3.3 1.4-1.4L12 4 6.3 9.7l1.4 1.4L11 7.8zM5 18h14v2H5z"/>'),
};

const player = new Audio();
let playerUrl = null;
function playUrl(url) {
  player.src = url;
  player.play().catch((e) => toast(`Can't play audio: ${e.message}`, "error"));
}
function playBlob(blob) {
  if (playerUrl) URL.revokeObjectURL(playerUrl);
  playerUrl = URL.createObjectURL(blob);
  playUrl(playerUrl);
}

const jobFraction = (j) => (j.stages?.length ? (Math.max(0, j.stage) + (j.stage_progress || 0)) / j.stages.length : 0);

function prettyVoice(id) {
  const name = id.split("_")[1] || id;
  return name[0].toUpperCase() + name.slice(1);
}
function voiceLabel(v) {
  if (!v) return "";
  if (typeof v === "string") return `Voice: ${prettyVoice(v)}`;
  const parts = Object.entries(v).sort((a, b) => b[1] - a[1]);
  if (parts.length === 1) return `Voice: ${prettyVoice(parts[0][0])}`;
  return "Blend: " + parts.map(([n, w]) => `${prettyVoice(n)} ${Math.round(w * 100)}%`).join(" + ");
}

// ---------------------------------------------------------------------------------------------
// Microphone: AudioWorklet resamples to 16 kHz Int16 chunks of 80 ms
class Mic {
  constructor() {
    this.listeners = new Set();
    this.level = 0;
    this.stream = null;
  }
  get active() { return !!this.stream; }
  get deviceId() { return this.stream?.getAudioTracks()[0]?.getSettings().deviceId; }
  async start(deviceId) {
    this.stop();
    if (!window.isSecureContext) throw new Error("the page must be opened over HTTPS");
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: { deviceId: deviceId ? { exact: deviceId } : undefined, echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: 1 },
    });
    this.ctx = new AudioContext();
    await this.ctx.audioWorklet.addModule("/static/pcm-worklet.js");
    const source = this.ctx.createMediaStreamSource(this.stream);
    const node = new AudioWorkletNode(this.ctx, "pcm16k", { processorOptions: { inputRate: this.ctx.sampleRate } });
    const mute = this.ctx.createGain();
    mute.gain.value = 0;
    node.port.onmessage = (e) => {
      const pcm = e.data;
      const r = rmsInt16(pcm);
      this.level = Math.max(r, this.level * 0.8);
      for (const fn of this.listeners) fn(pcm, r);
    };
    source.connect(node).connect(mute).connect(this.ctx.destination);
  }
  stop() {
    this.stream?.getTracks().forEach((t) => t.stop());
    this.ctx?.close().catch(() => {});
    this.stream = null;
    this.ctx = null;
    this.level = 0;
  }
  on(fn) {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }
}
const mic = new Mic();

function rmsInt16(pcm) {
  let sum = 0;
  for (let i = 0; i < pcm.length; i++) sum += pcm[i] * pcm[i];
  return Math.sqrt(sum / pcm.length) / 32768;
}
function concatPcm(chunks) {
  const out = new Int16Array(chunks.reduce((n, c) => n + c.length, 0));
  let o = 0;
  for (const c of chunks) { out.set(c, o); o += c.length; }
  return out;
}
function wavEncode(pcm, rate = 16000) {
  const buf = new ArrayBuffer(44 + pcm.length * 2);
  const v = new DataView(buf);
  const str = (o, s) => [...s].forEach((c, i) => v.setUint8(o + i, c.charCodeAt(0)));
  str(0, "RIFF"); v.setUint32(4, 36 + pcm.length * 2, true); str(8, "WAVE"); str(12, "fmt ");
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true); v.setUint32(24, rate, true);
  v.setUint32(28, rate * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true); str(36, "data");
  v.setUint32(40, pcm.length * 2, true);
  new Int16Array(buf, 44).set(pcm);
  return buf;
}
const levelPercent = (level) => Math.max(0, Math.min(100, ((20 * Math.log10(level + 1e-9) + 60) / 60) * 100));

// Records one clip. With fixedSeconds it records exactly that long, otherwise it waits for speech
// and stops after a short silence.
function recordClip({ fixedSeconds = 0, maxSeconds = 4, waitSeconds = 6, onState = () => {}, stopper = () => {} }) {
  return new Promise((resolve) => {
    const pre = [], chunks = [];
    let t = 0, speechAt = -1, lastLoud = 0, noise = null, finished = false, off = null;
    const finish = (ok = true) => {
      if (finished) return;
      finished = true;
      off?.();
      resolve(ok && chunks.length ? concatPcm(chunks) : null);
    };
    stopper(() => finish(true));
    onState({ phase: fixedSeconds ? "fixed" : "waiting", t: 0 });
    off = mic.on((pcm, level) => {
      t += pcm.length / 16000;
      if (fixedSeconds) {
        chunks.push(pcm);
        onState({ phase: "fixed", t });
        if (t >= fixedSeconds) finish();
        return;
      }
      if (speechAt < 0) {
        noise = noise === null ? level : level < noise ? level : noise * 0.97 + level * 0.03;
        pre.push(pcm);
        if (pre.length > 5) pre.shift();
        if (t > 0.25 && level > Math.max(noise * 3.5, 0.004)) {
          speechAt = t;
          lastLoud = t;
          chunks.push(...pre);
          onState({ phase: "speech", t });
        } else if (t > waitSeconds) finish(false);
        return;
      }
      chunks.push(pcm);
      if (level > Math.max(noise * 2.5, 0.003)) lastLoud = t;
      if (t - lastLoud > 0.8 || t - speechAt > maxSeconds) finish();
    });
  });
}

async function uploadClips(pid, kind, files) {
  const fd = new FormData();
  files.forEach((f, i) => fd.append("files", f, f.name || `clip-${i}.wav`));
  const { items } = await api(`/api/projects/${pid}/recordings/${kind}`, { body: fd });
  for (const it of items) {
    if (it.error) toast(`${it.filename}: ${it.error}`, "error", 7000);
    else if (it.warnings?.length) toast(it.warnings.join(" "), "warn", 7000);
  }
  return items.filter((i) => !i.error);
}

// ---------------------------------------------------------------------------------------------
// Global state: system info and the background job
async function refreshSystem() {
  try {
    system = await api("/api/system");
    renderChrome();
  } catch {}
  return system;
}
function notifyJob() {
  renderChrome();
  jobListeners.forEach((fn) => fn(job));
}
async function pollJobNow() {
  try {
    job = await api("/api/job");
    notifyJob();
  } catch {}
}
async function pollJobLoop() {
  try {
    const next = await api("/api/job");
    const changed = JSON.stringify(next) !== JSON.stringify(job);
    job = next;
    if (changed) notifyJob();
  } catch {}
  setTimeout(pollJobLoop, job.running ? 1500 : 4000);
}
function onJob(fn) {
  jobListeners.add(fn);
  cleanups.push(() => jobListeners.delete(fn));
  fn(job);
}

function renderChrome() {
  const chip = $("#gpu-chip");
  if (system) {
    const gpu = system.gpu || {};
    chip.textContent = gpu.available ? gpu.name.replace(/^NVIDIA /, "") : "CPU only";
    chip.className = `chip ${gpu.available ? "ok" : "warn"}`;
    chip.title = gpu.available ? "GPU available for training" : gpu.detail || "No GPU";
  }
  const pill = $("#job-pill");
  if (job.running) {
    pill.hidden = false;
    pill.href = job.project_id ? `#/p/${job.project_id}/train` : "#/setup";
    pill.innerHTML = `<span class="spin"></span>${esc(job.title)} · ${pct(jobFraction(job))}`;
  } else pill.hidden = true;
  const banner = $("#https-banner");
  if (!window.isSecureContext && system) {
    const url = `https://${location.hostname}:${system.https_port}/${location.hash}`;
    banner.hidden = false;
    banner.innerHTML = `Recording from the microphone needs a secure page. <a href="${esc(url)}">Open the HTTPS version</a> and accept the self-signed certificate warning once.`;
  }
}

// ---------------------------------------------------------------------------------------------
// Router
async function route() {
  cleanups.forEach((fn) => { try { fn(); } catch {} });
  cleanups = [];
  const hash = location.hash || "#/";
  const pm = hash.match(/^#\/p\/([a-z0-9_]+)(?:\/(\w+))?/);
  if (!pm || pm[1] !== lastProjectId) mic.stop();
  lastProjectId = pm?.[1] || null;
  $$("[data-nav]").forEach((a) => a.classList.toggle("active", (a.dataset.nav === "setup") === (hash === "#/setup")));
  window.scrollTo(0, 0);
  try {
    if (pm) await renderProject(pm[1], pm[2]);
    else if (hash === "#/new") await renderNew();
    else if (hash === "#/setup") await renderSetup();
    else await renderHome();
  } catch (e) {
    app.innerHTML = `<div class="panel"><h2>Something went wrong</h2><p class="bad-text">${esc(e.message)}</p><a class="btn" href="#/">Back to wake words</a></div>`;
  }
}

const PRESET_TIME = { quick: "About 10 minutes with a GPU", standard: "About 35 minutes with a GPU", max: "About 1.5 hours with a GPU" };

function presetCards(selected) {
  return `<div class="preset-grid">${Object.entries(system.presets).map(([id, p]) => `
    <label class="preset ${id === selected ? "selected" : ""}">
      <input type="radio" name="preset" value="${id}" ${id === selected ? "checked" : ""}>
      <div class="preset-title">${esc(p.label)}</div>
      <div class="muted small">${esc(p.description)}</div>
      <div class="preset-meta">${p.n_positive.toLocaleString()} generated clips · ${p.steps.toLocaleString()} steps</div>
      <div class="preset-meta">${PRESET_TIME[id] || ""}${system.gpu?.available ? "" : " (several times longer on CPU)"}</div>
    </label>`).join("")}</div>`;
}
function bindPresetCards(root, onChange) {
  $$('input[name="preset"]', root).forEach((input) => {
    input.onchange = () => {
      $$(".preset", root).forEach((l) => l.classList.toggle("selected", l.querySelector("input").checked));
      onChange?.(input.value);
    };
  });
}
const selectedPreset = (root) => $('input[name="preset"]:checked', root)?.value || "standard";

async function previewPhrase(body, infoEl, btn) {
  if (!body.text?.trim() && !body.phonemes?.trim()) return toast("Type the wake word first", "warn");
  const old = btn.innerHTML;
  btn.disabled = true;
  btn.innerHTML = `<span class="spin"></span> Generating`;
  infoEl.textContent = "The first preview loads Kokoro and can take a few seconds…";
  try {
    const res = await api("/api/tts/preview", { body });
    const blob = await res.blob();
    const phonemes = decodeURIComponent(res.headers.get("X-Phonemes") || "");
    const voice = JSON.parse(decodeURIComponent(res.headers.get("X-Voice") || "%22%22"));
    const unknown = decodeURIComponent(res.headers.get("X-Unknown") || "");
    infoEl.innerHTML = `${esc(voiceLabel(voice))} · <code>${esc(phonemes)}</code>${unknown ? ` <span class="warn-text">Ignored unknown symbols: ${esc(unknown)}</span>` : ""}`;
    playBlob(blob);
  } catch (e) {
    infoEl.textContent = "";
    toast(e.message, "error");
  } finally {
    btn.disabled = false;
    btn.innerHTML = old;
  }
}

// ---------------------------------------------------------------------------------------------
// Home
async function renderHome() {
  const [projects] = await Promise.all([api("/api/projects"), system ? null : refreshSystem()]);
  app.innerHTML = `
    <section class="hero">
      <div>
        <h1>Wake words</h1>
        <p class="muted">Train your own wake word for Home Assistant and openWakeWord. Kokoro speaks it in thousands of voices, your own recordings teach it how you sound, and training runs right here.</p>
      </div>
      <a class="btn primary big" href="#/new">+ New wake word</a>
    </section>
    ${system && !system.assets_ready ? `
      <div class="panel callout">
        <div><b>One-time download: training data (about 19 GB)</b>
        <div class="muted small">Negative speech, background sounds and room acoustics shared by every wake word. Start it now, or it downloads automatically on your first training run.</div></div>
        <a class="btn" href="#/setup">View downloads</a>
      </div>` : ""}
    ${projects.length ? `<div class="grid">${projects.map(projectCard).join("")}</div>` : `
      <div class="panel">
        <h2>How it works</h2>
        <div class="empty-steps">
          <div class="empty-step"><span class="num">1</span><b>Name it</b><span class="muted small">Type the phrase and listen to how it's pronounced.</span></div>
          <div class="empty-step"><span class="num">2</span><b>Record yourself</b><span class="muted small">Optional, but 20 or more clips of your voice make a big difference.</span></div>
          <div class="empty-step"><span class="num">3</span><b>Train</b><span class="muted small">One click. Voices, noise and reverb are generated automatically.</span></div>
          <div class="empty-step"><span class="num">4</span><b>Test and download</b><span class="muted small">Try it live in the browser, then copy the .tflite into Home Assistant.</span></div>
        </div>
      </div>`}`;
  onJob(() => projects.forEach((p) => {
    const el = $(`[data-status="${p.id}"]`);
    if (el) el.innerHTML = projectStatus(p);
  }));
}
function projectStatus(p) {
  if (job.running && job.project_id === p.id) return `<span class="badge run"><span class="spin"></span> Training ${pct(jobFraction(job))}</span>`;
  if (p.latest_run) return `<span class="badge ok">${icons.check} Trained ${fmtAgo(p.latest_run.created_at)}</span>`;
  return `<span class="badge">Not trained yet</span>`;
}
function projectCard(p) {
  const counts = p.recording_counts || {};
  const hl = p.latest_run?.headline;
  return `<a class="card" href="#/p/${p.id}">
    <div class="phrase">“${esc(p.phrase)}”</div>
    <div class="muted small"><code>${esc(p.model_name)}</code> · ${esc(system?.presets?.[p.preset]?.label || p.preset)}</div>
    <div class="card-stats">
      <span>${icons.mic} ${counts.positive || 0} recordings</span>
      ${hl && hl.recall_tts != null ? `<span>${pct(hl.recall_tts)} detected${hl.fp_per_hour != null ? ` · ${hl.fp_per_hour.toFixed(2)} false/h` : ""}</span>` : ""}
    </div>
    <div data-status="${p.id}">${projectStatus(p)}</div>
  </a>`;
}

// ---------------------------------------------------------------------------------------------
// New wake word
async function renderNew() {
  if (!system) await refreshSystem();
  app.innerHTML = `<div class="narrow">
    <a class="back" href="#/">← Wake words</a>
    <h1>New wake word</h1>
    <p class="muted">You can change everything later.</p>
    <div class="panel">
      <label class="label" for="phrase">What should it listen for?</label>
      <input id="phrase" class="input xl" placeholder="hey jarvis" maxlength="60" autocomplete="off" spellcheck="false">
      <p class="hint">Phrases with three or more syllables work best ("hey jarvis", "okay computer"). Short single words trigger by accident more often.</p>
      <div class="row gap wrap mt">
        <button class="btn" id="hear">${icons.play} Hear it</button>
        <button class="btn ghost" id="hear-random">${icons.play} Random voice</button>
      </div>
      <div id="hear-info" class="muted small mt"></div>
    </div>
    <div class="panel">
      <div class="label">Training quality</div>
      ${presetCards("standard")}
    </div>
    <button class="btn primary big wide" id="create">Create wake word</button>
  </div>`;
  const phrase = $("#phrase");
  phrase.focus();
  bindPresetCards(app);
  $("#hear").onclick = () => previewPhrase({ text: phrase.value, voice: "af_heart" }, $("#hear-info"), $("#hear"));
  $("#hear-random").onclick = () => previewPhrase({ text: phrase.value, random_voice: true }, $("#hear-info"), $("#hear-random"));
  $("#create").onclick = async () => {
    try {
      const p = await api("/api/projects", { body: { phrase: phrase.value, preset: selectedPreset(app) } });
      location.hash = `#/p/${p.id}/record`;
    } catch (e) {
      toast(e.message, "error");
    }
  };
  phrase.onkeydown = (e) => { if (e.key === "Enter") $("#create").click(); };
}

// ---------------------------------------------------------------------------------------------
// Project
const TABS = [["pronounce", "Pronunciation"], ["record", "Record"], ["train", "Train"], ["test", "Test"], ["download", "Download"]];

async function renderProject(id, tab) {
  const [project] = await Promise.all([api(`/api/projects/${id}`), system ? null : refreshSystem()]);
  if (!TABS.some((t) => t[0] === tab)) tab = project.runs.length ? "test" : "record";
  const c = project.recording_counts;
  const done = { record: c.positive >= 20, train: project.runs.length > 0 };
  app.innerHTML = `
    <div class="project-head">
      <a class="back" href="#/">← Wake words</a>
      <div class="row between wrap">
        <div>
          <h1>“${esc(project.phrase)}”</h1>
          <div class="muted small">Model file <code>${esc(project.model_name)}.tflite</code> · ${esc(system.presets[project.preset]?.label || project.preset)} quality</div>
        </div>
        <div id="head-status"></div>
      </div>
      <nav class="tabs">${TABS.map(([t, label], i) => `
        <a href="#/p/${id}/${t}" class="${t === tab ? "active" : ""}"><span class="step ${done[t] && t !== tab ? "done" : ""}">${done[t] && t !== tab ? icons.check : i + 1}</span>${label}</a>`).join("")}
      </nav>
    </div>
    <div id="tab"></div>`;
  onJob((j) => {
    const el = $("#head-status");
    if (el) el.innerHTML = j.running && j.project_id === id ? `<a class="badge run" href="#/p/${id}/train"><span class="spin"></span> Training ${pct(jobFraction(j))}</a>` : "";
  });
  const views = { pronounce: tabPronounce, record: tabRecord, train: tabTrain, test: tabTest, download: tabDownload };
  await views[tab](project, $("#tab"));
}

// Pronunciation ------------------------------------------------------------------------------
async function tabPronounce(project, el) {
  const voices = await api("/api/voices");
  el.innerHTML = `
    <div class="panel">
      <h2>How it sounds</h2>
      <p class="muted">Training clips are spoken by Kokoro, blending ${voices.length} voices into thousands of different speakers. Check it says your wake word the way you do.</p>
      <div class="field"><label class="label" for="phrase">Wake word</label><input id="phrase" class="input" value="${esc(project.phrase)}" maxlength="60"></div>
      <div class="row gap wrap">
        <select id="voice" class="input">${voices.map((v) => `<option value="${v.id}" ${v.id === "af_heart" ? "selected" : ""}>${esc(v.label)}</option>`).join("")}</select>
        <label class="row gap small">Speed <input type="range" id="speed" min="0.7" max="1.3" step="0.05" value="1"></label>
        <button class="btn" id="play">${icons.play} Play</button>
        <button class="btn ghost" id="play-random">${icons.play} Random training voice</button>
      </div>
      <div id="play-info" class="muted small mt"></div>
      <div class="phonemes mt">
        <div class="muted small">Automatic pronunciation</div>
        <div class="row gap wrap"><span class="small">US</span><code id="auto-us">…</code><span class="small">UK</span><code id="auto-gb">…</code>
        <button class="btn ghost small" id="copy-auto">Edit these</button></div>
      </div>
      <div class="field mt">
        <label class="label" for="pron">Custom pronunciation <span class="muted">(optional)</span></label>
        <input id="pron" class="input mono" value="${esc(project.pronunciation)}" placeholder="Leave empty to use the automatic pronunciation" spellcheck="false">
        <p class="hint">Only needed if it sounds wrong. Press "Edit these", change the sounds, then Play to check. ˈ goes before the stressed syllable. Leave empty to go back to automatic.</p>
      </div>
      <details>
        <summary>Other ways to write it</summary>
        <textarea id="variants" class="input" rows="3" placeholder="hey, jarvis">${esc(project.variants.join("\n"))}</textarea>
        <p class="hint">One per line. Every spelling is used when generating clips, which helps with names that can be read more than one way.</p>
      </details>
    </div>
    <div class="panel">
      <h2>Phrases that should not wake it</h2>
      <p class="muted">Similar-sounding phrases are generated automatically, for example:</p>
      <div id="suggestions" class="chips"><span class="muted small"><span class="spin"></span> Finding similar words…</span></div>
      <label class="label mt" for="negatives">Add your own, one per line</label>
      <textarea id="negatives" class="input" rows="4" placeholder="Names of people or pets, phrases from shows you watch">${esc(project.custom_negatives.join("\n"))}</textarea>
    </div>
    <div class="savebar">
      <button class="btn primary" id="save">Save changes</button>
      <span class="hint">Changing the phrase or pronunciation regenerates the voice clips on the next training run.</span>
    </div>`;

  const phrase = $("#phrase", el), pron = $("#pron", el);
  const loadAuto = async () => {
    try {
      const r = await api("/api/tts/phonemize", { body: { text: phrase.value } });
      $("#auto-us", el).textContent = r.american || "(nothing)";
      $("#auto-gb", el).textContent = r.british || "(nothing)";
    } catch (e) {
      $("#auto-us", el).textContent = "error";
      toast(e.message, "error");
    }
  };
  loadAuto();
  phrase.addEventListener("change", loadAuto);
  const speed = () => +$("#speed", el).value;
  $("#play", el).onclick = () => previewPhrase({ text: phrase.value, phonemes: pron.value, voice: $("#voice", el).value, speed: speed() }, $("#play-info", el), $("#play", el));
  $("#play-random", el).onclick = () => previewPhrase({ text: phrase.value, phonemes: pron.value, random_voice: true, speed: speed() }, $("#play-info", el), $("#play-random", el));
  $("#copy-auto", el).onclick = () => { pron.value = $("#auto-us", el).textContent; pron.focus(); };
  api(`/api/projects/${project.id}/suggestions`)
    .then((list) => { const s = $("#suggestions", el); if (s) s.innerHTML = list.map((x) => `<span class="chip">${esc(x)}</span>`).join("") || `<span class="muted small">None found</span>`; })
    .catch(() => {});
  $("#save", el).onclick = async () => {
    try {
      await api(`/api/projects/${project.id}`, {
        method: "PATCH",
        body: { phrase: phrase.value, pronunciation: pron.value, variants: lines($("#variants", el).value), custom_negatives: lines($("#negatives", el).value) },
      });
      toast("Saved", "ok");
      route();
    } catch (e) {
      toast(e.message, "error");
    }
  };
}

// Recording ----------------------------------------------------------------------------------
const POSITIVE_STYLES = [
  "in your normal voice", "a little faster", "slowly and clearly", "quietly, like someone is asleep",
  "louder, as if from across the room", "turned away from the microphone", "casually, like you say it every day",
  "as a question", "with a smile", "from a few steps further away", "with music or the TV on", "a bit lower pitched",
];
const RECORD_KINDS = {
  positive: { label: "Wake word", target: 20, say: "Say", hint: "Press Record, say the wake word, and it stops by itself when you finish.", maxSeconds: 4 },
  negative: { label: "Other speech", target: 10, say: "Say something that is not the wake word", hint: "Similar-sounding phrases and normal talk. These teach the model what to ignore.", maxSeconds: 6 },
  background: { label: "Background noise", target: 3, say: "Record the sound of the room", hint: "Leave the TV, music, fan or kitchen noise running and stay quiet for 30 seconds.", fixedSeconds: 30 },
};
let recordKind = "positive";
let styleIndex = 0;

async function tabRecord(project, el) {
  el.innerHTML = `
    <div class="panel mic-panel">
      <div class="row gap wrap">
        <button class="btn primary" id="mic-on">${icons.mic} Turn on microphone</button>
        <select id="mic-select" class="input" hidden></select>
        <div class="meter" title="Input level"><div id="meter"></div></div>
      </div>
      <p class="hint">Use a microphone and distance similar to your voice assistant. Browser noise suppression is turned off on purpose so clips sound like what the device hears.</p>
    </div>
    <div class="seg" id="kind-tabs"></div>
    <div class="two-col">
      <div>
        <div class="panel record-panel" id="rec-panel"></div>
        <div class="panel">
          <h3>Tips for a great model</h3>
          <ul class="tips">
            <li>20 to 50 wake word clips is plenty. Vary your tone, speed and distance.</li>
            <li>Record in the rooms where you'll use it, and get everyone in the house to add a few.</li>
            <li>After training, use the Test tab: it can save misses and false triggers straight into these lists.</li>
          </ul>
        </div>
      </div>
      <div class="panel">
        <div class="row between"><h3 id="list-title"></h3>
          <label class="btn ghost small">${icons.upload} Upload<input type="file" id="upload" accept="audio/*,video/webm" multiple hidden></label>
        </div>
        <div id="clips" class="clip-list"></div>
      </div>
    </div>`;

  let recs = project.recordings;
  let suggestions = [];
  let suggestionIndex = 0;
  let scores = {};
  let threshold = 0.5;
  let busy = false, stopHandsFree = false, stopCurrent = null;

  api(`/api/projects/${project.id}/runs`).then((runs) => {
    if (!runs.length) return;
    threshold = runs[0].recommended_threshold ?? 0.5;
    for (const r of runs[0].recordings || []) scores[`${r.kind}/${r.name}`] = r.max_score;
    renderList();
  }).catch(() => {});
  api(`/api/projects/${project.id}/suggestions`).then((s) => { suggestions = s; if (recordKind === "negative" && !busy) renderPanel(); }).catch(() => {});

  // Microphone
  const micOn = $("#mic-on", el), select = $("#mic-select", el);
  async function fillDevices() {
    const devices = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "audioinput");
    select.innerHTML = devices.map((d, i) => `<option value="${esc(d.deviceId)}">${esc(d.label || `Microphone ${i + 1}`)}</option>`).join("");
    select.value = mic.deviceId || "";
    select.hidden = devices.length < 2;
  }
  async function startMic(deviceId) {
    try {
      await mic.start(deviceId);
    } catch (e) {
      toast(`Microphone unavailable: ${e.message}`, "error", 8000);
      return false;
    }
    micOn.hidden = true;
    await fillDevices();
    return true;
  }
  micOn.onclick = () => startMic();
  select.onchange = () => startMic(select.value);
  if (mic.active) { micOn.hidden = true; fillDevices(); }
  let frame;
  const meter = () => {
    const m = $("#meter", el);
    if (m) m.style.width = `${levelPercent(mic.level)}%`;
    frame = requestAnimationFrame(meter);
  };
  meter();
  cleanups.push(() => { cancelAnimationFrame(frame); stopHandsFree = true; stopCurrent?.(); });

  function renderTabs() {
    $("#kind-tabs", el).innerHTML = Object.entries(RECORD_KINDS).map(([k, d]) =>
      `<button data-kind="${k}" class="${k === recordKind ? "active" : ""}">${d.label}<span class="count">${recs[k].length}</span></button>`).join("");
    $$("#kind-tabs button", el).forEach((b) => b.onclick = () => {
      if (busy) return toast("Stop recording first", "warn");
      recordKind = b.dataset.kind;
      renderTabs(); renderPanel(); renderList();
    });
  }
  function promptText() {
    if (recordKind === "positive") return `“${esc(project.phrase)}”`;
    if (recordKind === "negative") return suggestions.length ? `“${esc(suggestions[suggestionIndex % suggestions.length])}”` : "anything else";
    return "";
  }
  function renderGoal() {
    const d = RECORD_KINDS[recordKind], n = recs[recordKind].length;
    const goal = $("#goal", el);
    if (goal) goal.innerHTML = `<div class="bar"><div style="width:${Math.min(100, (n / d.target) * 100)}%"></div></div>
      <span class="small muted">${n} of ${d.target}+ recommended${recordKind === "positive" ? ". Optional, but it makes the model much better at hearing you." : ""}</span>`;
  }
  function renderPanel() {
    const d = RECORD_KINDS[recordKind];
    $("#rec-panel", el).innerHTML = `
      <div class="prompt">
        <div class="muted small">${d.say}</div>
        <div class="prompt-text" id="prompt-text">${promptText()}</div>
        <div class="prompt-style" id="prompt-style">${recordKind === "positive" ? esc(POSITIVE_STYLES[styleIndex % POSITIVE_STYLES.length]) : ""}</div>
      </div>
      <div class="rec-row">
        <button class="rec-btn" id="rec">${icons.mic} ${d.fixedSeconds ? `Record ${d.fixedSeconds} s` : "Record"}</button>
        ${d.fixedSeconds ? "" : `<button class="btn" id="auto">Hands-free × 10</button>`}
        ${recordKind === "negative" ? `<button class="btn ghost" id="next">Next suggestion</button>` : ""}
      </div>
      <div class="rec-status" id="rec-status">${d.hint}</div>
      <div class="goal" id="goal"></div>`;
    renderGoal();
    $("#rec", el).onclick = () => (busy ? (stopHandsFree = true, stopCurrent?.()) : recordOnce());
    const auto = $("#auto", el);
    if (auto) auto.onclick = () => (busy ? (stopHandsFree = true, stopCurrent?.()) : handsFree());
    const next = $("#next", el);
    if (next) next.onclick = () => { suggestionIndex++; $("#prompt-text", el).innerHTML = promptText(); };
  }
  function setBusy(on, handsFreeMode = false) {
    busy = on;
    const rec = $("#rec", el), auto = $("#auto", el);
    rec.classList.toggle("recording", on);
    const d = RECORD_KINDS[recordKind];
    rec.innerHTML = on ? `${icons.stop} Stop` : `${icons.mic} ${d.fixedSeconds ? `Record ${d.fixedSeconds} s` : "Record"}`;
    if (auto) { auto.disabled = on && !handsFreeMode; auto.textContent = on && handsFreeMode ? "Stop hands-free" : "Hands-free × 10"; }
  }
  async function recordOnce(handsFreeMode = false, label = "") {
    if (!mic.active && !(await startMic())) return null;
    const d = RECORD_KINDS[recordKind];
    const kind = recordKind;
    const status = $("#rec-status", el);
    setBusy(true, handsFreeMode);
    const pcm = await recordClip({
      fixedSeconds: d.fixedSeconds, maxSeconds: d.maxSeconds,
      stopper: (fn) => (stopCurrent = fn),
      onState: (s) => {
        if (s.phase === "waiting") status.innerHTML = `<span class="dot"></span> ${label}Listening… speak now`;
        else if (s.phase === "speech") status.innerHTML = `<span class="dot live"></span> ${label}Recording`;
        else status.innerHTML = `<span class="dot live"></span> Recording, ${Math.max(0, Math.ceil(d.fixedSeconds - s.t))} s left`;
      },
    });
    stopCurrent = null;
    if (!handsFreeMode) setBusy(false);
    if (!pcm) {
      status.textContent = "Didn't hear anything. Check the level meter moves when you talk.";
      return null;
    }
    status.textContent = "Saving…";
    try {
      const saved = await uploadClips(project.id, kind, [new File([wavEncode(pcm)], "clip.wav", { type: "audio/wav" })]);
      if (!saved.length) { status.textContent = "That clip was not saved."; return null; }
      recs = await api(`/api/projects/${project.id}/recordings`);
      status.innerHTML = `${icons.check} Saved (${saved[0].duration.toFixed(1)} s)`;
      if (kind === "positive") styleIndex++;
      if (kind === "negative") suggestionIndex++;
      const style = $("#prompt-style", el), text = $("#prompt-text", el);
      if (style && kind === "positive") style.textContent = POSITIVE_STYLES[styleIndex % POSITIVE_STYLES.length];
      if (text) text.innerHTML = promptText();
      renderTabs(); renderGoal(); renderList();
      return saved;
    } catch (e) {
      status.textContent = e.message;
      return null;
    }
  }
  async function handsFree() {
    stopHandsFree = false;
    for (let i = 1; i <= 10 && !stopHandsFree; i++) {
      const saved = await recordOnce(true, `Clip ${i} of 10. `);
      if (!saved || stopHandsFree) break;
      await sleep(600);
    }
    setBusy(false);
  }

  function renderList() {
    const items = recs[recordKind];
    const title = $("#list-title", el);
    if (!title) return;
    title.textContent = `${RECORD_KINDS[recordKind].label} (${items.length})`;
    $("#clips", el).innerHTML = items.length ? items.map((it) => {
      const sc = scores[`${recordKind}/${it.name}`];
      const good = sc == null ? null : recordKind === "positive" ? sc >= threshold : sc < threshold;
      return `<div class="clip">
        <button class="icon-btn" data-play="${esc(it.name)}" title="Play">${icons.play}</button>
        <div class="clip-main"><div class="small">${fmtAgo(it.created_at)}</div><div class="muted xsmall">${it.duration.toFixed(1)} s</div></div>
        ${sc != null && recordKind !== "background" ? `<span class="score ${good ? "good" : "bad"}" title="Highest score from the latest model">${sc.toFixed(2)}</span>` : ""}
        <button class="icon-btn danger" data-del="${esc(it.name)}" title="Delete">${icons.trash}</button>
      </div>`;
    }).join("") : `<div class="empty muted small">No recordings yet</div>`;
  }
  $("#clips", el).onclick = async (e) => {
    const play = e.target.closest("[data-play]"), del = e.target.closest("[data-del]");
    if (play) playUrl(`/api/projects/${project.id}/recordings/${recordKind}/${encodeURIComponent(play.dataset.play)}`);
    if (del) {
      await api(`/api/projects/${project.id}/recordings/${recordKind}/${encodeURIComponent(del.dataset.del)}`, { method: "DELETE" });
      recs = await api(`/api/projects/${project.id}/recordings`);
      renderTabs(); renderGoal(); renderList();
    }
  };
  $("#upload", el).onchange = async (e) => {
    const files = [...e.target.files];
    e.target.value = "";
    if (!files.length) return;
    toast(`Uploading ${files.length} file${files.length > 1 ? "s" : ""}…`);
    try {
      const saved = await uploadClips(project.id, recordKind, files);
      if (saved.length) toast(`Added ${saved.length} recording${saved.length > 1 ? "s" : ""}`, "ok");
      recs = await api(`/api/projects/${project.id}/recordings`);
      renderTabs(); renderGoal(); renderList();
    } catch (err) {
      toast(err.message, "error");
    }
  };

  renderTabs(); renderPanel(); renderList();
}

// Training -----------------------------------------------------------------------------------
const ADVANCED_FIELDS = [
  ["n_positive", "Wake word clips", "Kokoro clips of the wake word"],
  ["n_negative", "Similar phrase clips", "Kokoro clips of phrases that should not trigger"],
  ["steps", "Training steps", "Longer training usually means fewer false activations"],
  ["layer_size", "Model size", "Neurons per layer. 32 is standard, 64 can be more accurate"],
  ["n_blocks", "Hidden blocks", "Extra layers. 1 is standard"],
  ["target_fp_per_hour", "Target false activations per hour", "Used to pick the best checkpoint and threshold"],
  ["max_negative_weight", "Negative weight", "Higher means fewer false activations but more misses"],
  ["real_weight", "Share of your recordings", "Fraction of each batch of wake word examples taken from your recordings (0 to 0.9)"],
  ["micro_steps", "ESPHome training steps", "microWakeWord steps. More can be better but it trains on the CPU"],
  ["micro_target_faph", "ESPHome false accepts per hour target", "Used to choose the probability cutoff written into the JSON"],
];

async function tabTrain(project, el) {
  await refreshSystem();
  const s = system, c = project.recording_counts;
  const adv = { ...project.advanced };
  const totalAssets = s.assets.filter((a) => !a.optional).reduce((n, a) => n + a.size_bytes, 0);
  const types = project.model_types?.length ? project.model_types : ["openwakeword"];
  const microAsset = s.assets.find((a) => a.id === "microwakeword");
  const check = (ok, text, extra = "") => `<li class="${ok ? "ok" : ""}"><span class="ci">${ok ? icons.check : "!"}</span><span>${text}${extra ? ` <span class="muted small">${extra}</span>` : ""}</span></li>`;

  el.innerHTML = `
    <div class="panel">
      <h2>Before you start</h2>
      <ul class="checklist">
        ${check(s.assets_ready, s.assets_ready ? "Training data downloaded" : `Training data downloads first (about ${fmtBytes(totalAssets)}, one time only)`, s.assets_ready ? "" : `<a href="#/setup">Details</a>`)}
        ${check(s.gpu?.available, s.gpu?.available ? `GPU: ${esc(s.gpu.name)}` : "No GPU found. Training runs on the CPU and takes much longer.", s.gpu?.available ? "" : `<a href="#/setup">How to enable a GPU</a>`)}
        ${check(c.positive >= 20, `${c.positive} wake word recordings`, c.positive >= 20 ? "" : `Optional. <a href="#/p/${project.id}/record">Record ${20 - c.positive} more</a> so the model learns your voice.`)}
        ${check(c.negative + c.background >= 5, `${c.negative} other speech and ${c.background} background recordings`, c.negative + c.background >= 5 ? "" : "Optional")}
        ${s.tflite_available ? "" : check(false, "TFLite converter missing, only .onnx will be produced")}
        ${types.includes("microwakeword") ? check(microAsset?.ready, microAsset?.ready ? "ESPHome negative data downloaded" : `ESPHome negative data downloads first (${fmtBytes(microAsset?.size_bytes)}, one time only)`) : ""}
      </ul>
    </div>
    <div class="panel">
      <h2>What to build</h2>
      <div class="type-options">
        <label class="check-card"><input type="checkbox" id="type-oww" ${types.includes("openwakeword") ? "checked" : ""}>
          <div><b>Home Assistant (openWakeWord)</b>
          <div class="muted small">A .tflite for Home Assistant's openWakeWord add-on. Your voice satellite streams audio to Home Assistant, which listens for the wake word. Includes live testing in the browser.</div></div></label>
        <label class="check-card"><input type="checkbox" id="type-micro" ${types.includes("microwakeword") ? "checked" : ""} ${s.micro_available ? "" : "disabled"}>
          <div><b>ESPHome on-device (microWakeWord)</b>
          <div class="muted small">A .tflite and .json for ESPHome's micro_wake_word, so an ESP32-S3 device like the Voice PE hears the wake word itself. Adds a one-time ${fmtBytes(microAsset?.size_bytes)} download, and training runs on the CPU (roughly 30 to 90 minutes).${s.micro_available ? "" : " Not available: the microWakeWord trainer isn't installed."}</div></div></label>
      </div>
    </div>
    <div class="panel">
      <h2>Quality</h2>
      ${presetCards(project.preset)}
      <details class="mt">
        <summary>Advanced settings</summary>
        <div class="adv-grid">
          ${ADVANCED_FIELDS.map(([key, label, help]) => `
            <div><label class="label" for="adv-${key}">${label}</label>
            <input class="input" type="number" step="any" id="adv-${key}" data-key="${key}" value="${adv[key] ?? ""}">
            <div class="hint">${help}</div></div>`).join("")}
          <div><label class="label" for="adv-voices">Voices</label>
            <select class="input" id="adv-voices" data-key="voices">
              <option value="all" ${(adv.voices ?? "all") === "all" ? "selected" : ""}>All voices, including accented English</option>
              <option value="english" ${adv.voices === "english" ? "selected" : ""}>American and British voices only</option>
            </select>
            <div class="hint">Accented voices add variety and help with different speakers</div></div>
        </div>
        <button class="btn ghost small" id="adv-reset">Reset advanced settings</button>
      </details>
      <label class="check mt"><input type="checkbox" id="publish" ${project.publish_to_output ? "checked" : ""}> Copy finished models to the output folder <code>${esc(s.output_dir)}</code></label>
    </div>
    <div id="job-area"></div>`;

  const fillPlaceholders = () => {
    const preset = s.presets[selectedPreset(el)];
    $$(".adv-grid [data-key]", el).forEach((input) => {
      if (input.tagName === "INPUT") input.placeholder = preset[input.dataset.key] ?? s.advanced_defaults[input.dataset.key] ?? "";
    });
  };
  const collect = () => {
    const advanced = {};
    $$(".adv-grid [data-key]", el).forEach((input) => {
      const v = input.value.trim();
      if (v !== "") advanced[input.dataset.key] = input.tagName === "SELECT" ? v : Number(v);
    });
    if (advanced.voices === "all") delete advanced.voices;
    const model_types = [["openwakeword", "#type-oww"], ["microwakeword", "#type-micro"]]
      .filter(([, sel]) => $(sel, el).checked).map(([t]) => t);
    return { preset: selectedPreset(el), advanced, publish_to_output: $("#publish", el).checked, model_types };
  };
  let saveTimer;
  const save = async () => {
    clearTimeout(saveTimer);
    const body = collect();
    Object.assign(project, await api(`/api/projects/${project.id}`, { method: "PATCH", body }));
  };
  const saveSoon = () => { clearTimeout(saveTimer); saveTimer = setTimeout(() => save().catch((e) => toast(e.message, "error")), 500); };
  bindPresetCards(el, () => { fillPlaceholders(); saveSoon(); });
  $$(".adv-grid [data-key]", el).forEach((input) => (input.onchange = saveSoon));
  $("#publish", el).onchange = saveSoon;
  ["#type-oww", "#type-micro"].forEach((sel, i, all) => {
    $(sel, el).onchange = async () => {
      const other = $(all[1 - i], el);
      if (!$(sel, el).checked && !other.checked) {
        $(sel, el).checked = true;
        return toast("Pick at least one kind of model to build", "warn");
      }
      try { await save(); route(); } catch (e) { toast(e.message, "error"); }
    };
  });
  $("#adv-reset", el).onclick = () => { $$(".adv-grid [data-key]", el).forEach((i) => (i.value = i.tagName === "SELECT" ? "all" : "")); saveSoon(); };
  fillPlaceholders();

  async function startTraining() {
    try {
      await save();
      await api(`/api/projects/${project.id}/train`, { body: {} });
      toast("Training started", "ok");
      await pollJobNow();
    } catch (e) {
      toast(e.message, "error", 8000);
    }
  }

  let logOffset = 0;
  async function pollLog(reset = false) {
    const pre = $("#log", el);
    if (!pre) return;
    if (reset) { logOffset = 0; pre.textContent = ""; }
    try {
      const r = await api(`/api/job/log?offset=${logOffset}`);
      if (r.text) {
        const atBottom = pre.scrollTop + pre.clientHeight >= pre.scrollHeight - 40;
        pre.textContent += r.text;
        if (pre.textContent.length > 400000) pre.textContent = pre.textContent.slice(-300000);
        if (atBottom || reset) pre.scrollTop = pre.scrollHeight;
      }
      logOffset = r.offset;
    } catch {}
  }

  function resultBox(j) {
    const hl = j.result?.headline, m = j.result?.micro;
    if (!hl && !m) return "";
    const oww = hl && (!m || hl.threshold !== m.probability_cutoff || hl.recall_real_clips != null);
    return `<div class="tiles">
        ${oww ? tile(pct(hl.recall_tts), "Home Assistant model: test voices detected") : ""}
        ${oww ? tile(hl.fp_per_hour == null ? "n/a" : hl.fp_per_hour.toFixed(2), "Home Assistant model: false activations per hour") : ""}
        ${m ? tile(m.recall == null ? "n/a" : pct(m.recall), "ESPHome model: test voices detected") : ""}
        ${m ? tile(m.faph == null ? "n/a" : m.faph.toFixed(2), "ESPHome model: false accepts per hour") : ""}
      </div>
      <div class="row gap wrap">${oww ? `<a class="btn primary" href="#/p/${project.id}/test">${icons.mic} Test it live</a>` : ""}<a class="btn ${oww ? "" : "primary"}" href="#/p/${project.id}/download">${icons.download} Download</a></div>`;
  }

  function renderJob(j) {
    const area = $("#job-area", el);
    if (!area) return;
    const mine = j.kind === "train" && j.project_id === project.id;
    const other = j.running && !mine;
    const key = `${j.id}|${j.running}|${j.state}|${!!j.stages?.length}`;
    if (area.dataset.render !== key) {
      area.dataset.render = key;
      const title = j.running ? "Training" : { done: "Training finished", failed: "Training failed", cancelled: "Training cancelled", interrupted: "Training interrupted" }[j.state] || "Training";
      area.innerHTML = `
        <div class="train-action">
          ${j.running && mine
            ? `<button class="btn danger" id="cancel">${icons.stop} Cancel training</button>`
            : `<button class="btn primary big" id="start" ${other ? "disabled" : ""}>${project.runs.length || (mine && j.state === "done") ? "Train again" : "Start training"}</button>`}
          ${other ? `<span class="muted">Another job is running: ${esc(j.title)}. <a href="${j.project_id ? `#/p/${j.project_id}/train` : "#/setup"}">View it</a></span>` : ""}
          ${!j.running && project.runs.length ? `<span class="muted small">Generated voices and features are cached, so retraining after adding recordings is much quicker.</span>` : ""}
        </div>
        ${mine && j.stages?.length ? `
          <div class="panel">
            <div class="row between"><h2>${title}</h2><span class="muted small" id="elapsed"></span></div>
            <ol class="stages" id="stages"></ol>
            <div class="bar big"><div id="overall"></div></div>
            <div class="job-msg" id="job-msg"></div>
            ${j.state === "failed" ? `<div class="error-box">${esc(j.error || "Unknown error")}. The log below has the details.</div>` : ""}
            ${j.state === "done" ? resultBox(j) : ""}
            <details class="mt" ${j.running || j.state === "failed" ? "open" : ""}><summary>Log</summary><pre class="log" id="log"></pre></details>
          </div>` : ""}`;
      $("#start", area)?.addEventListener("click", startTraining);
      $("#cancel", area)?.addEventListener("click", async () => {
        if (!confirm("Cancel training? Generated clips are kept and reused next time.")) return;
        await api("/api/job/cancel", { body: {} });
        toast("Cancelling…");
      });
      if (mine && !j.running && j.state === "done") refreshSystem();
      pollLog(true);
    }
    if (mine && j.stages?.length) {
      $("#stages", area).innerHTML = j.stages.map((name, i) => {
        const state = j.state === "done" || i < j.stage ? "done" : i === j.stage ? (j.running ? "active" : j.state) : "";
        return `<li class="${state}"><span class="stage-icon">${state === "done" ? icons.check : i + 1}</span>${esc(name)}${state === "active" ? ` <span class="muted small">${pct(j.stage_progress)}</span>` : ""}</li>`;
      }).join("");
      $("#overall", area).style.width = `${(j.state === "done" ? 1 : jobFraction(j)) * 100}%`;
      $("#job-msg", area).textContent = j.running ? j.message || "" : "";
      const end = j.running ? Date.now() / 1000 : j.finished_at || j.updated_at;
      $("#elapsed", area).textContent = j.started_at ? `Elapsed ${fmtDuration(end - j.started_at)}` : "";
    }
  }
  onJob(renderJob);
  every(1500, () => { if (job.running && job.project_id === project.id) pollLog(); });
}

const tile = (value, label, help = "") => `<div class="tile"><div class="tile-l">${label}</div><div class="tile-v">${value}</div>${help ? `<div class="muted xsmall">${help}</div>` : ""}</div>`;

function emptyPanel(title, text, href, label) {
  return `<div class="panel empty"><h2>${title}</h2><p class="muted">${text}</p>${href ? `<a class="btn primary" href="${href}">${label}</a>` : ""}</div>`;
}

// Live test ----------------------------------------------------------------------------------
async function tabTest(project, el) {
  const allRuns = await api(`/api/projects/${project.id}/runs`);
  const runs = allRuns.filter((r) => r.files.some((f) => f.endsWith(".onnx")));
  if (!runs.length) {
    el.innerHTML = allRuns.length
      ? emptyPanel("Live testing needs the Home Assistant model", "Your models so far were built for ESPHome only, which runs on the device itself. Tick Home Assistant (openWakeWord) on the Train tab to also build a model you can test here.", `#/p/${project.id}/train`, "Go to training")
      : emptyPanel("No model yet", "Train the wake word first, then come back here to try it with your microphone.", `#/p/${project.id}/train`, "Go to training");
    return;
  }
  el.innerHTML = `
    <div class="panel">
      <div class="row between wrap gap">
        <div class="row gap wrap">
          <button class="btn primary" id="listen">${icons.mic} Start listening</button>
          <select id="run" class="input">${runs.map((r, i) => `<option value="${r.run_id}">${i === 0 ? "Latest model" : "Model"}, ${new Date(r.created_at * 1000).toLocaleString()}</option>`).join("")}</select>
        </div>
        <div class="lamp" id="lamp">Not listening</div>
      </div>
      <canvas id="chart" class="chart"></canvas>
      <div class="row gap wrap mt">
        <label class="grow small">Threshold <b id="thr-val"></b><input type="range" id="thr" min="0.05" max="0.95" step="0.01" style="width:100%"></label>
        <span class="small muted">Recommended for this model: <b id="rec-thr"></b></span>
      </div>
      <div class="row gap wrap mt">
        <div class="stat"><div class="muted xsmall">Detections</div><div class="stat-v" id="count">0</div></div>
        <div class="stat"><div class="muted xsmall">Highest score</div><div class="stat-v" id="peak">0.00</div></div>
        <div class="grow"></div>
        <button class="btn" id="missed" disabled>It missed me: save last 4 s</button>
        <button class="btn" id="false" disabled>False trigger: save it</button>
      </div>
      <p class="hint">Saved clips go straight into your recordings: misses as wake word clips, false triggers as other speech. Train again afterwards to fix them (it's quick, generated voices are cached).</p>
    </div>
    <div class="panel"><h2>How your recordings score</h2><div id="rec-scores"></div></div>`;

  const scores = new Array(125).fill(0), marks = new Array(125).fill(false);
  let run = runs[0], threshold = 0.5, ws = null, off = null, detections = 0, peak = 0, lampTimer;
  const thr = $("#thr", el);

  function setRun(r) {
    run = r;
    threshold = r.recommended_threshold ?? 0.5;
    thr.value = threshold;
    $("#thr-val", el).textContent = threshold.toFixed(2);
    $("#rec-thr", el).textContent = threshold.toFixed(2);
    renderScores();
    draw();
  }
  thr.oninput = () => {
    threshold = +thr.value;
    $("#thr-val", el).textContent = threshold.toFixed(2);
    if (ws?.readyState === 1) ws.send(JSON.stringify({ action: "threshold", value: threshold }));
    draw();
  };
  $("#run", el).onchange = () => {
    setRun(runs.find((r) => r.run_id === $("#run", el).value));
    if (ws) { stop(); start(); }
  };

  const lamp = (text, cls = "") => { const l = $("#lamp", el); if (l) { l.textContent = text; l.className = `lamp ${cls}`; } };

  async function start() {
    if (!mic.active) {
      try { await mic.start(); } catch (e) { return toast(`Microphone unavailable: ${e.message}`, "error", 8000); }
    }
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const socket = new WebSocket(`${proto}://${location.host}/api/projects/${project.id}/live?run=${run.run_id}`);
    ws = socket;
    lamp("Loading model…");
    socket.onmessage = (ev) => {
      const m = JSON.parse(ev.data);
      if (m.type === "ready") {
        socket.send(JSON.stringify({ action: "threshold", value: threshold }));
        off = mic.on((pcm) => socket.readyState === 1 && socket.send(pcm));
        lamp("Listening. Say the wake word", "on");
        $("#missed", el).disabled = false;
      } else if (m.type === "score") {
        scores.push(m.score); scores.shift();
        marks.push(m.detected); marks.shift();
        if (m.score > peak) { peak = m.score; $("#peak", el).textContent = peak.toFixed(2); }
        if (m.detected) {
          detections++;
          $("#count", el).textContent = detections;
          $("#false", el).disabled = false;
          lamp("Detected!", "hit");
          clearTimeout(lampTimer);
          lampTimer = setTimeout(() => ws && lamp("Listening. Say the wake word", "on"), 1200);
        }
        draw();
      } else if (m.type === "saved") {
        toast(m.kind === "positive" ? "Saved as a wake word recording" : "Saved as other speech", "ok");
      } else if (m.type === "error") {
        toast(m.message, "error");
        stop();
      }
    };
    socket.onclose = () => { if (ws === socket) stop(); };
    $("#listen", el).innerHTML = `${icons.stop} Stop listening`;
  }
  function stop() {
    off?.(); off = null;
    const socket = ws; ws = null;
    socket?.close();
    lamp("Not listening");
    const b = $("#listen", el);
    if (b) b.innerHTML = `${icons.mic} Start listening`;
  }
  cleanups.push(stop);
  $("#listen", el).onclick = () => (ws ? stop() : start());
  $("#missed", el).onclick = () => ws?.readyState === 1 && ws.send(JSON.stringify({ action: "save", kind: "positive", seconds: 4 }));
  $("#false", el).onclick = () => {
    if (ws?.readyState === 1) ws.send(JSON.stringify({ action: "save", kind: "negative", source: "detection" }));
    $("#false", el).disabled = true;
  };

  function draw() {
    const c = $("#chart", el);
    if (!c) return;
    const dpr = window.devicePixelRatio || 1, w = c.clientWidth, h = c.clientHeight;
    if (c.width !== Math.round(w * dpr)) { c.width = Math.round(w * dpr); c.height = Math.round(h * dpr); }
    const g = c.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, h);
    const css = getComputedStyle(document.documentElement);
    const color = (name) => css.getPropertyValue(name).trim();
    const x = (i) => (i / (scores.length - 1)) * w, y = (v) => h - 8 - v * (h - 16);
    g.strokeStyle = color("--border"); g.lineWidth = 1;
    [0, 0.5, 1].forEach((v) => { g.beginPath(); g.moveTo(0, y(v)); g.lineTo(w, y(v)); g.stroke(); });
    g.beginPath(); g.moveTo(0, y(0));
    scores.forEach((v, i) => g.lineTo(x(i), y(v)));
    g.lineTo(w, y(0)); g.closePath();
    g.fillStyle = color("--accent-soft"); g.fill();
    g.beginPath();
    scores.forEach((v, i) => (i ? g.lineTo(x(i), y(v)) : g.moveTo(x(i), y(v))));
    g.strokeStyle = color("--accent"); g.lineWidth = 2; g.stroke();
    g.setLineDash([6, 4]); g.strokeStyle = color("--warn"); g.lineWidth = 1.5;
    g.beginPath(); g.moveTo(0, y(threshold)); g.lineTo(w, y(threshold)); g.stroke(); g.setLineDash([]);
    g.fillStyle = color("--ok");
    marks.forEach((m, i) => { if (m) { g.beginPath(); g.arc(x(i), y(scores[i]), 5, 0, Math.PI * 2); g.fill(); } });
  }
  window.addEventListener("resize", draw);
  cleanups.push(() => window.removeEventListener("resize", draw));

  function renderScores() {
    const box = $("#rec-scores", el);
    const recs = run.recordings || [];
    if (!recs.length) {
      box.innerHTML = `<p class="muted">This model had no recordings of your voice to score. Add some in the <a href="#/p/${project.id}/record">Record</a> tab and train again.</p>`;
      return;
    }
    const t = run.recommended_threshold ?? 0.5;
    const rows = [...recs].sort((a, b) => a.kind.localeCompare(b.kind) || (a.kind === "positive" ? a.max_score - b.max_score : b.max_score - a.max_score));
    box.innerHTML = `<p class="muted small">The highest score each clip reached. Wake word clips below the orange line would be missed, other speech above it would be a false trigger. Worst clips are listed first. Clips marked "test" were held out of training, so they are an honest check.</p>
      <div class="score-list">${rows.map((r) => {
        const good = r.kind === "positive" ? r.max_score >= t : r.max_score < t;
        return `<div class="score-row">
          <button class="icon-btn" data-kind="${r.kind}" data-name="${esc(r.name)}" title="Play">${icons.play}</button>
          <span class="tag ${r.kind}">${r.kind === "positive" ? "Wake word" : "Other speech"}</span>
          <span class="tag" ${r.held_out ? "" : "hidden"}>test</span>
          <div class="score-bar"><div class="${good ? "good" : "bad"}" style="width:${Math.max(2, r.max_score * 100)}%"></div><i style="left:${t * 100}%"></i></div>
          <b class="${good ? "good-text" : "bad-text"}">${r.max_score.toFixed(2)}</b>
        </div>`;
      }).join("")}</div>`;
    box.onclick = (e) => {
      const b = e.target.closest("[data-name]");
      if (b) playUrl(`/api/projects/${project.id}/recordings/${b.dataset.kind}/${encodeURIComponent(b.dataset.name)}`);
    };
  }
  setRun(run);
}

// Download -----------------------------------------------------------------------------------
async function tabDownload(project, el) {
  const runs = await api(`/api/projects/${project.id}/runs`);
  const fileUrl = (r, f) => `/api/projects/${project.id}/runs/${r.run_id}/files/${encodeURIComponent(f)}`;
  const espUrl = (r, f) => `/api/projects/${project.id}/runs/${r.run_id}/esphome/${encodeURIComponent(f)}`;
  const espLinks = (r, primary) => (r.esphome_files || []).map((f) =>
    `<a class="btn ${primary && f.endsWith(".json") ? "primary" : "ghost"}" href="${espUrl(r, f)}" download>${icons.download} ${esc(f)}${primary && f.endsWith(".json") ? ` <span class="sub">for ESPHome</span>` : ""}</a>`).join("");
  const espPanel = (r) => {
    const m = r.micro, json = r.esphome_files.find((f) => f.endsWith(".json"));
    const url = `http://${location.hostname}:${system.http_port}/esphome/${json}`;
    return `<div class="panel">
      <div class="row between wrap"><h2>ESPHome model (microWakeWord)</h2><span class="muted small">${m.steps.toLocaleString()} steps, ${fmtDuration(m.duration_seconds)} of CPU training</span></div>
      <div class="tiles">
        ${tile(m.recall == null ? "n/a" : pct(m.recall), "Test voices detected", "At the probability cutoff, on held-out Kokoro voices")}
        ${tile(m.faph == null ? "n/a" : m.faph.toFixed(2), "False accepts per hour", "Streaming test on dinner party recordings")}
        ${tile(m.probability_cutoff.toFixed(2), "Probability cutoff", m.target_met ? "Written into the JSON manifest" : "Target not met, so this is the fewest false accepts")}
      </div>
      <div class="row gap wrap">${espLinks(r, true)}</div>
      <h3 class="mt">Add it to your ESPHome device</h3>
      <p class="muted small">This container serves the published files, so the device YAML can point straight at it. ESPHome downloads the model when it compiles. The device needs micro_wake_word support, such as an ESP32-S3 with PSRAM like the Voice PE.</p>
      <pre class="code">micro_wake_word:
  models:
    - model: ${esc(url)}</pre>
      <p class="muted small">If ESPHome can't reach this server, copy both files into one folder on the machine running ESPHome and set <code>model:</code> to the full path of the .json file. To make it more or less sensitive, add <code>probability_cutoff: 95%</code> under the model. If the device logs that it can't allocate tensors, raise <code>tensor_arena_size</code> in the .json.</p>
      <details class="mt"><summary>Manifest</summary><pre class="code">${esc(JSON.stringify(m.manifest, null, 2))}</pre></details>
      <details><summary>Cutoff table</summary>
        <div class="table-wrap"><table class="table">
          <thead><tr><th>Cutoff</th><th>Test voices detected</th><th>False accepts / hour</th></tr></thead>
          <tbody>${m.roc.filter((row) => row.cutoff >= 0.5).map((row) => `<tr class="${row.cutoff.toFixed(2) === m.probability_cutoff.toFixed(2) ? "hl" : ""}">
            <td>${row.cutoff.toFixed(2)}</td><td>${pct(1 - row.frr, 1)}</td><td>${row.faph.toFixed(2)}</td></tr>`).join("")}</tbody>
        </table></div>
      </details>
    </div>`;
  };
  const downloads = (r, primary) => {
    if (!r.files.length) return "";
    const tflite = r.files.find((f) => f.endsWith(".tflite")), onnx = r.files.find((f) => f.endsWith(".onnx"));
    return `${tflite ? `<a class="btn ${primary ? "primary" : ""}" href="${fileUrl(r, tflite)}" download>${icons.download} ${esc(tflite)}${primary ? ` <span class="sub">for Home Assistant</span>` : ""}</a>` : `<span class="warn-text small">No .tflite${r.tflite?.error ? `: ${esc(r.tflite.error)}` : ""}</span>`}
      ${onnx ? `<a class="btn ${primary ? "" : "ghost"}" href="${fileUrl(r, onnx)}" download>${icons.download} ${esc(onnx)}</a>` : ""}`;
  };

  let html = "";
  if (!runs.length) {
    html = emptyPanel("No model yet", "Once training finishes, the .tflite and .onnx files appear here.", `#/p/${project.id}/train`, "Go to training");
  } else {
    const r = runs[0], hl = r.headline || {};
    html = (r.files.length ? `
      <div class="panel">
        <div class="row between wrap"><h2>Home Assistant model (openWakeWord)</h2><span class="muted small">Trained ${fmtAgo(r.created_at)} in ${fmtDuration(r.duration_seconds)} on ${esc(r.device)}</span></div>
        <div class="tiles">
          ${tile(pct(hl.recall_tts), "Test voices detected", "Held-out Kokoro voices with noise and reverb")}
          ${tile(hl.recall_real_clips == null ? "n/a" : pct(hl.recall_real_clips), "Your recordings detected", hl.recall_real_clips == null ? "Record your voice to measure this" : "Includes clips used for training")}
          ${tile(hl.fp_per_hour == null ? "n/a" : hl.fp_per_hour.toFixed(2), "False activations per hour", "Measured on 11 hours of speech, music and noise")}
          ${tile((r.recommended_threshold ?? 0.5).toFixed(2), "Recommended threshold", "Balance of detections and false activations")}
        </div>
        <div class="row gap wrap">${downloads(r, true)}<button class="btn ghost" id="publish">Copy to output folder</button></div>
        <details class="mt"><summary>Threshold table</summary>
          <div class="table-wrap"><table class="table">
            <thead><tr><th>Threshold</th><th>Test voices</th><th>Your voice (held out)</th><th>Similar phrases accepted</th><th>False activations / hour</th></tr></thead>
            <tbody>${r.thresholds.map((row) => `<tr class="${row.threshold === r.recommended_threshold ? "hl" : ""}">
              <td>${row.threshold.toFixed(2)}</td><td>${pct(row.recall_tts, 1)}</td><td>${row.recall_real == null ? "n/a" : pct(row.recall_real, 1)}</td>
              <td>${pct(row.similar_phrase_accept, 1)}</td><td>${row.fp_per_hour.toFixed(2)}</td></tr>`).join("")}</tbody>
          </table></div>
        </details>
      </div>
      <div class="panel">
        <h2>Use it in Home Assistant</h2>
        <ol class="steps">
          <li>Download <code>${esc(r.model_name)}.tflite</code> above, or take it from the output folder.</li>
          <li>Copy it into <code>/share/openwakeword</code> on Home Assistant with the Samba share or File editor add-on. Create the folder if it doesn't exist.</li>
          <li>Restart the <b>openWakeWord</b> add-on.</li>
          <li>Go to <b>Settings › Voice assistants</b>, open your assistant and pick <b>${esc(r.model_name)}</b> as the wake word.</li>
        </ol>
        <p class="muted small">Running wyoming-openwakeword yourself? Put the file in the folder you pass to <code>--custom-model-dir</code>. In Python: <code>openwakeword.Model(wakeword_models=["${esc(r.model_name)}.onnx"], inference_framework="onnx")</code>. ESPHome devices that detect wake words on the device itself need the ESPHome model instead: tick it on the Train tab.</p>
      </div>` : "") + (r.micro ? espPanel(r) : "") + `
      ${runs.length > 1 ? `<div class="panel"><h2>Earlier models</h2>${runs.slice(1).map((x) => `
        <div class="row between wrap gap" style="padding:8px 0;border-top:1px solid var(--border)">
          <div class="small">${new Date(x.created_at * 1000).toLocaleString()}<div class="muted xsmall">${x.headline ? `${pct(x.headline.recall_tts)} detected · ${x.headline.fp_per_hour.toFixed(2)} false/h · threshold ${x.recommended_threshold}` : ""}</div></div>
          <div class="row gap wrap">${downloads(x, false)}${espLinks(x, false)}</div>
        </div>`).join("")}</div>` : ""}`;
  }
  el.innerHTML = `${html}
    <div class="panel">
      <h2>Manage</h2>
      <label class="label" for="model-name">Model file name</label>
      <div class="row gap wrap"><input id="model-name" class="input mono" style="max-width:320px" value="${esc(project.model_name)}"><button class="btn" id="rename">Save</button></div>
      <p class="hint">The file name becomes the wake word's name in Home Assistant. Applies from the next training run.</p>
      <div class="row gap wrap mt">
        <button class="btn ghost" id="clear-cache">Clear generated clips</button>
        <button class="btn danger" id="delete">${icons.trash} Delete this wake word</button>
      </div>
    </div>`;

  $("#publish", el)?.addEventListener("click", async () => {
    try {
      const { copied } = await api(`/api/projects/${project.id}/runs/${runs[0].run_id}/publish`, { body: {} });
      toast(`Copied to ${copied.join(", ")}`, "ok", 7000);
    } catch (e) { toast(e.message, "error"); }
  });
  $("#rename", el).onclick = async () => {
    try {
      const p = await api(`/api/projects/${project.id}`, { method: "PATCH", body: { model_name: $("#model-name", el).value } });
      $("#model-name", el).value = p.model_name;
      toast(`Saved. Next model will be ${p.model_name}.tflite`, "ok");
    } catch (e) { toast(e.message, "error"); }
  };
  $("#clear-cache", el).onclick = async () => {
    if (!confirm("Delete the generated voice clips and cached features? They are rebuilt on the next training run. Your recordings and models are kept.")) return;
    try { await api(`/api/projects/${project.id}/cache`, { method: "DELETE" }); toast("Cache cleared", "ok"); } catch (e) { toast(e.message, "error"); }
  };
  $("#delete", el).onclick = async () => {
    if (prompt(`This deletes "${project.phrase}" with all its recordings and models. Type DELETE to confirm.`) !== "DELETE") return;
    try { await api(`/api/projects/${project.id}`, { method: "DELETE" }); toast("Deleted", "ok"); location.hash = "#/"; } catch (e) { toast(e.message, "error"); }
  };
}

// ---------------------------------------------------------------------------------------------
// Setup
async function renderSetup() {
  await refreshSystem();
  const s = system;
  const total = s.assets.reduce((n, a) => n + a.size_bytes, 0);
  app.innerHTML = `<div class="narrow">
    <h1>Setup</h1>
    <p class="muted">Everything needed runs inside this container. The only extra is shared training data, downloaded once.</p>
    <div class="panel">
      <div class="row between wrap"><h2>Training data</h2><span class="muted small">${fmtBytes(total)} in total · ${fmtBytes(s.disk_free_bytes)} free</span></div>
      <p class="muted">Shared by every wake word. If you skip this step it downloads automatically the first time you train. Downloads resume if interrupted.</p>
      <div class="asset-list" id="assets"></div>
      <div id="dl-area" class="mt"></div>
    </div>
    <div class="panel">
      <h2>GPU</h2>
      ${s.gpu?.available
        ? `<p>${icons.check} <b>${esc(s.gpu.name)}</b> with ${fmtBytes(s.gpu.memory_total_mb * 1048576)} of memory. Kokoro, feature extraction and training all use it.</p>`
        : `<p class="warn-text"><b>No NVIDIA GPU detected.</b> Everything still works on the CPU, it just takes several times longer.</p>
           <p class="muted small">On Unraid: install the <b>Nvidia Driver</b> plugin from Community Applications, then edit this container: <b>Extra Parameters</b> should contain <code>--runtime=nvidia</code> and <code>NVIDIA_VISIBLE_DEVICES</code> should be <code>all</code> or your GPU's UUID. ${esc(s.gpu?.detail || "")}</p>`}
    </div>
    <div class="panel">
      <h2>Folders</h2>
      <dl class="kv">
        <dt>Data</dt><dd><code>${esc(s.data_dir)}</code> wake words, recordings, downloads and the HTTPS certificate</dd>
        <dt>Output</dt><dd><code>${esc(s.output_dir)}</code> finished .tflite and .onnx files are copied here</dd>
        <dt>Your background audio</dt><dd><code>${esc(s.custom_background_dir)}</code> drop recordings of your home here (TV, music, appliances) to use them as background noise for every wake word</dd>
      </dl>
    </div>
    <div class="panel">
      <h2>Microphone access</h2>
      <p class="muted">Browsers only allow microphone recording on HTTPS pages. The container serves HTTPS on port ${s.https_port} with a self-signed certificate, so accept the browser warning once. Behind a reverse proxy with a real certificate, point the proxy at port ${s.http_port} and enable WebSocket support.</p>
    </div>
    <p class="muted small center">Wake Word Trainer ${esc(s.version)}</p>
  </div>`;

  const renderAssets = () => {
    $("#assets").innerHTML = system.assets.map((a) => `
      <div class="asset">
        <div><b>${esc(a.label)}</b><div class="muted small">${esc(a.description)}</div></div>
        <div class="asset-size small">${fmtBytes(a.size_bytes)}</div>
        <div>${a.ready ? `<span class="badge ok">${icons.check} Ready</span>` : a.partial_bytes ? `<span class="badge run">${fmtBytes(a.partial_bytes)} so far</span>` : `<span class="badge">Not downloaded</span>`}</div>
      </div>`).join("");
  };
  renderAssets();

  onJob((j) => {
    const area = $("#dl-area");
    if (!area) return;
    const downloading = j.running && j.kind === "download";
    if (downloading) {
      area.innerHTML = `<div class="bar big"><div style="width:${jobFraction(j) * 100}%"></div></div>
        <div class="row between mt"><span class="job-msg">${esc(j.message || "")}</span><button class="btn danger small" id="dl-cancel">Cancel</button></div>`;
      $("#dl-cancel").onclick = () => api("/api/job/cancel", { body: {} }).then(() => toast("Cancelling. Partial downloads resume next time."));
    } else if (j.running) {
      area.innerHTML = `<p class="muted small">A training job is running. Missing data is downloaded as part of it.</p>`;
    } else if (!system.assets_ready) {
      area.innerHTML = `${j.kind === "download" && j.state === "failed" ? `<div class="error-box">Download failed: ${esc(j.error || "")}</div>` : ""}
        <button class="btn primary" id="dl">${icons.download} Download everything now</button>`;
      $("#dl").onclick = async () => {
        try { await api("/api/assets/download", { body: {} }); await pollJobNow(); } catch (e) { toast(e.message, "error"); }
      };
    } else {
      area.innerHTML = `<p class="good-text">${icons.check} All training data is ready.</p>`;
    }
  });
  every(5000, async () => {
    if (job.running && job.kind === "download") { await refreshSystem(); if ($("#assets")) renderAssets(); }
  });
}

// ---------------------------------------------------------------------------------------------
window.addEventListener("hashchange", route);
refreshSystem().then(renderChrome);
pollJobLoop();
route();
setInterval(refreshSystem, 60000);
