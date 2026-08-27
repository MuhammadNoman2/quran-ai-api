/**
 * Live recitation demo.
 *
 * Talks to the API the same way any client would: REST to open a session,
 * WebSocket to stream pcm_s16le, and the documented events back. No model, no ML
 * dependency, nothing server-specific.
 *
 * One rule this UI takes seriously: `mistake_detected` is the *only* thing shown
 * in red. Provisional highlights and unconfirmed observations are styled
 * differently on purpose - rendering them as mistakes would re-introduce exactly
 * the false-correction problem the server works to avoid.
 */

const API = "/api/v1";
const $ = (id) => document.getElementById(id);

const ui = {
  surah: $("surah"), ayah: $("ayah"), tier: $("tier"),
  record: $("record"), reset: $("reset"),
  verse: $("verse"), bar: $("bar"), progressText: $("progressText"),
  score: $("score"), transcript: $("transcript"), log: $("log"),
  status: $("status"), statusText: $("statusText"),
};

let ayahData = null;      // the ayah as the API returned it
let chips = new Map();    // uthmani_index -> {el, indices:Set}
let stateByIndex = new Map();
let ws = null, audio = null, stream = null, node = null, recording = false;

// ── helpers ─────────────────────────────────────────────────────────────────

function setStatus(text, kind = "") {
  ui.statusText.textContent = text;
  ui.status.className = "pill" + (kind ? " " + kind : "");
}

function log(kind, detail, bad = false) {
  const line = document.createElement("div");
  if (bad) line.className = "bad";
  const time = new Date().toLocaleTimeString([], { hour12: false });
  line.innerHTML =
    `<span class="t">${time}</span> <span class="k">${kind}</span> ` +
    (detail ? escapeHtml(detail) : "");
  ui.log.prepend(line);
  while (ui.log.childElementCount > 300) ui.log.lastElementChild.remove();
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// ── verse rendering ─────────────────────────────────────────────────────────

async function loadSurahs() {
  const surahs = await (await fetch(`${API}/surahs`)).json();
  ui.surah.innerHTML = surahs
    .map((s) => `<option value="${s.surah}" data-count="${s.ayah_count}">${s.surah}. ${s.name}</option>`)
    .join("");
  ui.surah.value = "1";
}

async function loadAyah() {
  const surah = Number(ui.surah.value);
  const ayah = Number(ui.ayah.value);
  const selected = ui.surah.selectedOptions[0];
  ui.ayah.max = selected ? selected.dataset.count : 1;
  if (ayah > Number(ui.ayah.max)) ui.ayah.value = ui.ayah.max;

  const response = await fetch(`${API}/surahs/${surah}/ayahs/${ui.ayah.value}`);
  if (!response.ok) { ui.verse.textContent = "That ayah does not exist."; return; }
  ayahData = await response.json();
  renderVerse();
}

function renderVerse() {
  ui.verse.innerHTML = "";
  chips = new Map();
  stateByIndex = new Map();

  // One chip per *Uthmani* word. Several Imlaey words can share one Uthmani word
  // (يَـٰٓأَيُّهَا is written as two in Imlaey), and word indices in results are
  // Imlaey-based - so the chip tracks every Imlaey index that maps onto it.
  for (const word of ayahData.words) {
    let chip = chips.get(word.uthmani_index);
    if (!chip) {
      const el = document.createElement("span");
      el.className = "w";
      el.textContent = word.uthmani;
      ui.verse.appendChild(el);
      ui.verse.appendChild(document.createTextNode(" "));
      chip = { el, indices: new Set() };
      chips.set(word.uthmani_index, chip);
    }
    chip.indices.add(word.index);
  }
  resetProgress();
}

function chipFor(wordIndex) {
  for (const chip of chips.values()) if (chip.indices.has(wordIndex)) return chip;
  return null;
}

const RANK = { provisional: 1, uncertain: 1, correct: 2, mistake: 3 };

function markWord(wordIndex, state, heard) {
  const chip = chipFor(wordIndex);
  if (!chip) return;
  stateByIndex.set(wordIndex, state);

  // A chip covering several words shows the strongest signal among them, so a
  // mistake is never hidden by a neighbouring correct word.
  let strongest = "";
  for (const index of chip.indices) {
    const current = stateByIndex.get(index);
    if (current && (!strongest || RANK[current] > RANK[strongest])) strongest = current;
  }
  chip.el.className = "w " + strongest;

  const existing = chip.el.querySelector(".heard");
  if (existing) existing.remove();
  if (state === "mistake" && heard) {
    const tag = document.createElement("span");
    tag.className = "heard";
    tag.textContent = heard;
    chip.el.appendChild(tag);
  }
}

function resetProgress() {
  for (const chip of chips.values()) {
    chip.el.className = "w";
    const tag = chip.el.querySelector(".heard");
    if (tag) tag.remove();
  }
  stateByIndex.clear();
  ui.bar.style.width = "0%";
  ui.score.textContent = "—";
  ui.transcript.textContent = "—";
  ui.progressText.textContent = `0 / ${ayahData ? ayahData.words.length : 0} words`;
}

// ── events from the server ──────────────────────────────────────────────────

function handle(message) {
  switch (message.event) {
    case "ready":
      setStatus("Listening", "live");
      log("ready", `${message.expected_words} words expected`);
      break;

    case "speech_started":
      setStatus("Reciting", "live");
      break;

    case "speech_stopped":
      setStatus("Checking…", "live");
      log("pause", `${message.silence_seconds}s silence — checking`);
      break;

    case "partial_transcript":
      ui.transcript.textContent = message.text || "—";
      break;

    case "word_detected":
      markWord(message.word_index, "provisional");
      break;

    case "word_confirmed":
      markWord(message.word_index, "correct");
      break;

    case "correction":
      log("correction", `word ${message.word_index}: ${message.previous_status} → ${message.status}`);
      break;

    case "mistake_detected":
      markWord(message.word_index, "mistake", message.spoken);
      log("mistake", `word ${message.word_index}: ${message.error_type} — expected ${message.expected}`, true);
      break;

    case "ayah_progress":
      ui.bar.style.width = `${Math.round(message.fraction * 100)}%`;
      ui.progressText.textContent = `${message.words_confirmed} / ${message.words_total} words`;
      break;

    case "ayah_completed":
      ui.score.textContent = message.word_accuracy_score;
      log("done", `score ${message.word_accuracy_score}, ${message.errors} mistake(s)`);
      break;

    case "session_completed":
      setStatus("Finished");
      log("session", `${message.audio_seconds}s, ${message.segments_analysed} pass(es)`);
      break;

    case "warning":
      log("warning", message.message);
      break;

    case "error":
      log("error", `${message.code}: ${message.message}`, true);
      if (message.fatal) stopRecording();
      break;
  }
}

// ── recording ───────────────────────────────────────────────────────────────

async function startRecording() {
  if (!ayahData) { alert("Pick a verse first."); return; }
  resetProgress();
  setStatus("Connecting…");

  let session;
  try {
    const response = await fetch(`${API}/sessions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        surah: ayahData.surah, ayah: ayahData.ayah, tier: ui.tier.value,
      }),
    });
    session = await response.json();
    if (!response.ok) throw new Error(session.error ? session.error.message : "could not open a session");
  } catch (err) {
    setStatus("Server unreachable", "error");
    log("error", err.message, true);
    return;
  }

  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false },
    });
  } catch (err) {
    setStatus("Microphone blocked", "error");
    log("error", "microphone permission denied", true);
    return;
  }

  // Ask for 16 kHz directly so the browser resamples rather than us.
  audio = new AudioContext({ sampleRate: 16000 });
  if (audio.sampleRate !== 16000) {
    log("warning", `browser gave ${audio.sampleRate} Hz, not 16000 — audio may be misread`);
  }
  await audio.audioWorklet.addModule("/pcm-worklet.js");

  const protocol = location.protocol === "https:" ? "wss:" : "ws:";
  ws = new WebSocket(`${protocol}//${location.host}${API}/sessions/${session.session_id}/stream`);
  ws.binaryType = "arraybuffer";

  ws.onopen = () => {
    ws.send(JSON.stringify({ type: "start", surah: ayahData.surah, ayah: ayahData.ayah }));
    const source = audio.createMediaStreamSource(stream);
    node = new AudioWorkletNode(audio, "pcm-worklet");
    node.port.onmessage = (event) => {
      if (ws && ws.readyState === WebSocket.OPEN) ws.send(event.data);
    };
    source.connect(node);
    // Keep the worklet pulling without making the microphone audible.
    const mute = audio.createGain();
    mute.gain.value = 0;
    node.connect(mute).connect(audio.destination);
  };

  ws.onmessage = (event) => handle(JSON.parse(event.data));
  ws.onerror = () => { setStatus("Connection error", "error"); };
  ws.onclose = () => { if (recording) stopRecording(); };

  recording = true;
  ui.record.textContent = "Stop";
  ui.record.classList.add("recording");
}

function stopRecording() {
  recording = false;
  ui.record.textContent = "Start reciting";
  ui.record.classList.remove("recording");
  setStatus("Finishing…");

  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: "stop" }));
  if (node) { node.port.onmessage = null; node.disconnect(); node = null; }
  if (stream) { stream.getTracks().forEach((t) => t.stop()); stream = null; }
  if (audio) { audio.close(); audio = null; }
  // The socket is left open briefly so the final events still arrive.
  setTimeout(() => { if (ws) { ws.close(); ws = null; } }, 12000);
}

// ── wiring ──────────────────────────────────────────────────────────────────

ui.record.onclick = () => (recording ? stopRecording() : startRecording());
ui.reset.onclick = () => { if (!recording) resetProgress(); };
ui.surah.onchange = () => { ui.ayah.value = 1; loadAyah(); };
ui.ayah.onchange = loadAyah;
ui.ayah.oninput = () => { clearTimeout(ui.ayah._t); ui.ayah._t = setTimeout(loadAyah, 350); };

function applyUrlParams() {
  // ?surah=2&ayah=255 - handy for demos and for linking someone straight to a verse.
  const params = new URLSearchParams(location.search);
  const surah = params.get("surah");
  const ayah = params.get("ayah");
  if (surah) ui.surah.value = surah;
  if (ayah) ui.ayah.value = ayah;
}

(async function init() {
  try {
    await loadSurahs();
    applyUrlParams();
    await loadAyah();
    const health = await (await fetch(`${API}/health`)).json();
    setStatus("Ready");
    log("connected", `${health.device} · ${health.compute_type}`);
  } catch (err) {
    setStatus("Server unreachable", "error");
    log("error", "is the API running?", true);
  }
})();
