"""v0.14 lightweight Web UI served from ``GET /``.

A single self-contained HTML page (no external CSS/JS deps, no
build step) that POSTs to ``/ask/stream`` and renders the SSE
events in real time. Meant as a zero-setup "open the browser and
go" frontend for ``pagewiki serve``.

Design goals:
  * No npm, no bundler, no framework — just ``<script>``/``<style>``
    inline in a single string constant.
  * Streams ``trace`` + ``usage`` + ``answer`` events using the
    Fetch API's ``ReadableStream``. No EventSource (doesn't support
    POST with body).
  * Korean-first copy to match the rest of the codebase.
  * Works offline once served — no CDN links.

Upgrade hooks:
  * ``PAGEWIKI_UI_HTML`` env var can override the inlined HTML for
    custom branding.
  * ``build_ui_html(title)`` is exposed so tests can inject a stub.
"""

from __future__ import annotations

import os

_DEFAULT_TITLE = "pagewiki"

_HTML_TEMPLATE = r"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root {
    --bg: #0e1217;
    --fg: #e4e7ec;
    --muted: #8b93a1;
    --border: #1e242e;
    --accent: #7aa2f7;
    --success: #9ece6a;
    --danger: #f7768e;
    --card: #151a21;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 0;
    font-family: -apple-system, "SF Pro Text", Roboto, "Segoe UI",
                 "Noto Sans KR", sans-serif;
    background: var(--bg); color: var(--fg);
    font-size: 14px; line-height: 1.5;
  }
  header {
    padding: 20px 24px; border-bottom: 1px solid var(--border);
    display: flex; justify-content: space-between; align-items: center;
  }
  header h1 { margin: 0; font-size: 18px; font-weight: 600; }
  header .sub { color: var(--muted); font-size: 12px; }
  main {
    max-width: 960px; margin: 0 auto; padding: 24px;
  }
  .card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 8px; padding: 16px 20px; margin-bottom: 16px;
  }
  textarea, input {
    width: 100%; background: var(--bg); color: var(--fg);
    border: 1px solid var(--border); border-radius: 6px;
    padding: 10px 12px; font: inherit; resize: vertical;
  }
  textarea:focus, input:focus {
    outline: none; border-color: var(--accent);
  }
  .row { display: flex; gap: 8px; align-items: center; margin-top: 8px; }
  .row label { color: var(--muted); font-size: 12px; }
  button {
    background: var(--accent); color: #0e1217; border: none;
    border-radius: 6px; padding: 10px 16px; font-weight: 600;
    cursor: pointer; font-size: 14px;
  }
  button:disabled { opacity: 0.5; cursor: not-allowed; }
  button.secondary { background: transparent; color: var(--fg); border: 1px solid var(--border); }
  button.danger { background: var(--danger); color: #0e1217; }
  .trace {
    font-family: "SF Mono", Menlo, Consolas, monospace;
    font-size: 12px; color: var(--muted);
    max-height: 240px; overflow-y: auto;
    background: var(--bg); border: 1px solid var(--border);
    border-radius: 6px; padding: 10px 12px; margin-top: 8px;
  }
  .trace .phase { color: var(--accent); font-weight: 600; }
  .trace .cancel { color: var(--danger); }
  .trace .budget { color: var(--danger); }
  .trace .finalize { color: var(--success); }
  .answer {
    white-space: pre-wrap; word-break: break-word;
    font-size: 14px; line-height: 1.7;
  }
  .usage-pill {
    display: inline-block; background: var(--bg); color: var(--muted);
    border: 1px solid var(--border); border-radius: 12px;
    padding: 2px 10px; font-size: 11px; margin-left: 6px;
  }
  .cited { color: var(--muted); font-size: 12px; }
  .cited code { background: var(--bg); padding: 1px 6px; border-radius: 3px; }
  .sparkline-card {
    display: flex; align-items: center; gap: 16px;
  }
  .sparkline-card svg { display: block; }
  .sparkline-card .labels {
    display: flex; flex-direction: column; gap: 4px;
    font-size: 11px; color: var(--muted);
  }
  .sparkline-card .labels strong { color: var(--fg); font-size: 13px; }
  details.history-card summary {
    cursor: pointer;
    color: var(--muted);
    font-size: 12px;
    margin-bottom: 6px;
    user-select: none;
  }
  details.history-card summary:hover { color: var(--fg); }
  .history-table {
    width: 100%;
    font-family: "SF Mono", Menlo, Consolas, monospace;
    font-size: 11px;
    max-height: 240px;
    overflow-y: auto;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 6px 10px;
    margin-top: 6px;
  }
  .history-table .row {
    display: grid;
    grid-template-columns: 160px 80px 1fr 1fr;
    gap: 8px;
    padding: 2px 0;
    border-bottom: 1px solid var(--border);
  }
  .history-table .row:last-child { border-bottom: none; }
  .history-table .header {
    color: var(--fg);
    font-weight: 600;
    border-bottom: 1px solid var(--accent);
  }
  .history-controls {
    display: flex; gap: 8px; align-items: center;
  }
  .history-controls button { padding: 4px 10px; font-size: 11px; }
  .history-status {
    font-size: 11px; color: var(--muted);
    margin-left: auto;
  }
</style>
</head>
<body>
<header>
  <div>
    <h1>__TITLE__</h1>
    <div class="sub">Vectorless reasoning-based RAG for Obsidian vaults</div>
  </div>
  <div>
    <span id="status" class="usage-pill">idle</span>
    <span id="tokens" class="usage-pill">0 tokens</span>
  </div>
</header>

<main>
  <div class="card">
    <textarea id="query" rows="4" placeholder="질문을 입력하세요..."></textarea>
    <div class="row">
      <label><input type="checkbox" id="decompose"> Decompose</label>
      <label><input type="checkbox" id="jsonMode"> JSON mode</label>
      <label><input type="checkbox" id="reuseContext"> Reuse context</label>
      <div style="flex:1"></div>
      <button id="askBtn">Ask</button>
      <button id="cancelBtn" class="danger" disabled>Cancel</button>
    </div>
  </div>

  <div class="card">
    <div style="color: var(--muted); font-size: 12px; margin-bottom: 6px;">
      Trace
    </div>
    <div class="trace" id="trace">(no events yet)</div>
  </div>

  <div class="card">
    <div style="color: var(--muted); font-size: 12px; margin-bottom: 6px;">
      Answer
    </div>
    <div class="answer" id="answer">(ask a question to get started)</div>
    <div id="cited" class="cited" style="margin-top: 12px;"></div>
  </div>

  <div class="card sparkline-card">
    <svg id="sparkline" width="200" height="40" viewBox="0 0 200 40"
         preserveAspectRatio="none">
      <polyline id="sparkline-path" fill="none" stroke="#7aa2f7" stroke-width="1.5"
                points=""/>
      <circle id="sparkline-dot" cx="0" cy="0" r="2" fill="#7aa2f7"
              style="display:none"/>
    </svg>
    <div class="labels">
      <strong id="spark-total">0</strong>
      <span>tokens this query</span>
      <span id="spark-points" style="font-size: 10px;">0 calls</span>
    </div>
  </div>

  <details class="card history-card">
    <summary>Historical usage events (/usage/history/stream)</summary>
    <div class="history-controls">
      <button id="history-start">Start</button>
      <button id="history-stop">Stop</button>
      <span class="history-status" id="history-status">idle</span>
    </div>
    <div class="history-table" id="history-table">
      <div class="row header">
        <div>timestamp</div>
        <div>phase</div>
        <div>prompt</div>
        <div>completion</div>
      </div>
      <div id="history-rows"></div>
    </div>
  </details>
</main>

<script>
(function () {
  const qEl = document.getElementById("query");
  const decomposeEl = document.getElementById("decompose");
  const jsonModeEl = document.getElementById("jsonMode");
  const reuseCtxEl = document.getElementById("reuseContext");
  const askBtn = document.getElementById("askBtn");
  const cancelBtn = document.getElementById("cancelBtn");
  const traceEl = document.getElementById("trace");
  const answerEl = document.getElementById("answer");
  const citedEl = document.getElementById("cited");
  const statusEl = document.getElementById("status");
  const tokensEl = document.getElementById("tokens");
  const sparklinePath = document.getElementById("sparkline-path");
  const sparkTotal = document.getElementById("spark-total");
  const sparkPoints = document.getElementById("spark-points");

  let abortController = null;
  let usageSeries = [];  // cumulative token counts per usage event

  askBtn.addEventListener("click", runQuery);
  cancelBtn.addEventListener("click", () => {
    if (abortController) {
      abortController.abort();
      statusEl.textContent = "cancelled";
    }
  });
  qEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
      e.preventDefault();
      runQuery();
    }
  });

  async function runQuery() {
    const query = qEl.value.trim();
    if (!query) return;

    traceEl.innerHTML = "";
    answerEl.textContent = "Thinking...";
    citedEl.textContent = "";
    statusEl.textContent = "streaming";
    tokensEl.textContent = "0 tokens";
    usageSeries = [];
    renderSparkline();
    askBtn.disabled = true;
    cancelBtn.disabled = false;

    abortController = new AbortController();

    try {
      const resp = await fetch("/ask/stream", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Accept": "text/event-stream",
        },
        body: JSON.stringify({
          query,
          decompose: decomposeEl.checked,
        }),
        signal: abortController.signal,
      });

      if (!resp.ok || !resp.body) {
        throw new Error("HTTP " + resp.status);
      }

      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        let idx;
        while ((idx = buf.indexOf("\n\n")) !== -1) {
          const frame = buf.slice(0, idx);
          buf = buf.slice(idx + 2);
          if (!frame.trim()) continue;
          handleFrame(frame);
        }
      }
      statusEl.textContent = "done";
    } catch (err) {
      if (err.name === "AbortError") {
        statusEl.textContent = "cancelled";
      } else {
        statusEl.textContent = "error";
        answerEl.textContent = "Error: " + err.message;
      }
    } finally {
      askBtn.disabled = false;
      cancelBtn.disabled = true;
      abortController = null;
    }
  }

  function handleFrame(frame) {
    let eventName = "message";
    let dataStr = "";
    for (const line of frame.split("\n")) {
      if (line.startsWith("event: ")) eventName = line.slice(7).trim();
      else if (line.startsWith("data: ")) dataStr += line.slice(6);
    }
    if (!dataStr) return;
    let data;
    try { data = JSON.parse(dataStr); }
    catch { return; }

    if (eventName === "trace") {
      const line = document.createElement("div");
      const phaseClass = data.phase || "trace";
      line.innerHTML =
        '<span class="phase ' + phaseClass + '">' +
        escapeHtml(data.phase || "-") +
        '</span> ' +
        (data.node_id ? '<code>' + escapeHtml(data.node_id) + '</code> ' : '') +
        escapeHtml(data.detail || "");
      traceEl.appendChild(line);
      traceEl.scrollTop = traceEl.scrollHeight;
    } else if (eventName === "usage") {
      tokensEl.textContent = (data.total_tokens || 0) + " tokens";
      usageSeries.push(data.total_tokens || 0);
      renderSparkline();
    } else if (eventName === "answer") {
      answerEl.textContent = data.answer || "";
      if (Array.isArray(data.cited_nodes) && data.cited_nodes.length) {
        citedEl.innerHTML = "Cited: " +
          data.cited_nodes.map((c) => '<code>' + escapeHtml(c) + '</code>').join(" ");
      }
    } else if (eventName === "error") {
      answerEl.textContent = "Error: " + (data.message || "unknown");
      statusEl.textContent = "error";
    }
  }

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
  }

  function renderSparkline() {
    const W = 200, H = 40, pad = 2;
    const points = usageSeries;
    if (points.length === 0) {
      sparklinePath.setAttribute("points", "");
      sparkTotal.textContent = "0";
      sparkPoints.textContent = "0 calls";
      return;
    }
    const max = Math.max(...points) || 1;
    const step = points.length > 1 ? (W - pad * 2) / (points.length - 1) : 0;
    const coords = points.map((v, i) => {
      const x = pad + i * step;
      const y = H - pad - ((v / max) * (H - pad * 2));
      return x.toFixed(1) + "," + y.toFixed(1);
    });
    sparklinePath.setAttribute("points", coords.join(" "));
    sparkTotal.textContent = String(points[points.length - 1] || 0);
    sparkPoints.textContent = points.length + " calls";
  }

  // ── v0.16 historical view — subscribes to /usage/history/stream ──
  const historyStartBtn = document.getElementById("history-start");
  const historyStopBtn = document.getElementById("history-stop");
  const historyStatusEl = document.getElementById("history-status");
  const historyRowsEl = document.getElementById("history-rows");
  let historyAbortController = null;
  const MAX_HISTORY_ROWS = 200;

  historyStartBtn.addEventListener("click", startHistoryStream);
  historyStopBtn.addEventListener("click", stopHistoryStream);

  async function startHistoryStream() {
    if (historyAbortController) return;
    historyRowsEl.innerHTML = "";
    historyStatusEl.textContent = "connecting...";
    historyAbortController = new AbortController();
    try {
      const resp = await fetch(
        "/usage/history/stream?poll_interval=1&initial_limit=50&max_duration=900",
        { signal: historyAbortController.signal },
      );
      if (!resp.ok) {
        if (resp.status === 503) {
          historyStatusEl.textContent = "no --usage-db";
        } else {
          historyStatusEl.textContent = "HTTP " + resp.status;
        }
        historyAbortController = null;
        return;
      }
      historyStatusEl.textContent = "streaming";
      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        let idx;
        while ((idx = buf.indexOf("\n\n")) !== -1) {
          const frame = buf.slice(0, idx);
          buf = buf.slice(idx + 2);
          handleHistoryFrame(frame);
        }
      }
      historyStatusEl.textContent = "closed";
    } catch (err) {
      if (err.name === "AbortError") {
        historyStatusEl.textContent = "stopped";
      } else {
        historyStatusEl.textContent = "error: " + err.message;
      }
    } finally {
      historyAbortController = null;
    }
  }

  function stopHistoryStream() {
    if (historyAbortController) {
      historyAbortController.abort();
    }
  }

  function handleHistoryFrame(frame) {
    let eventName = "message";
    let dataStr = "";
    for (const line of frame.split("\n")) {
      if (line.startsWith("event: ")) eventName = line.slice(7).trim();
      else if (line.startsWith("data: ")) dataStr += line.slice(6);
    }
    if (!dataStr) return;
    let data;
    try { data = JSON.parse(dataStr); }
    catch { return; }

    if (eventName === "initial") {
      for (const e of (data.events || [])) appendHistoryRow(e);
    } else if (eventName === "event") {
      appendHistoryRow(data);
    } else if (eventName === "heartbeat") {
      historyStatusEl.textContent =
        "streaming (last seen " +
        (data.last_seen ? new Date(data.last_seen * 1000).toLocaleTimeString() : "—") +
        ")";
    } else if (eventName === "done") {
      historyStatusEl.textContent = "closed";
    }
  }

  function appendHistoryRow(e) {
    const row = document.createElement("div");
    row.className = "row";
    const ts = new Date(e.timestamp * 1000).toISOString().replace("T", " ").slice(0, 19);
    row.innerHTML =
      '<div>' + escapeHtml(ts) + '</div>' +
      '<div>' + escapeHtml(e.phase || "—") + '</div>' +
      '<div>' + (e.prompt || 0) + '</div>' +
      '<div>' + (e.completion || 0) + '</div>';
    historyRowsEl.appendChild(row);
    while (historyRowsEl.childElementCount > MAX_HISTORY_ROWS) {
      historyRowsEl.removeChild(historyRowsEl.firstChild);
    }
    historyRowsEl.parentElement.scrollTop =
      historyRowsEl.parentElement.scrollHeight;
  }
})();
</script>
</body>
</html>
"""


def build_ui_html(title: str = _DEFAULT_TITLE) -> str:
    """Return the embedded UI HTML with the title substituted in.

    Respects the ``PAGEWIKI_UI_HTML`` environment variable for
    users who want to point at a customized HTML file without
    editing the source.
    """
    override = os.getenv("PAGEWIKI_UI_HTML")
    if override and os.path.exists(override):
        with open(override, encoding="utf-8") as f:
            return f.read()
    return _HTML_TEMPLATE.replace("__TITLE__", title)
