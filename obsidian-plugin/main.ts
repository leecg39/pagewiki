/**
 * PageWiki Obsidian Plugin (v0.13)
 *
 * Wraps the pagewiki CLI so users can scan, ask, chat, compile, and watch
 * from within Obsidian — no terminal switching required.
 *
 * v0.12 added optional "server mode": Ask/Chat can talk directly to a
 * running `pagewiki serve` instance via POST /ask/stream or POST
 * /chat/stream. v0.13 extends server mode with WebSocket support so
 * long-running queries can be cancelled mid-loop via a Cancel button.
 *
 * Version history:
 *   - v0.7: maxWorkers + decomposeByDefault
 *   - v0.8: N/A (plugin-side)
 *   - v0.9: --usage, --max-tokens flags
 *   - v0.10: --json-mode, --reuse-context flags
 *   - v0.11: Chat modal full flag surface
 *   - v0.12: Server mode (SSE streaming from pagewiki serve)
 *   - v0.13: WebSocket mode + Cancel button for in-flight queries
 */

import {
	App,
	Modal,
	Notice,
	Plugin,
	PluginSettingTab,
	Setting,
	TextAreaComponent,
} from "obsidian";
import { exec } from "child_process";

// ─────────────────────────────────────────────────────────────────────────────
// Settings
// ─────────────────────────────────────────────────────────────────────────────

interface PageWikiSettings {
	model: string;
	numCtx: number;
	folder: string;
	pythonPath: string;
	maxWorkers: number;
	decomposeByDefault: boolean;
	// v0.9
	showUsage: boolean;
	maxTokens: number;  // 0 = unlimited
	// v0.10
	jsonMode: boolean;
	reuseContext: boolean;
	// v0.12 — when set, Ask/Chat stream via SSE from this server
	// instead of spawning `python -m pagewiki` as a subprocess.
	serverUrl: string;
	// v0.13 — prefer WebSocket /ask/ws over SSE /ask/stream so
	// in-flight queries can be cancelled via the Cancel button.
	useWebSocket: boolean;
	// v0.15 — per-phase token budget split
	// (summarize:retrieve:synthesis). Only meaningful when
	// ``maxTokens > 0`` and server mode is active.
	tokenSplit: string;
	// v0.16 — opt in to the server's prompt-cache mode over
	// WebSocket. Requires `pagewiki serve --prompt-cache` on the
	// server side; ignored when the server wasn't started with it.
	promptCacheWebSocket: boolean;
}

const DEFAULT_SETTINGS: PageWikiSettings = {
	model: "ollama/gemma4:26b",
	numCtx: 131072,
	folder: "Research",
	pythonPath: "python",
	maxWorkers: 4,
	decomposeByDefault: false,
	showUsage: false,
	maxTokens: 0,
	jsonMode: false,
	reuseContext: false,
	serverUrl: "",
	useWebSocket: false,
	tokenSplit: "",
	promptCacheWebSocket: false,
};

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/** Strip ANSI escape codes and Rich markup from CLI output. */
function stripAnsi(text: string): string {
	// eslint-disable-next-line no-control-regex
	return text.replace(/\x1b\[[0-9;]*m/g, "").replace(/\[\/?\w+\]/g, "");
}

// ─────────────────────────────────────────────────────────────────────────────
// Server mode SSE client (v0.12)
// ─────────────────────────────────────────────────────────────────────────────

interface SSEEvent {
	event: string;
	data: any;
}

/**
 * POST JSON to an SSE endpoint and invoke ``onEvent`` for each parsed frame.
 *
 * Uses the standard ``text/event-stream`` format:
 *   event: <name>\n
 *   data: <json>\n\n
 *
 * Returns when the stream closes (either naturally or on error).
 * The caller can abort mid-stream by passing an AbortSignal in
 * ``opts.signal``.
 */
// ─────────────────────────────────────────────────────────────────────────────
// WebSocket client (v0.13)
// ─────────────────────────────────────────────────────────────────────────────

interface WSCallbacks {
	onTrace?: (data: any) => void;
	onUsage?: (data: any) => void;
	onAnswer?: (data: any) => void;
	onError?: (data: any) => void;
	onCancelled?: () => void;
}

/**
 * Connect to ``ws://.../ask/ws``, send one ``ask`` message, and
 * dispatch incoming frames to the provided callbacks.
 *
 * Returns a handle with ``cancel()`` that sends ``{type: "cancel"}``
 * to the server and eventually closes the socket. The promise
 * resolves when the server finishes streaming (either with an
 * ``answer``, ``cancelled``, or ``error`` frame).
 */
interface AskOptions {
	decompose: boolean;
	// v0.15 extensions — all optional so older plugin versions keep
	// working against a v0.12 server.
	maxTokens?: number;
	tokenSplit?: string;
	jsonMode?: boolean;
	reuseContext?: boolean;
	// v0.16 — opt into server-side prompt cache. Server must have
	// been started with `--prompt-cache` for this to take effect.
	promptCache?: boolean;
}

function connectAskWS(
	wsUrl: string,
	query: string,
	opts: AskOptions,
	callbacks: WSCallbacks,
): { cancel: () => void; finished: Promise<void> } {
	const ws = new WebSocket(wsUrl);

	let resolveFinished!: () => void;
	let rejectFinished!: (err: Error) => void;
	const finished = new Promise<void>((resolve, reject) => {
		resolveFinished = resolve;
		rejectFinished = reject;
	});

	ws.addEventListener("open", () => {
		const askFrame: Record<string, unknown> = {
			type: "ask",
			query,
			decompose: opts.decompose,
		};
		if (opts.maxTokens && opts.maxTokens > 0) {
			askFrame.max_tokens = opts.maxTokens;
		}
		if (opts.tokenSplit && opts.tokenSplit.trim()) {
			askFrame.token_split = opts.tokenSplit.trim();
		}
		if (opts.jsonMode) askFrame.json_mode = true;
		if (opts.reuseContext) askFrame.reuse_context = true;
		if (opts.promptCache) askFrame.prompt_cache = true;
		ws.send(JSON.stringify(askFrame));
	});

	ws.addEventListener("message", (ev: MessageEvent) => {
		let msg: any;
		try {
			msg = JSON.parse(ev.data);
		} catch {
			return;
		}
		switch (msg.type) {
			case "trace":
				callbacks.onTrace?.(msg);
				break;
			case "usage":
				callbacks.onUsage?.(msg);
				break;
			case "answer":
				callbacks.onAnswer?.(msg);
				ws.close();
				resolveFinished();
				break;
			case "cancelled":
				callbacks.onCancelled?.();
				ws.close();
				resolveFinished();
				break;
			case "error":
				callbacks.onError?.(msg);
				ws.close();
				rejectFinished(new Error(msg.message || "server error"));
				break;
			case "pong":
				// ignore
				break;
		}
	});

	ws.addEventListener("error", () => {
		rejectFinished(new Error("WebSocket error"));
	});

	ws.addEventListener("close", () => {
		// If we close without resolving, treat as completion.
		resolveFinished();
	});

	return {
		cancel: () => {
			try {
				if (ws.readyState === WebSocket.OPEN) {
					ws.send(JSON.stringify({ type: "cancel" }));
				}
			} catch {
				// swallow
			}
		},
		finished,
	};
}

async function streamSSE(
	url: string,
	body: object,
	onEvent: (ev: SSEEvent) => void,
	opts: { signal?: AbortSignal } = {},
): Promise<void> {
	const response = await fetch(url, {
		method: "POST",
		headers: {
			"Content-Type": "application/json",
			"Accept": "text/event-stream",
		},
		body: JSON.stringify(body),
		signal: opts.signal,
	});
	if (!response.ok) {
		throw new Error(`SSE endpoint returned ${response.status}`);
	}
	if (!response.body) {
		throw new Error("SSE response has no body stream");
	}

	const reader = response.body.getReader();
	const decoder = new TextDecoder();
	let buf = "";

	try {
		while (true) {
			const { value, done } = await reader.read();
			if (done) break;
			buf += decoder.decode(value, { stream: true });

			// Split on double newline — each SSE frame ends with "\n\n".
			let idx: number;
			while ((idx = buf.indexOf("\n\n")) !== -1) {
				const frame = buf.slice(0, idx);
				buf = buf.slice(idx + 2);
				if (!frame.trim()) continue;

				let eventName = "message";
				let dataStr = "";
				for (const line of frame.split("\n")) {
					if (line.startsWith("event: ")) {
						eventName = line.slice(7).trim();
					} else if (line.startsWith("data: ")) {
						dataStr += line.slice(6);
					}
				}
				if (dataStr) {
					try {
						onEvent({ event: eventName, data: JSON.parse(dataStr) });
					} catch {
						onEvent({ event: eventName, data: dataStr });
					}
				}
			}
		}
	} finally {
		reader.releaseLock();
	}
}

/** Run a pagewiki CLI command and return stdout. */
function runPagewiki(
	app: App,
	settings: PageWikiSettings,
	args: string,
): Promise<string> {
	const vaultPath = (app.vault.adapter as any).basePath as string;
	const cmd = `${settings.pythonPath} -m pagewiki ${args} --vault "${vaultPath}"`;

	return new Promise((resolve, reject) => {
		exec(
			cmd,
			{ timeout: 600_000, maxBuffer: 10 * 1024 * 1024, env: { ...process.env, NO_COLOR: "1" } },
			(error, stdout, stderr) => {
				if (error && !stdout) {
					reject(new Error(stderr || error.message));
				} else {
					resolve(stripAnsi(stdout));
				}
			},
		);
	});
}

// ─────────────────────────────────────────────────────────────────────────────
// Results Modal
// ─────────────────────────────────────────────────────────────────────────────

class ResultModal extends Modal {
	private title: string;
	private content: string;

	constructor(app: App, title: string, content: string) {
		super(app);
		this.title = title;
		this.content = content;
	}

	onOpen(): void {
		const { contentEl } = this;
		contentEl.createEl("h2", { text: this.title });

		const pre = contentEl.createEl("pre", {
			cls: "pagewiki-result",
		});
		pre.style.whiteSpace = "pre-wrap";
		pre.style.maxHeight = "70vh";
		pre.style.overflow = "auto";
		pre.style.fontSize = "13px";
		pre.style.lineHeight = "1.5";
		pre.setText(this.content);

		const copyBtn = contentEl.createEl("button", { text: "Copy to clipboard" });
		copyBtn.style.marginTop = "12px";
		copyBtn.addEventListener("click", () => {
			navigator.clipboard.writeText(this.content);
			new Notice("Copied to clipboard");
		});
	}

	onClose(): void {
		this.contentEl.empty();
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Ask Input Modal
// ─────────────────────────────────────────────────────────────────────────────

class AskModal extends Modal {
	private settings: PageWikiSettings;
	private onSubmit: (query: string, decompose: boolean) => void;
	private decomposeChecked: boolean;

	constructor(
		app: App,
		settings: PageWikiSettings,
		onSubmit: (q: string, decompose: boolean) => void,
	) {
		super(app);
		this.settings = settings;
		this.onSubmit = onSubmit;
		this.decomposeChecked = settings.decomposeByDefault;
	}

	onOpen(): void {
		const { contentEl } = this;
		contentEl.createEl("h2", { text: "PageWiki Ask" });
		contentEl.createEl("p", {
			text: `Model: ${this.settings.model} | Folder: ${this.settings.folder}`,
			cls: "setting-item-description",
		});

		let queryText = "";
		const textArea = new TextAreaComponent(contentEl);
		textArea.setPlaceholder("질문을 입력하세요...");
		textArea.inputEl.style.width = "100%";
		textArea.inputEl.style.height = "100px";
		textArea.inputEl.style.marginBottom = "12px";
		textArea.onChange((value) => {
			queryText = value;
		});

		// v0.7 decompose toggle (per-query override).
		const controlsRow = contentEl.createDiv();
		controlsRow.style.display = "flex";
		controlsRow.style.alignItems = "center";
		controlsRow.style.gap = "12px";
		controlsRow.style.marginBottom = "12px";

		const decomposeLabel = controlsRow.createEl("label");
		decomposeLabel.style.display = "flex";
		decomposeLabel.style.alignItems = "center";
		decomposeLabel.style.gap = "6px";
		decomposeLabel.style.fontSize = "13px";
		decomposeLabel.style.color = "var(--text-muted)";

		const decomposeInput = decomposeLabel.createEl("input", {
			type: "checkbox",
		}) as HTMLInputElement;
		decomposeInput.checked = this.decomposeChecked;
		decomposeInput.addEventListener("change", () => {
			this.decomposeChecked = decomposeInput.checked;
		});
		decomposeLabel.appendText("Decompose complex query (v0.7)");

		const submitBtn = controlsRow.createEl("button", {
			text: "Ask",
			cls: "mod-cta",
		});
		submitBtn.style.marginLeft = "auto";
		submitBtn.addEventListener("click", () => {
			if (queryText.trim()) {
				this.close();
				this.onSubmit(queryText.trim(), this.decomposeChecked);
			}
		});

		// Enter key submits (Shift+Enter for newline)
		textArea.inputEl.addEventListener("keydown", (e: KeyboardEvent) => {
			if (e.key === "Enter" && !e.shiftKey) {
				e.preventDefault();
				if (queryText.trim()) {
					this.close();
					this.onSubmit(queryText.trim(), this.decomposeChecked);
				}
			}
		});

		textArea.inputEl.focus();
	}

	onClose(): void {
		this.contentEl.empty();
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Chat Modal (v0.6)
// ─────────────────────────────────────────────────────────────────────────────

interface ChatMessage {
	role: "user" | "assistant";
	text: string;
}

class ChatModal extends Modal {
	private settings: PageWikiSettings;
	private messages: ChatMessage[] = [];
	private messagesEl!: HTMLElement;
	private inputEl!: HTMLTextAreaElement;
	private submitBtn!: HTMLButtonElement;
	private cancelBtn!: HTMLButtonElement;
	private busy = false;
	// v0.12 server-mode session state.
	private serverSessionId: string | null = null;
	// v0.13 currently-running WebSocket handle (if any).
	private activeWsHandle: { cancel: () => void } | null = null;

	constructor(app: App, settings: PageWikiSettings) {
		super(app);
		this.settings = settings;
	}

	onOpen(): void {
		const { contentEl } = this;
		contentEl.addClass("pagewiki-chat-modal");

		contentEl.createEl("h2", { text: "PageWiki Chat" });
		contentEl.createEl("p", {
			text: `Model: ${this.settings.model} | Folder: ${this.settings.folder}`,
			cls: "setting-item-description",
		});

		// Messages container
		this.messagesEl = contentEl.createDiv({ cls: "pagewiki-chat-messages" });
		this.messagesEl.style.maxHeight = "50vh";
		this.messagesEl.style.overflowY = "auto";
		this.messagesEl.style.marginBottom = "12px";
		this.messagesEl.style.padding = "8px";
		this.messagesEl.style.border = "1px solid var(--background-modifier-border)";
		this.messagesEl.style.borderRadius = "6px";

		this._appendSystemMessage("대화를 시작하세요. 후속 질문이 이전 맥락을 이어받습니다.");

		// Input area
		const inputWrapper = contentEl.createDiv();
		inputWrapper.style.display = "flex";
		inputWrapper.style.gap = "8px";

		const ta = new TextAreaComponent(inputWrapper);
		ta.setPlaceholder("질문을 입력하세요... (Enter 전송, Shift+Enter 줄바꿈)");
		ta.inputEl.style.flex = "1";
		ta.inputEl.style.height = "60px";
		ta.inputEl.style.resize = "vertical";
		this.inputEl = ta.inputEl;

		this.submitBtn = inputWrapper.createEl("button", {
			text: "Send",
			cls: "mod-cta",
		});
		this.submitBtn.style.alignSelf = "flex-end";
		this.submitBtn.addEventListener("click", () => this._handleSubmit());

		// v0.13 Cancel button — only useful when server mode is on
		// and a WebSocket query is in flight.
		this.cancelBtn = inputWrapper.createEl("button", {
			text: "Cancel",
		});
		this.cancelBtn.style.alignSelf = "flex-end";
		this.cancelBtn.disabled = true;
		this.cancelBtn.addEventListener("click", () => {
			if (this.activeWsHandle) {
				this.activeWsHandle.cancel();
				this._appendSystemMessage("Cancel requested — waiting for server...");
			}
		});

		this.inputEl.addEventListener("keydown", (e: KeyboardEvent) => {
			if (e.key === "Enter" && !e.shiftKey) {
				e.preventDefault();
				this._handleSubmit();
			}
		});

		this.inputEl.focus();
	}

	onClose(): void {
		this.contentEl.empty();
	}

	private _appendSystemMessage(text: string): void {
		const el = this.messagesEl.createDiv({ cls: "pagewiki-chat-system" });
		el.style.color = "var(--text-muted)";
		el.style.fontStyle = "italic";
		el.style.fontSize = "12px";
		el.style.marginBottom = "8px";
		el.setText(text);
	}

	private _appendMessage(msg: ChatMessage): void {
		this.messages.push(msg);

		const wrapper = this.messagesEl.createDiv({ cls: "pagewiki-chat-msg" });
		wrapper.style.marginBottom = "12px";

		const label = wrapper.createEl("strong");
		label.style.color = msg.role === "user"
			? "var(--text-accent)"
			: "var(--text-success)";
		label.setText(msg.role === "user" ? "Q: " : "A: ");

		const body = wrapper.createEl("span");
		body.style.whiteSpace = "pre-wrap";
		body.setText(msg.text);

		this.messagesEl.scrollTop = this.messagesEl.scrollHeight;
	}

	private _buildContextQuery(query: string): string {
		// Prepend recent history as context so the single `ask` call
		// can benefit from prior turns. Limited to last 3 turns.
		if (this.messages.length === 0) return query;

		const pairs: string[] = [];
		const msgs = this.messages;
		let i = msgs.length - 1;
		let turns = 0;
		while (i >= 0 && turns < 3) {
			if (msgs[i].role === "assistant" && i > 0 && msgs[i - 1].role === "user") {
				pairs.unshift(`Previous Q: ${msgs[i - 1].text}\nPrevious A: ${msgs[i].text.substring(0, 300)}`);
				i -= 2;
				turns++;
			} else {
				i--;
			}
		}

		if (pairs.length === 0) return query;
		return `[대화 맥락]\n${pairs.join("\n\n")}\n\n[현재 질문]\n${query}`;
	}

	private async _handleSubmit(): Promise<void> {
		const query = this.inputEl.value.trim();
		if (!query || this.busy) return;

		this.inputEl.value = "";
		this._appendMessage({ role: "user", text: query });

		this.busy = true;
		this.submitBtn.setText("...");
		this.submitBtn.disabled = true;

		try {
			// v0.12 server mode — stream from /chat/stream if configured.
			if (this.settings.serverUrl.trim()) {
				await this._submitServerMode(query);
			} else {
				await this._submitCliMode(query);
			}
		} catch (e: any) {
			this._appendSystemMessage(`Error: ${e.message}`);
		} finally {
			this.busy = false;
			this.submitBtn.setText("Send");
			this.submitBtn.disabled = false;
			this.inputEl.focus();
		}
	}

	private async _submitCliMode(query: string): Promise<void> {
		const contextQuery = this._buildContextQuery(query);
		const escaped = contextQuery.replace(/"/g, '\\"');
		let args =
			`ask "${escaped}" --folder "${this.settings.folder}" ` +
			`--model "${this.settings.model}" --num-ctx ${this.settings.numCtx} ` +
			`--max-workers ${this.settings.maxWorkers}`;
		if (this.settings.decomposeByDefault) {
			args += " --decompose";
		}
		if (this.settings.showUsage) {
			args += " --usage";
		}
		if (this.settings.maxTokens > 0) {
			args += ` --max-tokens ${this.settings.maxTokens}`;
		}
		if (this.settings.jsonMode) {
			args += " --json-mode";
		}
		if (this.settings.reuseContext) {
			args += " --reuse-context";
		}
		const output = await runPagewiki(this.app, this.settings, args);
		const answer = this._extractAnswer(output);
		this._appendMessage({ role: "assistant", text: answer });
	}

	private async _submitServerMode(query: string): Promise<void> {
		const base = this.settings.serverUrl.trim().replace(/\/$/, "");

		// Placeholder assistant bubble we mutate in-place as tokens stream.
		const placeholder = { role: "assistant" as const, text: "…" };
		this._appendMessage(placeholder);
		const bodySpan = this.messagesEl.lastElementChild?.querySelector(
			"span",
		) as HTMLSpanElement | null;

		let answer = "";
		const cited: string[] = [];
		let cancelled = false;

		// v0.13: when useWebSocket is enabled, use /ask/ws for
		// cancellation support. Chat mode history isn't yet
		// bridged through WebSocket, so we prepend context manually
		// (same pattern as CLI fallback).
		if (this.settings.useWebSocket) {
			const wsUrl = base.replace(/^http/, "ws") + "/ask/ws";
			const contextQuery = this._buildContextQuery(query);

			this.cancelBtn.disabled = false;

			const handle = connectAskWS(
				wsUrl,
				contextQuery,
				{
					decompose: this.settings.decomposeByDefault,
					maxTokens: this.settings.maxTokens,
					tokenSplit: this.settings.tokenSplit,
					jsonMode: this.settings.jsonMode,
					reuseContext: this.settings.reuseContext,
					promptCache: this.settings.promptCacheWebSocket,
				},
				{
					onTrace: (data) => {
						if (bodySpan) {
							bodySpan.setText(
								`[${data.phase}] ${data.detail?.substring(0, 80) ?? ""}`,
							);
						}
					},
					onAnswer: (data) => {
						answer = data.answer || "";
						if (Array.isArray(data.cited_nodes)) {
							cited.push(...data.cited_nodes);
						}
					},
					onCancelled: () => {
						cancelled = true;
					},
				},
			);

			this.activeWsHandle = handle;
			try {
				await handle.finished;
			} finally {
				this.activeWsHandle = null;
				this.cancelBtn.disabled = true;
			}

			if (cancelled) {
				answer = answer || "[cancelled]";
			}
		} else {
			// SSE path (v0.12 default).
			const url = `${base}/chat/stream`;
			await streamSSE(
				url,
				{
					query,
					session_id: this.serverSessionId,
					decompose: this.settings.decomposeByDefault,
				},
				(ev) => {
					if (ev.event === "trace" && bodySpan) {
						bodySpan.setText(
							`[${ev.data.phase}] ${ev.data.detail?.substring(0, 80) ?? ""}`,
						);
					} else if (ev.event === "answer") {
						answer = ev.data.answer || "";
						if (Array.isArray(ev.data.cited_nodes)) {
							cited.push(...ev.data.cited_nodes);
						}
						if (typeof ev.data.session_id === "string") {
							this.serverSessionId = ev.data.session_id;
						}
					} else if (ev.event === "error") {
						throw new Error(ev.data.message || "server error");
					}
				},
			);
		}

		if (bodySpan) {
			bodySpan.setText(
				answer +
					(cited.length > 0
						? `\n\nCited:\n${cited.map((c) => `  • ${c}`).join("\n")}`
						: ""),
			);
		}
		placeholder.text = answer;
	}

	/** Parse the answer from pagewiki ask CLI output. */
	private _extractAnswer(output: string): string {
		const lines = output.split("\n");
		let answerStart = -1;
		let citedStart = -1;

		for (let i = 0; i < lines.length; i++) {
			if (lines[i].startsWith("A: ") || lines[i].startsWith("A:")) {
				answerStart = i;
			}
			if (lines[i].startsWith("Cited nodes:") || lines[i].startsWith("Cited:")) {
				citedStart = i;
				break;
			}
		}

		if (answerStart === -1) return output.trim();

		const endIdx = citedStart !== -1 ? citedStart : lines.length;
		const answerLines = lines.slice(answerStart, endIdx);

		// Strip the "A: " prefix from the first line
		if (answerLines.length > 0) {
			answerLines[0] = answerLines[0].replace(/^A:\s*/, "");
		}

		let answer = answerLines.join("\n").trim();

		// Append cited nodes if present
		if (citedStart !== -1) {
			const citedLines = lines.slice(citedStart).filter((l) => l.trim());
			if (citedLines.length > 1) {
				answer += "\n\n" + citedLines.join("\n");
			}
		}

		return answer || output.trim();
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Settings Tab
// ─────────────────────────────────────────────────────────────────────────────

class PageWikiSettingTab extends PluginSettingTab {
	plugin: PageWikiPlugin;

	constructor(app: App, plugin: PageWikiPlugin) {
		super(app, plugin);
		this.plugin = plugin;
	}

	display(): void {
		const { containerEl } = this;
		containerEl.empty();
		containerEl.createEl("h2", { text: "PageWiki Settings" });

		new Setting(containerEl)
			.setName("Python path")
			.setDesc("Path to the Python interpreter with pagewiki installed.")
			.addText((text) =>
				text
					.setPlaceholder("python")
					.setValue(this.plugin.settings.pythonPath)
					.onChange(async (value) => {
						this.plugin.settings.pythonPath = value;
						await this.plugin.saveSettings();
					}),
			);

		new Setting(containerEl)
			.setName("Model")
			.setDesc("LiteLLM model id (e.g. ollama/gemma4:26b).")
			.addText((text) =>
				text
					.setPlaceholder("ollama/gemma4:26b")
					.setValue(this.plugin.settings.model)
					.onChange(async (value) => {
						this.plugin.settings.model = value;
						await this.plugin.saveSettings();
					}),
			);

		new Setting(containerEl)
			.setName("Context window")
			.setDesc("Ollama num_ctx parameter.")
			.addText((text) =>
				text
					.setPlaceholder("131072")
					.setValue(String(this.plugin.settings.numCtx))
					.onChange(async (value) => {
						const num = parseInt(value, 10);
						if (!isNaN(num)) {
							this.plugin.settings.numCtx = num;
							await this.plugin.saveSettings();
						}
					}),
			);

		new Setting(containerEl)
			.setName("Default folder")
			.setDesc("Subfolder inside the vault to scan/ask against.")
			.addText((text) =>
				text
					.setPlaceholder("Research")
					.setValue(this.plugin.settings.folder)
					.onChange(async (value) => {
						this.plugin.settings.folder = value;
						await this.plugin.saveSettings();
					}),
			);

		new Setting(containerEl)
			.setName("Max parallel LLM workers")
			.setDesc(
				"Number of concurrent LLM calls during summarization and " +
				"compilation (v0.7). Higher values speed up bulk work but " +
				"may thrash VRAM. Default: 4.",
			)
			.addText((text) =>
				text
					.setPlaceholder("4")
					.setValue(String(this.plugin.settings.maxWorkers))
					.onChange(async (value) => {
						const num = parseInt(value, 10);
						if (!isNaN(num) && num >= 1) {
							this.plugin.settings.maxWorkers = num;
							await this.plugin.saveSettings();
						}
					}),
			);

		new Setting(containerEl)
			.setName("Decompose complex queries by default")
			.setDesc(
				"When enabled, every Ask request uses --decompose to split " +
				"complex questions into sub-queries (v0.7). Slower but " +
				"produces better answers on multi-part questions.",
			)
			.addToggle((toggle) =>
				toggle
					.setValue(this.plugin.settings.decomposeByDefault)
					.onChange(async (value) => {
						this.plugin.settings.decomposeByDefault = value;
						await this.plugin.saveSettings();
					}),
			);

		// ── v0.9 settings ──────────────────────────────────────────────
		containerEl.createEl("h3", { text: "v0.9 Token Budget" });

		new Setting(containerEl)
			.setName("Show token usage")
			.setDesc(
				"Append --usage to Ask/Chat so the CLI prints a token " +
				"usage breakdown after each query (v0.9).",
			)
			.addToggle((toggle) =>
				toggle
					.setValue(this.plugin.settings.showUsage)
					.onChange(async (value) => {
						this.plugin.settings.showUsage = value;
						await this.plugin.saveSettings();
					}),
			);

		new Setting(containerEl)
			.setName("Max tokens per query")
			.setDesc(
				"Hard cap on total tokens for one query. 0 = unlimited. " +
				"Loop aborts cleanly when the cap is hit (v0.9).",
			)
			.addText((text) =>
				text
					.setPlaceholder("0")
					.setValue(String(this.plugin.settings.maxTokens))
					.onChange(async (value) => {
						const num = parseInt(value, 10);
						if (!isNaN(num) && num >= 0) {
							this.plugin.settings.maxTokens = num;
							await this.plugin.saveSettings();
						}
					}),
			);

		// ── v0.10 settings ─────────────────────────────────────────────
		containerEl.createEl("h3", { text: "v0.10 Reliability & Efficiency" });

		new Setting(containerEl)
			.setName("JSON-mode prompts")
			.setDesc(
				"Ask the LLM to respond in strict JSON for SELECT/EVALUATE " +
				"phases. Auto-falls-back to text parser if JSON fails twice " +
				"(v0.10).",
			)
			.addToggle((toggle) =>
				toggle
					.setValue(this.plugin.settings.jsonMode)
					.onChange(async (value) => {
						this.plugin.settings.jsonMode = value;
						await this.plugin.saveSettings();
					}),
			);

		new Setting(containerEl)
			.setName("Reuse context")
			.setDesc(
				"Compact path_so_far and suppress already-shown candidates " +
				"on deep loops to shorten prompts (v0.10).",
			)
			.addToggle((toggle) =>
				toggle
					.setValue(this.plugin.settings.reuseContext)
					.onChange(async (value) => {
						this.plugin.settings.reuseContext = value;
						await this.plugin.saveSettings();
					}),
			);

		// ── v0.12 server mode ──────────────────────────────────────────
		containerEl.createEl("h3", { text: "v0.12 Server Mode (optional)" });

		new Setting(containerEl)
			.setName("Server URL")
			.setDesc(
				"When set, Ask/Chat stream SSE from this `pagewiki serve` " +
				"URL instead of spawning a Python subprocess. " +
				"Example: http://localhost:8000 (v0.12). Leave blank for CLI mode.",
			)
			.addText((text) =>
				text
					.setPlaceholder("http://localhost:8000")
					.setValue(this.plugin.settings.serverUrl)
					.onChange(async (value) => {
						this.plugin.settings.serverUrl = value;
						await this.plugin.saveSettings();
					}),
			);

		new Setting(containerEl)
			.setName("Use WebSocket (Cancel support)")
			.setDesc(
				"When server mode is active, use /ask/ws (WebSocket) instead " +
				"of /chat/stream (SSE) so in-flight queries can be cancelled " +
				"via the Cancel button in the Chat modal (v0.13).",
			)
			.addToggle((toggle) =>
				toggle
					.setValue(this.plugin.settings.useWebSocket)
					.onChange(async (value) => {
						this.plugin.settings.useWebSocket = value;
						await this.plugin.saveSettings();
					}),
			);

		new Setting(containerEl)
			.setName("Token split (WebSocket mode)")
			.setDesc(
				"Per-phase token budget split — SUMMARIZE:RETRIEVE:SYNTH, " +
				"e.g. '20:60:20'. Only applied when Max tokens > 0 AND " +
				"WebSocket mode is active. Leave blank to use the flat cap (v0.15).",
			)
			.addText((text) =>
				text
					.setPlaceholder("20:60:20")
					.setValue(this.plugin.settings.tokenSplit)
					.onChange(async (value) => {
						this.plugin.settings.tokenSplit = value;
						await this.plugin.saveSettings();
					}),
			);

		new Setting(containerEl)
			.setName("Prompt cache (WebSocket mode)")
			.setDesc(
				"Ask the server to use its prompt-cache chat_fn for this " +
				"connection. Requires `pagewiki serve --prompt-cache` on " +
				"the server side; ignored otherwise (v0.16).",
			)
			.addToggle((toggle) =>
				toggle
					.setValue(this.plugin.settings.promptCacheWebSocket)
					.onChange(async (value) => {
						this.plugin.settings.promptCacheWebSocket = value;
						await this.plugin.saveSettings();
					}),
			);
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Plugin
// ─────────────────────────────────────────────────────────────────────────────

export default class PageWikiPlugin extends Plugin {
	settings: PageWikiSettings = DEFAULT_SETTINGS;
	private statusBarEl: HTMLElement | null = null;
	private watchInterval: ReturnType<typeof setInterval> | null = null;

	async onload(): Promise<void> {
		await this.loadSettings();

		// Status bar item for watch mode
		this.statusBarEl = this.addStatusBarItem();
		this.statusBarEl.setText("PageWiki: idle");

		// Command: Scan
		this.addCommand({
			id: "pagewiki-scan",
			name: "Scan vault",
			callback: () => this.runScan(),
		});

		// Command: Scan with graph
		this.addCommand({
			id: "pagewiki-scan-graph",
			name: "Scan vault + wiki-link graph",
			callback: () => this.runScan(true),
		});

		// Command: Ask
		this.addCommand({
			id: "pagewiki-ask",
			name: "Ask a question",
			callback: () => this.openAskModal(),
		});

		// Command: Chat (v0.6)
		this.addCommand({
			id: "pagewiki-chat",
			name: "Chat (multi-turn conversation)",
			callback: () => this.openChatModal(),
		});

		// Command: Compile LLM-Wiki
		this.addCommand({
			id: "pagewiki-compile",
			name: "Compile LLM-Wiki",
			callback: () => this.runCompile(),
		});

		// Command: List vaults
		this.addCommand({
			id: "pagewiki-vaults",
			name: "List discovered vaults",
			callback: () => this.runVaults(),
		});

		// Command: Toggle watch mode
		this.addCommand({
			id: "pagewiki-watch-toggle",
			name: "Toggle watch mode",
			callback: () => this.toggleWatch(),
		});

		// Settings tab
		this.addSettingTab(new PageWikiSettingTab(this.app, this));

		// Ribbon icon
		this.addRibbonIcon("search", "PageWiki Ask", () => {
			this.openAskModal();
		});
	}

	onunload(): void {
		this.stopWatch();
	}

	async loadSettings(): Promise<void> {
		this.settings = Object.assign({}, DEFAULT_SETTINGS, await this.loadData());
	}

	async saveSettings(): Promise<void> {
		await this.saveData(this.settings);
	}

	private async runScan(showGraph = false): Promise<void> {
		const notice = new Notice("PageWiki: Scanning...", 0);
		try {
			let args = `scan --folder "${this.settings.folder}"`;
			if (showGraph) args += " --show-graph";
			const output = await runPagewiki(this.app, this.settings, args);
			notice.hide();
			new ResultModal(this.app, "Scan Results", output).open();
		} catch (e: any) {
			notice.hide();
			new Notice(`PageWiki scan failed: ${e.message}`, 10000);
		}
	}

	private openAskModal(): void {
		new AskModal(
			this.app,
			this.settings,
			(query, decompose) => this.runAsk(query, decompose),
		).open();
	}

	private openChatModal(): void {
		new ChatModal(this.app, this.settings).open();
	}

	private async runAsk(query: string, decompose?: boolean): Promise<void> {
		const useDecompose = decompose ?? this.settings.decomposeByDefault;

		// v0.12 server mode — SSE stream from a running `pagewiki serve`.
		if (this.settings.serverUrl.trim()) {
			await this._runAskServerMode(query, useDecompose);
			return;
		}

		const notice = new Notice("PageWiki: Thinking...", 0);
		try {
			const escaped = query.replace(/"/g, '\\"');
			let args =
				`ask "${escaped}" --folder "${this.settings.folder}" ` +
				`--model "${this.settings.model}" --num-ctx ${this.settings.numCtx} ` +
				`--max-workers ${this.settings.maxWorkers}`;
			if (useDecompose) {
				args += " --decompose";
			}
			// v0.9+ flags
			if (this.settings.showUsage) {
				args += " --usage";
			}
			if (this.settings.maxTokens > 0) {
				args += ` --max-tokens ${this.settings.maxTokens}`;
			}
			// v0.10+ flags
			if (this.settings.jsonMode) {
				args += " --json-mode";
			}
			if (this.settings.reuseContext) {
				args += " --reuse-context";
			}
			const output = await runPagewiki(this.app, this.settings, args);
			notice.hide();
			new ResultModal(this.app, "Answer", output).open();
		} catch (e: any) {
			notice.hide();
			new Notice(`PageWiki ask failed: ${e.message}`, 10000);
		}
	}

	/** v0.12 server-mode: stream /ask/stream and render the final answer. */
	private async _runAskServerMode(
		query: string,
		decompose: boolean,
	): Promise<void> {
		const base = this.settings.serverUrl.trim().replace(/\/$/, "");
		const url = `${base}/ask/stream`;

		const notice = new Notice("PageWiki: Streaming answer...", 0);
		let answer = "";
		const cited: string[] = [];
		let traceCount = 0;
		let tokenTotal = 0;

		try {
			await streamSSE(
				url,
				{
					query,
					decompose,
				},
				(ev) => {
					if (ev.event === "trace") {
						traceCount++;
						// Update the notice so users see progress.
						notice.setMessage(
							`PageWiki: ${traceCount} steps, ${tokenTotal} tokens`,
						);
					} else if (ev.event === "usage") {
						tokenTotal = ev.data.total_tokens || 0;
					} else if (ev.event === "answer") {
						answer = ev.data.answer || "";
						if (Array.isArray(ev.data.cited_nodes)) {
							cited.push(...ev.data.cited_nodes);
						}
					} else if (ev.event === "error") {
						throw new Error(ev.data.message || "server error");
					}
				},
			);
		} catch (e: any) {
			notice.hide();
			new Notice(`PageWiki server error: ${e.message}`, 10000);
			return;
		}

		notice.hide();
		const body =
			`A: ${answer}\n\n` +
			(cited.length > 0 ? `Cited:\n${cited.map((c) => `  • ${c}`).join("\n")}\n\n` : "") +
			`[server mode: ${traceCount} trace events, ${tokenTotal} tokens]`;
		new ResultModal(this.app, "Answer (server mode)", body).open();
	}

	private async runCompile(): Promise<void> {
		const notice = new Notice("PageWiki: Compiling LLM-Wiki...", 0);
		try {
			const args =
				`compile --folder "${this.settings.folder}" ` +
				`--model "${this.settings.model}" --num-ctx ${this.settings.numCtx} ` +
				`--max-workers ${this.settings.maxWorkers}`;
			const output = await runPagewiki(this.app, this.settings, args);
			notice.hide();
			new ResultModal(this.app, "LLM-Wiki Compiled", output).open();
		} catch (e: any) {
			notice.hide();
			new Notice(`PageWiki compile failed: ${e.message}`, 10000);
		}
	}

	private async runVaults(): Promise<void> {
		try {
			const cmd = `${this.settings.pythonPath} -m pagewiki vaults`;
			const output = await new Promise<string>((resolve, reject) => {
				exec(
					cmd,
					{ timeout: 10_000, env: { ...process.env, NO_COLOR: "1" } },
					(error, stdout, stderr) => {
						if (error && !stdout) reject(new Error(stderr || error.message));
						else resolve(stripAnsi(stdout));
					},
				);
			});
			new ResultModal(this.app, "Discovered Vaults", output).open();
		} catch (e: any) {
			new Notice(`PageWiki vaults failed: ${e.message}`, 10000);
		}
	}

	private toggleWatch(): void {
		if (this.watchInterval) {
			this.stopWatch();
			new Notice("PageWiki: Watch stopped");
		} else {
			this.startWatch();
			new Notice("PageWiki: Watch started");
		}
	}

	/** Build the env dict used by watch commands — passes vault/folder
	 *  via environment variables so special characters (apostrophes,
	 *  spaces, backslashes) never break the embedded Python string. */
	private _watchEnv(): Record<string, string> {
		const vaultPath = (this.app.vault.adapter as any).basePath as string;
		return {
			...process.env as Record<string, string>,
			NO_COLOR: "1",
			PAGEWIKI_VAULT: vaultPath,
			PAGEWIKI_FOLDER: this.settings.folder || "",
		};
	}

	/** Seed ``scan-state.json`` so the first poll only reports real deltas. */
	private _seedSnapshot(): Promise<void> {
		const cmd =
			`${this.settings.pythonPath} -c "` +
			`import os; from pathlib import Path; ` +
			`from pagewiki.watcher import save_state; ` +
			`save_state(Path(os.environ['PAGEWIKI_VAULT']), os.environ.get('PAGEWIKI_FOLDER') or None)"`;

		return new Promise((resolve, reject) => {
			exec(
				cmd,
				{ timeout: 30_000, env: this._watchEnv() },
				(error, _stdout, stderr) => {
					if (error) reject(new Error(stderr || error.message));
					else resolve();
				},
			);
		});
	}

	private async startWatch(): Promise<void> {
		if (this.watchInterval) return;

		// Seed the mtime snapshot before polling so the first cycle
		// only reports actual deltas, not the entire vault as "added".
		try {
			await this._seedSnapshot();
		} catch {
			new Notice("PageWiki: failed to seed watch snapshot", 5000);
		}

		if (this.statusBarEl) {
			this.statusBarEl.setText("PageWiki: watching");
			this.statusBarEl.style.color = "var(--text-success)";
		}

		this.watchInterval = setInterval(async () => {
			try {
				const cmd =
					`${this.settings.pythonPath} -c "` +
					`import os; from pathlib import Path; ` +
					`from pagewiki.watcher import detect_changes, save_state; ` +
					`v = Path(os.environ['PAGEWIKI_VAULT']); ` +
					`f = os.environ.get('PAGEWIKI_FOLDER') or None; ` +
					`cs = detect_changes(v, f); ` +
					`print(f'{len(cs.added)}|{len(cs.modified)}|{len(cs.deleted)}'); ` +
					`save_state(v, f) if cs.has_changes else None"`;

				const output = await new Promise<string>((resolve, reject) => {
					exec(
						cmd,
						{ timeout: 15_000, env: this._watchEnv() },
						(error, stdout, stderr) => {
							if (error) reject(new Error(stderr || error.message));
							else resolve(stdout.trim());
						},
					);
				});

				const [added, modified, deleted] = output.split("|").map(Number);
				const total = added + modified + deleted;

				if (total > 0 && this.statusBarEl) {
					this.statusBarEl.setText(
						`PageWiki: +${added} ~${modified} -${deleted}`,
					);
					new Notice(
						`PageWiki: ${total} file(s) changed (${added} new, ${modified} modified, ${deleted} deleted)`,
						5000,
					);
				}
			} catch {
				// Silently ignore poll errors
			}
		}, 10_000);
	}

	private stopWatch(): void {
		if (this.watchInterval) {
			clearInterval(this.watchInterval);
			this.watchInterval = null;
		}
		if (this.statusBarEl) {
			this.statusBarEl.setText("PageWiki: idle");
			this.statusBarEl.style.color = "";
		}
	}
}
