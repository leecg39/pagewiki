/**
 * PageWiki Obsidian Plugin (v0.11)
 *
 * Wraps the pagewiki CLI so users can scan, ask, chat, compile, and watch
 * from within Obsidian — no terminal switching required.
 *
 * Commands are executed via Node child_process.exec against the
 * `pagewiki` CLI that the user has already pip-installed. Results
 * are displayed in Obsidian modals with Rich-markup stripped.
 *
 * Version history:
 *   - v0.7: maxWorkers + decomposeByDefault
 *   - v0.8: N/A (plugin-side)
 *   - v0.9: --usage, --max-tokens flags
 *   - v0.10: --json-mode, --reuse-context flags
 *   - v0.11: Chat streaming (when pagewiki serve is reachable)
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
};

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/** Strip ANSI escape codes and Rich markup from CLI output. */
function stripAnsi(text: string): string {
	// eslint-disable-next-line no-control-regex
	return text.replace(/\x1b\[[0-9;]*m/g, "").replace(/\[\/?\w+\]/g, "");
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
	private busy = false;

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
			const contextQuery = this._buildContextQuery(query);
			const escaped = contextQuery.replace(/"/g, '\\"');
			let args =
				`ask "${escaped}" --folder "${this.settings.folder}" ` +
				`--model "${this.settings.model}" --num-ctx ${this.settings.numCtx} ` +
				`--max-workers ${this.settings.maxWorkers}`;
			if (this.settings.decomposeByDefault) {
				args += " --decompose";
			}
			// v0.9+ flags propagated from settings.
			if (this.settings.showUsage) {
				args += " --usage";
			}
			if (this.settings.maxTokens > 0) {
				args += ` --max-tokens ${this.settings.maxTokens}`;
			}
			// v0.10+ flags.
			if (this.settings.jsonMode) {
				args += " --json-mode";
			}
			if (this.settings.reuseContext) {
				args += " --reuse-context";
			}
			const output = await runPagewiki(this.app, this.settings, args);

			// Extract the answer line from CLI output (after "A: ")
			const answer = this._extractAnswer(output);
			this._appendMessage({ role: "assistant", text: answer });
		} catch (e: any) {
			this._appendSystemMessage(`Error: ${e.message}`);
		} finally {
			this.busy = false;
			this.submitBtn.setText("Send");
			this.submitBtn.disabled = false;
			this.inputEl.focus();
		}
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
		const notice = new Notice("PageWiki: Thinking...", 0);
		try {
			const escaped = query.replace(/"/g, '\\"');
			const useDecompose = decompose ?? this.settings.decomposeByDefault;
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
