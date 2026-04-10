/**
 * PageWiki Obsidian Plugin (v0.5)
 *
 * Wraps the pagewiki CLI so users can scan, ask, compile, and watch
 * from within Obsidian — no terminal switching required.
 *
 * Commands are executed via Node child_process.exec against the
 * `pagewiki` CLI that the user has already pip-installed. Results
 * are displayed in Obsidian modals with Rich-markup stripped.
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
}

const DEFAULT_SETTINGS: PageWikiSettings = {
	model: "ollama/gemma4:26b",
	numCtx: 131072,
	folder: "Research",
	pythonPath: "python",
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
	private onSubmit: (query: string) => void;

	constructor(app: App, settings: PageWikiSettings, onSubmit: (q: string) => void) {
		super(app);
		this.settings = settings;
		this.onSubmit = onSubmit;
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

		const submitBtn = contentEl.createEl("button", {
			text: "Ask",
			cls: "mod-cta",
		});
		submitBtn.addEventListener("click", () => {
			if (queryText.trim()) {
				this.close();
				this.onSubmit(queryText.trim());
			}
		});

		// Enter key submits (Shift+Enter for newline)
		textArea.inputEl.addEventListener("keydown", (e: KeyboardEvent) => {
			if (e.key === "Enter" && !e.shiftKey) {
				e.preventDefault();
				if (queryText.trim()) {
					this.close();
					this.onSubmit(queryText.trim());
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
		new AskModal(this.app, this.settings, (query) => this.runAsk(query)).open();
	}

	private async runAsk(query: string): Promise<void> {
		const notice = new Notice("PageWiki: Thinking...", 0);
		try {
			const escaped = query.replace(/"/g, '\\"');
			const args =
				`ask "${escaped}" --folder "${this.settings.folder}" ` +
				`--model "${this.settings.model}" --num-ctx ${this.settings.numCtx}`;
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
				`--model "${this.settings.model}" --num-ctx ${this.settings.numCtx}`;
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

	private startWatch(): void {
		if (this.watchInterval) return;

		// Save initial state
		runPagewiki(this.app, this.settings, `scan --folder "${this.settings.folder}"`)
			.catch(() => {});

		if (this.statusBarEl) {
			this.statusBarEl.setText("PageWiki: watching");
			this.statusBarEl.style.color = "var(--text-success)";
		}

		this.watchInterval = setInterval(async () => {
			try {
				const vaultPath = (this.app.vault.adapter as any).basePath as string;
				const cmd =
					`${this.settings.pythonPath} -c "` +
					`from pagewiki.watcher import detect_changes, save_state; ` +
					`cs = detect_changes(r'${vaultPath}', '${this.settings.folder}'); ` +
					`print(f'{len(cs.added)}|{len(cs.modified)}|{len(cs.deleted)}'); ` +
					`save_state(r'${vaultPath}', '${this.settings.folder}') if cs.has_changes else None"`;

				const output = await new Promise<string>((resolve, reject) => {
					exec(
						cmd,
						{ timeout: 15_000, env: { ...process.env, NO_COLOR: "1" } },
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
