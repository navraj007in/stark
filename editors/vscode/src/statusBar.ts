import * as vscode from 'vscode';
import { LspStatus, onLspStatusChange } from './lspClient';

let statusBarItem: vscode.StatusBarItem | undefined;
let lastStatus: LspStatus = 'stopped';

const ICONS: Record<LspStatus, string> = {
  stopped: '$(circle-slash)',
  starting: '$(sync~spin)',
  running: '$(check)',
  error: '$(error)',
};

const LABELS: Record<LspStatus, string> = {
  stopped: 'stopped',
  starting: 'starting…',
  running: 'running',
  error: 'error',
};

function render(): void {
  if (!statusBarItem) {
    return;
  }
  const editor = vscode.window.activeTextEditor;
  const isStark = editor?.document.languageId === 'stark';
  if (!isStark) {
    statusBarItem.hide();
    return;
  }
  statusBarItem.text = `${ICONS[lastStatus]} STARK`;
  statusBarItem.tooltip = `STARK language server: ${LABELS[lastStatus]}`;
  statusBarItem.show();
}

export function initStatusBar(context: vscode.ExtensionContext): void {
  statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
  statusBarItem.command = 'stark.restartLanguageServer';
  context.subscriptions.push(statusBarItem);

  onLspStatusChange((status) => {
    lastStatus = status;
    render();
  });

  context.subscriptions.push(vscode.window.onDidChangeActiveTextEditor(() => render()));
  render();
}
