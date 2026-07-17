import * as vscode from 'vscode';
import * as path from 'path';
import { runCompiler } from './compiler';
import { updateDiagnostics, clearAllDiagnostics } from './diagnostics';
import { getConfiguration } from './configuration';
import { restartLspClient, showLspOutput } from './lspClient';

function escapeShellArg(arg: string): string {
  return arg.replace(/([\\"$`])/g, '\\$1');
}

export async function checkCurrentFile() {
  if (!vscode.workspace.isTrusted) {
    vscode.window.showErrorMessage('STARK command execution is disabled in untrusted workspaces.');
    return;
  }

  const editor = vscode.window.activeTextEditor;
  if (!editor || editor.document.languageId !== 'stark') {
    vscode.window.showInformationMessage('No active STARK file to check.');
    return;
  }

  const document = editor.document;
  await vscode.window.withProgress(
    {
      location: vscode.ProgressLocation.Notification,
      title: 'Running STARK Semantic Check...',
      cancellable: false,
    },
    async () => {
      const result = await runCompiler(document.getText(), document.fileName);
      if (result.success && result.output) {
        updateDiagnostics(document, result.output.diagnostics);
        if (result.output.diagnostics.length === 0) {
          vscode.window.showInformationMessage('STARK check: OK');
        }
      } else {
        vscode.window.showErrorMessage(`STARK check failed: ${result.error}`);
      }
    }
  );
}

export async function runCurrentFile() {
  if (!vscode.workspace.isTrusted) {
    vscode.window.showErrorMessage('STARK command execution is disabled in untrusted workspaces.');
    return;
  }

  const editor = vscode.window.activeTextEditor;
  if (!editor || editor.document.languageId !== 'stark') {
    vscode.window.showInformationMessage('No active STARK file to run.');
    return;
  }

  // Save the document first if it's dirty
  if (editor.document.isDirty) {
    const saved = await editor.document.save();
    if (!saved) {
      vscode.window.showWarningMessage('Run cancelled: File could not be saved.');
      return;
    }
  }

  const document = editor.document;
  const config = getConfiguration();
  
  let terminal = vscode.window.terminals.find((t) => t.name === 'STARK Run');
  if (!terminal) {
    terminal = vscode.window.createTerminal('STARK Run');
  }
  
  const extArgsList: string[] = [];
  for (const ext of config.extensions) {
    extArgsList.push('--extension', escapeShellArg(ext));
  }
  const extArgs = extArgsList.length > 0 ? ' ' + extArgsList.join(' ') : '';
  
  terminal.show();
  terminal.sendText(`"${escapeShellArg(config.compilerPath)}" run${extArgs} "${escapeShellArg(document.fileName)}"`);
}

export async function openInStarkIde() {
  if (!vscode.workspace.isTrusted) {
    vscode.window.showErrorMessage('STARK command execution is disabled in untrusted workspaces.');
    return;
  }

  const editor = vscode.window.activeTextEditor;
  if (!editor || editor.document.languageId !== 'stark') {
    vscode.window.showInformationMessage('No active STARK file to open.');
    return;
  }

  // Save the document first if it's dirty
  if (editor.document.isDirty) {
    const saved = await editor.document.save();
    if (!saved) {
      vscode.window.showWarningMessage('Open cancelled: File could not be saved.');
      return;
    }
  }

  const document = editor.document;
  const config = getConfiguration();

  let terminal = vscode.window.terminals.find((t) => t.name === 'STARK IDE');
  if (!terminal) {
    terminal = vscode.window.createTerminal('STARK IDE');
  }

  const compilerDir = path.dirname(config.compilerPath);
  let idePath = 'starkide';
  if (compilerDir !== '.' && path.isAbsolute(config.compilerPath)) {
    idePath = path.join(compilerDir, 'starkide');
  }

  terminal.show();
  terminal.sendText(`"${escapeShellArg(idePath)}" "${escapeShellArg(document.fileName)}"`);
}

export async function restartCompiler(triggerDocumentCheck: (doc: vscode.TextDocument) => void) {
  if (!vscode.workspace.isTrusted) {
    vscode.window.showErrorMessage('STARK command execution is disabled in untrusted workspaces.');
    return;
  }

  clearAllDiagnostics();
  await restartLspClient();
  vscode.window.showInformationMessage('Restarted STARK compiler integration.');
  const editor = vscode.window.activeTextEditor;
  if (editor && editor.document.languageId === 'stark') {
    triggerDocumentCheck(editor.document);
  }
}

export async function formatCurrentFile() {
  const editor = vscode.window.activeTextEditor;
  if (!editor || editor.document.languageId !== 'stark') {
    vscode.window.showInformationMessage('No active STARK file to format.');
    return;
  }
  await vscode.commands.executeCommand('editor.action.formatDocument');
}

export function showLanguageServerOutput() {
  showLspOutput();
}

export async function toggleTensorMode() {
  const config = vscode.workspace.getConfiguration('stark');
  const current = config.get<boolean>('tensorExtensionEnabled', false);
  await config.update('tensorExtensionEnabled', !current, vscode.ConfigurationTarget.Workspace);
  vscode.window.showInformationMessage(
    `STARK: tensor extension ${!current ? 'enabled' : 'disabled'}.`
  );
}
