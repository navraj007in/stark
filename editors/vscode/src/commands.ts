import * as vscode from 'vscode';
import * as path from 'path';
import { runCompiler } from './compiler';
import { updateDiagnostics, clearAllDiagnostics } from './diagnostics';
import { getConfiguration } from './configuration';

export async function checkCurrentFile() {
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

export function runCurrentFile() {
  const editor = vscode.window.activeTextEditor;
  if (!editor || editor.document.languageId !== 'stark') {
    vscode.window.showInformationMessage('No active STARK file to run.');
    return;
  }

  // Save the document first if it's dirty
  if (editor.document.isDirty) {
    editor.document.save();
  }

  const document = editor.document;
  const config = getConfiguration();
  
  let terminal = vscode.window.terminals.find((t) => t.name === 'STARK Run');
  if (!terminal) {
    terminal = vscode.window.createTerminal('STARK Run');
  }
  
  terminal.show();
  terminal.sendText(`"${config.compilerPath}" run "${document.fileName}"`);
}

export function openInStarkIde() {
  const editor = vscode.window.activeTextEditor;
  if (!editor || editor.document.languageId !== 'stark') {
    vscode.window.showInformationMessage('No active STARK file to open.');
    return;
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
  terminal.sendText(`"${idePath}" "${document.fileName}"`);
}

export function restartCompiler(triggerDocumentCheck: (doc: vscode.TextDocument) => void) {
  clearAllDiagnostics();
  vscode.window.showInformationMessage('Restarted STARK compiler integration.');
  const editor = vscode.window.activeTextEditor;
  if (editor && editor.document.languageId === 'stark') {
    triggerDocumentCheck(editor.document);
  }
}
