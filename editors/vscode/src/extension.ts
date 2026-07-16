import * as vscode from 'vscode';
import { runCompiler } from './compiler';
import { updateDiagnostics, clearDiagnostics, clearAllDiagnostics, starkDiagnosticCollection } from './diagnostics';
import { getConfiguration } from './configuration';
import * as commands from './commands';

let checkTimeout: NodeJS.Timeout | undefined;

export function activate(context: vscode.ExtensionContext) {
  // Register diagnostic collection
  context.subscriptions.push(starkDiagnosticCollection);

  // Register commands
  context.subscriptions.push(
    vscode.commands.registerCommand('stark.checkCurrentFile', commands.checkCurrentFile),
    vscode.commands.registerCommand('stark.runCurrentFile', commands.runCurrentFile),
    vscode.commands.registerCommand('stark.openInStarkIde', commands.openInStarkIde),
    vscode.commands.registerCommand('stark.restartCompiler', () => 
      commands.restartCompiler((doc) => triggerDocumentCheck(doc, true))
    )
  );

  // Helper to run check on a document
  async function triggerDocumentCheck(document: vscode.TextDocument, force = false) {
    if (document.languageId !== 'stark') {
      return;
    }

    // Security boundary: Only run compiler if the workspace is trusted
    if (!vscode.workspace.isTrusted) {
      return;
    }

    const config = getConfiguration();
    if (!force && !config.checkOnSave && !config.checkOnType) {
      return;
    }

    const result = await runCompiler(document.getText(), document.fileName);
    if (result.success && result.output) {
      updateDiagnostics(document, result.output.diagnostics);
    } else if (result.error) {
      console.error(result.error);
    }
  }

  // Check on open
  if (vscode.window.activeTextEditor) {
    triggerDocumentCheck(vscode.window.activeTextEditor.document);
  }

  context.subscriptions.push(
    vscode.window.onDidChangeActiveTextEditor((editor) => {
      if (editor) {
        triggerDocumentCheck(editor.document);
      }
    })
  );

  // Check on save
  context.subscriptions.push(
    vscode.workspace.onDidSaveTextDocument((document) => {
      const config = getConfiguration();
      if (config.checkOnSave) {
        triggerDocumentCheck(document, true);
      }
    })
  );

  // Check on type (with debounce)
  context.subscriptions.push(
    vscode.workspace.onDidChangeTextDocument((event) => {
      const config = getConfiguration();
      if (config.checkOnType && event.document.languageId === 'stark') {
        if (checkTimeout) {
          clearTimeout(checkTimeout);
        }
        checkTimeout = setTimeout(() => {
          triggerDocumentCheck(event.document, true);
        }, config.checkOnTypeDebounceMs);
      }
    })
  );

  // Clear diagnostics on close
  context.subscriptions.push(
    vscode.workspace.onDidCloseTextDocument((document) => {
      clearDiagnostics(document);
    })
  );

  // Handle Workspace Trust transitions
  context.subscriptions.push(
    vscode.workspace.onDidGrantWorkspaceTrust(() => {
      const editor = vscode.window.activeTextEditor;
      if (editor) {
        triggerDocumentCheck(editor.document);
      }
    })
  );
}

export function deactivate() {
  if (checkTimeout) {
    clearTimeout(checkTimeout);
  }
  clearAllDiagnostics();
}
