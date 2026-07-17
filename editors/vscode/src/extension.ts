import * as vscode from 'vscode';
import { spawn } from 'child_process';
import * as path from 'path';
import { runCompiler } from './compiler';
import { updateDiagnostics, clearDiagnostics, clearAllDiagnostics, starkDiagnosticCollection } from './diagnostics';
import { getConfiguration } from './configuration';
import * as commands from './commands';
import { startLspClient, stopLspClient } from './lspClient';
import { initStatusBar } from './statusBar';

let checkTimeout: NodeJS.Timeout | undefined;

export async function activate(context: vscode.ExtensionContext) {
  // Register diagnostic collection
  context.subscriptions.push(starkDiagnosticCollection);

  initStatusBar(context);

  // Register commands
  context.subscriptions.push(
    vscode.commands.registerCommand('stark.checkCurrentFile', commands.checkCurrentFile),
    vscode.commands.registerCommand('stark.runCurrentFile', commands.runCurrentFile),
    vscode.commands.registerCommand('stark.openInStarkIde', commands.openInStarkIde),
    vscode.commands.registerCommand('stark.formatCurrentFile', commands.formatCurrentFile),
    vscode.commands.registerCommand('stark.toggleTensorMode', commands.toggleTensorMode),
    vscode.commands.registerCommand('stark.showLanguageServerOutput', commands.showLanguageServerOutput),
    vscode.commands.registerCommand('stark.restartLanguageServer', () =>
      commands.restartCompiler((doc) => triggerDocumentCheck(doc, true))
    ),
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
    if (!config.extensionEnabled) {
      return;
    }
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

  async function runTestsForDocument(document: vscode.TextDocument) {
    if (!vscode.workspace.isTrusted) {
      return;
    }
    const config = getConfiguration();
    // `stark test` (unlike `starkc check`/`starkc run`) does not currently
    // accept `--extension` — its CLI only parses a name filter, `--ignored`,
    // and `--show-output`; passing anything else is a usage error and the
    // whole run fails. Extension flags are intentionally not forwarded
    // here; note it in the output channel rather than silently dropping it
    // without explanation.
    const args = ['test'];
    const channel = getTestOutputChannel();
    if (config.extensions.length > 0) {
      channel.appendLine(
        `[stark.testOnSave] Note: configured extension(s) ${config.extensions.join(', ')} are not passed to 'stark test' (it does not accept --extension yet).`
      );
    }
    // Run from the file's own directory, not the workspace folder:
    // `find_package_root` walks upward from cwd looking for
    // `starkpkg.json`, so this finds the nearest enclosing package even in
    // a workspace containing several nested STARK packages (a workspace
    // folder is not necessarily a package root itself).
    const proc = spawn(config.packagePath, args, {
      cwd: path.dirname(document.fileName),
      shell: false,
    });
    let output = '';
    proc.stdout?.on('data', (d) => (output += d.toString()));
    proc.stderr?.on('data', (d) => (output += d.toString()));
    proc.on('error', (err) => {
      channel.appendLine(`[stark.testOnSave] Failed to start '${config.packagePath}': ${err.message}`);
      vscode.window.showErrorMessage(
        `STARK: could not run 'stark test' (${err.message}). Check stark.package.path.`
      );
    });
    proc.on('exit', (code) => {
      if (code !== 0) {
        vscode.window.showWarningMessage('STARK: tests failed on save. See "STARK Test" output.');
      }
      channel.appendLine(output);
    });
  }

  // Start the LSP client (hover, go-to-definition, references, formatting)
  await startLspClient();

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

  // Check (and optionally format/test) on save
  context.subscriptions.push(
    vscode.workspace.onDidSaveTextDocument((document) => {
      if (document.languageId !== 'stark') {
        return;
      }
      const config = getConfiguration();
      if (config.checkOnSave) {
        triggerDocumentCheck(document, true);
      }
      if (config.testOnSave) {
        runTestsForDocument(document);
      }
    })
  );

  // Format on save (runs before the save-triggered check/test above, via
  // VS Code's built-in willSaveTextDocument formatting hook)
  context.subscriptions.push(
    vscode.workspace.onWillSaveTextDocument((event) => {
      const config = getConfiguration();
      if (event.document.languageId === 'stark' && config.formatOnSave && vscode.workspace.isTrusted) {
        event.waitUntil(
          vscode.commands.executeCommand<vscode.TextEdit[]>(
            'vscode.executeFormatDocumentProvider',
            event.document.uri
          ).then((edits) => edits ?? [])
        );
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
      startLspClient();
      const editor = vscode.window.activeTextEditor;
      if (editor) {
        triggerDocumentCheck(editor.document);
      }
    })
  );

  // Restart the LSP client when tensor mode or the compiler path changes,
  // so the running session picks up the new extension set / binary.
  context.subscriptions.push(
    vscode.workspace.onDidChangeConfiguration((event) => {
      if (
        event.affectsConfiguration('stark.tensorExtensionEnabled') ||
        event.affectsConfiguration('stark.compiler.extensions') ||
        event.affectsConfiguration('stark.compiler.path') ||
        event.affectsConfiguration('stark.extensionEnabled')
      ) {
        commands.restartCompiler((doc) => triggerDocumentCheck(doc, true));
      }
    })
  );
}

let testOutputChannel: vscode.OutputChannel | undefined;
function getTestOutputChannel(): vscode.OutputChannel {
  if (!testOutputChannel) {
    testOutputChannel = vscode.window.createOutputChannel('STARK Test');
  }
  return testOutputChannel;
}

export async function deactivate() {
  if (checkTimeout) {
    clearTimeout(checkTimeout);
  }
  clearAllDiagnostics();
  await stopLspClient();
}
