import * as vscode from 'vscode';
import {
  LanguageClient,
  LanguageClientOptions,
  ServerOptions,
  Trace,
  TransportKind,
} from 'vscode-languageclient/node';
import { getConfiguration } from './configuration';

function traceFromLogLevel(level: string): Trace {
  switch (level) {
    case 'off':
      return Trace.Off;
    case 'error':
    case 'info':
      return Trace.Messages;
    case 'verbose':
      return Trace.Verbose;
    default:
      return Trace.Messages;
  }
}

let client: LanguageClient | undefined;

export type LspStatus = 'stopped' | 'starting' | 'running' | 'error';

const statusListeners: ((status: LspStatus) => void)[] = [];

export function onLspStatusChange(listener: (status: LspStatus) => void): void {
  statusListeners.push(listener);
}

function setStatus(status: LspStatus): void {
  for (const listener of statusListeners) {
    listener(status);
  }
}

/** Starts the `starkc lsp` client, or does nothing if already running. */
export async function startLspClient(): Promise<void> {
  if (client) {
    return;
  }
  if (!vscode.workspace.isTrusted) {
    return;
  }

  const config = getConfiguration();
  if (!config.extensionEnabled) {
    return;
  }

  const args = ['lsp'];
  const serverOptions: ServerOptions = {
    run: { command: config.compilerPath, args, transport: TransportKind.stdio },
    debug: { command: config.compilerPath, args, transport: TransportKind.stdio },
  };

  const clientOptions: LanguageClientOptions = {
    documentSelector: [{ scheme: 'file', language: 'stark' }],
    outputChannelName: 'STARK Language Server',
    initializationOptions: {
      extensions: config.extensions,
    },
    initializationFailedHandler: (error: unknown) => {
      console.error('STARK LSP failed to initialize:', error);
      setStatus('error');
      return false;
    },
  };

  client = new LanguageClient('starkLanguageServer', 'STARK Language Server', serverOptions, clientOptions);

  setStatus('starting');
  try {
    await client.start();
    await client.setTrace(traceFromLogLevel(config.lspLogLevel));
    setStatus('running');
  } catch (err) {
    console.error('Failed to start STARK language server:', err);
    setStatus('error');
    client = undefined;
  }
}

export async function stopLspClient(): Promise<void> {
  if (!client) {
    return;
  }
  const toStop = client;
  client = undefined;
  setStatus('stopped');
  try {
    await toStop.stop();
  } catch (err) {
    console.error('Error stopping STARK language server:', err);
  }
}

export async function restartLspClient(): Promise<void> {
  await stopLspClient();
  await startLspClient();
}

export function getLspClient(): LanguageClient | undefined {
  return client;
}

export function showLspOutput(): void {
  client?.outputChannel.show();
}
