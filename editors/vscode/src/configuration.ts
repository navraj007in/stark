import * as vscode from 'vscode';

export interface StarkConfiguration {
  compilerPath: string;
  extensions: string[];
  checkOnSave: boolean;
  checkOnType: boolean;
  checkOnTypeDebounceMs: number;
}

export function getConfiguration(): StarkConfiguration {
  const config = vscode.workspace.getConfiguration('stark');
  
  return {
    compilerPath: config.get<string>('compiler.path', 'starkc'),
    extensions: config.get<string[]>('compiler.extensions', []),
    checkOnSave: config.get<boolean>('check.onSave', true),
    checkOnType: config.get<boolean>('check.onType', false),
    checkOnTypeDebounceMs: config.get<number>('check.onTypeDebounceMs', 500),
  };
}
