import * as vscode from 'vscode';

export interface StarkConfiguration {
  compilerPath: string;
  /** Path to the `stark` package-manager binary (check/build/run/test/fmt),
   * distinct from `compilerPath` (`starkc`, the single-file compiler). */
  packagePath: string;
  extensions: string[];
  checkOnSave: boolean;
  checkOnType: boolean;
  checkOnTypeDebounceMs: number;
  extensionEnabled: boolean;
  lspLogLevel: string;
  formatOnSave: boolean;
  tensorExtensionEnabled: boolean;
  testOnSave: boolean;
}

export function getConfiguration(): StarkConfiguration {
  const config = vscode.workspace.getConfiguration('stark');
  const tensorExtensionEnabled = config.get<boolean>('tensorExtensionEnabled', false);
  const configuredExtensions = config.get<string[]>('compiler.extensions', []);

  // `tensorExtensionEnabled` is the plan's convenience boolean toggle for
  // the common case; `compiler.extensions` remains the lower-level array
  // setting for anyone enabling other/future extensions. The two are
  // merged (deduplicated) so either can be used independently.
  const extensions = new Set(configuredExtensions);
  if (tensorExtensionEnabled) {
    extensions.add('tensor');
  }

  return {
    compilerPath: config.get<string>('compiler.path', 'starkc'),
    packagePath: config.get<string>('package.path', 'stark'),
    extensions: Array.from(extensions),
    checkOnSave: config.get<boolean>('check.onSave', true),
    checkOnType: config.get<boolean>('check.onType', false),
    checkOnTypeDebounceMs: config.get<number>('check.onTypeDebounceMs', 500),
    extensionEnabled: config.get<boolean>('extensionEnabled', true),
    lspLogLevel: config.get<string>('lspLogLevel', 'info'),
    formatOnSave: config.get<boolean>('formatOnSave', false),
    tensorExtensionEnabled,
    testOnSave: config.get<boolean>('testOnSave', false),
  };
}
