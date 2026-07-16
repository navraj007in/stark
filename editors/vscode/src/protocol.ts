export interface DiagnosticRange {
  startByte: number;
  endByte: number;
}

export interface DiagnosticLabel {
  message: string;
  file: string;
  range: DiagnosticRange;
}

export interface CompilerDiagnostic {
  severity: 'error' | 'warning';
  code: string | null;
  message: string;
  file: string;
  range: DiagnosticRange;
  labels: DiagnosticLabel[];
  notes: string[];
  help: string | null;
}

export interface CompilerOutput {
  schemaVersion: number;
  tool: string;
  toolVersion: string;
  diagnostics: CompilerDiagnostic[];
}
