import * as vscode from 'vscode';
import * as fs from 'fs';
import { CompilerDiagnostic } from './protocol';

export const starkDiagnosticCollection = vscode.languages.createDiagnosticCollection('stark');

export function byteOffsetToRange(text: string, startByte: number, endByte: number): vscode.Range {
  const buf = Buffer.from(text, 'utf8');
  
  const startPrefix = buf.slice(0, startByte).toString('utf8');
  const startLines = startPrefix.split('\n');
  const startLine = startLines.length - 1;
  const startChar = startLines[startLines.length - 1].length;
  
  const endPrefix = buf.slice(0, endByte).toString('utf8');
  const endLines = endPrefix.split('\n');
  const endLine = endLines.length - 1;
  const endChar = endLines[endLines.length - 1].length;
  
  return new vscode.Range(
    new vscode.Position(startLine, startChar),
    new vscode.Position(endLine, endChar)
  );
}

export function updateDiagnostics(
  document: vscode.TextDocument,
  compilerDiagnostics: CompilerDiagnostic[]
) {
  starkDiagnosticCollection.delete(document.uri);
  
  const documentText = document.getText();
  const filename = document.fileName;
  const diagnostics: vscode.Diagnostic[] = [];

  for (const diag of compilerDiagnostics) {
    // Only display diagnostics for the active document or related files
    // If the diagnostic is in a different file, we still register it to its own file URI
    const targetFile = diag.file;

    let targetText = '';
    if (targetFile === filename) {
      targetText = documentText;
    } else {
      try {
        targetText = fs.readFileSync(targetFile, 'utf8');
      } catch {
        targetText = '';
      }
    }

    const range = targetText
      ? byteOffsetToRange(targetText, diag.range.startByte, diag.range.endByte)
      : new vscode.Range(0, 0, 0, 0);

    let message = diag.message;
    if (diag.notes && diag.notes.length > 0) {
      message += '\n\nNote: ' + diag.notes.join('\nNote: ');
    }
    if (diag.help) {
      message += '\n\nHelp: ' + diag.help;
    }

    const severity =
      diag.severity === 'error'
        ? vscode.DiagnosticSeverity.Error
        : vscode.DiagnosticSeverity.Warning;

    const diagnostic = new vscode.Diagnostic(range, message, severity);
    if (diag.code) {
      diagnostic.code = diag.code;
    }
    diagnostic.source = 'starkc';

    // Map labels to relatedInformation
    if (diag.labels && diag.labels.length > 0) {
      const related: vscode.DiagnosticRelatedInformation[] = [];
      for (const label of diag.labels) {
        let labelText = '';
        if (label.file === filename) {
          labelText = documentText;
        } else {
          try {
            labelText = fs.readFileSync(label.file, 'utf8');
          } catch {
            labelText = '';
          }
        }

        const labelRange = labelText
          ? byteOffsetToRange(labelText, label.range.startByte, label.range.endByte)
          : new vscode.Range(0, 0, 0, 0);

        related.push(
          new vscode.DiagnosticRelatedInformation(
            new vscode.Location(vscode.Uri.file(label.file), labelRange),
            label.message
          )
        );
      }
      diagnostic.relatedInformation = related;
    }

    diagnostics.push(diagnostic);
  }

  // To be safe and simple, let's group by file and update the collection:
  const grouped = new Map<string, { uri: vscode.Uri; diags: vscode.Diagnostic[] }>();
  
  // Initialize current document in the map so we always clear it if no errors
  grouped.set(document.uri.toString(), { uri: document.uri, diags: [] });

  for (let i = 0; i < diagnostics.length; i++) {
    const diag = diagnostics[i];
    const compilerDiag = compilerDiagnostics[i];
    const uri = vscode.Uri.file(compilerDiag.file);
    const key = uri.toString();
    
    let group = grouped.get(key);
    if (!group) {
      group = { uri, diags: [] };
      grouped.set(key, group);
    }
    group.diags.push(diag);
  }

  for (const [, entry] of grouped) {
    starkDiagnosticCollection.set(entry.uri, entry.diags);
  }
}

export function clearDiagnostics(document: vscode.TextDocument) {
  starkDiagnosticCollection.delete(document.uri);
}

export function clearAllDiagnostics() {
  starkDiagnosticCollection.clear();
}
