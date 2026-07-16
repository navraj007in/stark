import { spawn, ChildProcess } from 'child_process';
import { CompilerOutput } from './protocol';
import { getConfiguration } from './configuration';

const activeProcesses = new Map<string, ChildProcess>();

export interface CompilerRunResult {
  success: boolean;
  output?: CompilerOutput;
  error?: string;
}

export function runCompiler(
  documentText: string,
  filename: string,
  timeoutMs = 5000
): Promise<CompilerRunResult> {
  return new Promise((resolve) => {
    const config = getConfiguration();
    const args: string[] = [
      'check',
      '--message-format',
      'json',
    ];

    for (const ext of config.extensions) {
      args.push('--extension', ext);
    }

    args.push('--stdin', '--filename', filename);

    // Cancel any active compiler run for the same file
    const oldProcess = activeProcesses.get(filename);
    if (oldProcess) {
      oldProcess.kill();
      activeProcesses.delete(filename);
    }

    let resolved = false;
    const process = spawn(config.compilerPath, args, {
      shell: false,
    });
    activeProcesses.set(filename, process);

    let stdout = '';
    let stderr = '';

    const timer = setTimeout(() => {
      if (!resolved) {
        resolved = true;
        if (activeProcesses.get(filename) === process) {
          activeProcesses.delete(filename);
        }
        process.kill();
        resolve({
          success: false,
          error: `Compiler timed out after ${timeoutMs}ms`,
        });
      }
    }, timeoutMs);

    process.stdout.on('data', (data) => {
      stdout += data.toString();
    });

    process.stderr.on('data', (data) => {
      stderr += data.toString();
    });

    process.on('error', (err) => {
      if (!resolved) {
        resolved = true;
        clearTimeout(timer);
        if (activeProcesses.get(filename) === process) {
          activeProcesses.delete(filename);
          resolve({
            success: false,
            error: `Failed to spawn compiler: ${err.message}`,
          });
        } else {
          resolve({
            success: false,
            error: 'superseded',
          });
        }
      }
    });

    process.on('exit', (code, signal) => {
      if (!resolved) {
        resolved = true;
        clearTimeout(timer);

        if (activeProcesses.get(filename) === process) {
          activeProcesses.delete(filename);
        } else {
          resolve({
            success: false,
            error: 'superseded',
          });
          return;
        }

        if (code === null) {
          resolve({
            success: false,
            error: `Compiler terminated by signal: ${signal}`,
          });
          return;
        }

        // Code 0 = no errors, Code 1 = source diagnostics. Both are successfully completed checks!
        if (code === 0 || code === 1) {
          try {
            const output: CompilerOutput = JSON.parse(stdout);
            if (!output || typeof output !== 'object') {
              throw new Error('Output is not an object');
            }
            if (output.schemaVersion !== 1) {
              throw new Error(`Unsupported schema version: ${output.schemaVersion}`);
            }
            if (output.tool !== 'starkc') {
              throw new Error(`Invalid tool name: ${output.tool}`);
            }
            if (!Array.isArray(output.diagnostics)) {
              throw new Error('Diagnostics field is not an array');
            }
            for (const diag of output.diagnostics) {
              if (
                typeof diag.severity !== 'string' ||
                (diag.severity !== 'error' && diag.severity !== 'warning')
              ) {
                throw new Error(`Invalid diagnostic severity: ${diag.severity}`);
              }
              if (typeof diag.message !== 'string') {
                throw new Error('Diagnostic message is missing or invalid');
              }
              if (typeof diag.file !== 'string') {
                throw new Error('Diagnostic file is missing or invalid');
              }
              if (
                !diag.range ||
                typeof diag.range.startByte !== 'number' ||
                typeof diag.range.endByte !== 'number' ||
                diag.range.startByte < 0 ||
                diag.range.endByte < diag.range.startByte
              ) {
                throw new Error(`Invalid diagnostic range: ${JSON.stringify(diag.range)}`);
              }
            }
            resolve({
              success: true,
              output,
            });
          } catch (err) {
            resolve({
              success: false,
              error: `Failed to parse or validate compiler output: ${(err as Error).message}\nStdout: ${stdout}\nStderr: ${stderr}`,
            });
          }
        } else {
          resolve({
            success: false,
            error: `Compiler exited with error code ${code}\nStderr: ${stderr}`,
          });
        }
      }
    });

    try {
      process.stdin.write(documentText);
      process.stdin.end();
    } catch (err) {
      if (!resolved) {
        resolved = true;
        clearTimeout(timer);
        if (activeProcesses.get(filename) === process) {
          activeProcesses.delete(filename);
        }
        process.kill();
        resolve({
          success: false,
          error: `Failed to write to compiler stdin: ${(err as Error).message}`,
        });
      }
    }
  });
}
