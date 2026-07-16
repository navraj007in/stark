import { spawn } from 'child_process';
import { CompilerOutput } from './protocol';
import { getConfiguration } from './configuration';

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

    let resolved = false;
    const process = spawn(config.compilerPath, args, {
      shell: false,
    });

    let stdout = '';
    let stderr = '';

    const timer = setTimeout(() => {
      if (!resolved) {
        resolved = true;
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
        resolve({
          success: false,
          error: `Failed to spawn compiler: ${err.message}`,
        });
      }
    });

    process.on('exit', (code, signal) => {
      if (!resolved) {
        resolved = true;
        clearTimeout(timer);

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
            resolve({
              success: true,
              output,
            });
          } catch (err) {
            resolve({
              success: false,
              error: `Failed to parse compiler output as JSON: ${(err as Error).message}\nStdout: ${stdout}\nStderr: ${stderr}`,
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
        process.kill();
        resolve({
          success: false,
          error: `Failed to write to compiler stdin: ${(err as Error).message}`,
        });
      }
    }
  });
}
