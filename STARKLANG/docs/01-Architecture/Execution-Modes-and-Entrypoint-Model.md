# STARKLANG Execution Modes and Entrypoint Model

This document defines the execution semantics of STARKLANG programs within the STARKVM runtime. It outlines the supported execution modes, how entrypoints are defined, and how the STARK compiler and VM interpret different program types.

---

## 🚀 Execution Modes Overview
STARKVM supports multiple execution modes to accommodate diverse application types:

### 1️⃣ **Standalone Application Mode**
- Default execution mode for CLI tools, scripts, and long-running services.
- Expects a `main()` function as the program’s entrypoint.

```stark
fn main():
    print("Hello STARK World")
```
- Compiled and executed via:
```bash
stark build
stark run build/main.sb
```

### 2️⃣ **Server Mode (Microservices)**
- Executes a program that exposes server endpoints via `@server` or `@http` annotations.
- Each annotated function becomes a route in the HTTP interface.

```stark
@server(path="/predict")
fn handle_request(input: JSON) -> JSON:
    let output = model.predict(input["features"])
    return { "result": output }
```
- Executed via:
```bash
stark serve
```
- Runs a lightweight HTTP server with request routing and JSON I/O.

### 3️⃣ **Serverless Handler Mode**
- Lightweight function execution via `@serverless` annotation.
- Handler is mapped to cloud-compatible runtime (stdin/stdout).

```stark
@serverless
fn handle(input: JSON) -> JSON:
    return { "response": "Handled by serverless" }
```
- Execution Entry:
  - Receives serialized JSON via stdin
  - Emits response via stdout

- Used in FaaS environments like:
  - Azure Functions
  - AWS Lambda
  - GCP Cloud Functions

### 4️⃣ **Actor Mode**
- Used for concurrency-driven programs using `actor` syntax.
- Each actor defines a lifecycle entrypoint via `start()`:

```stark
actor Worker:
    fn start():
        listen_messages()
```
- STARKVM spawns and dispatches actors using its scheduler engine.

### 5️⃣ **REPL/Script Mode**
- Quick evaluation for experiments or one-off executions.
- No explicit `main()` required.
- Executes top-level expressions in order.

```bash
stark repl
```
- Example:
```stark
let x = 5 * 10
print(x)
```

### 6️⃣ **Pipeline Mode (ML Pipelines)**
- Executes DSL-defined ML pipelines as long-running tasks or deployable services.

```stark
ml_pipeline MyPipeline:
    load data from "data.csv"
    preprocess with Normalize
    train using LinearRegression()
    deploy as service "predictor"
```
- Automatically maps to either `main()` or `@server` execution mode depending on context.

---

## 🔍 Entrypoint Resolution Rules
- If `main()` is defined → Standalone Application Mode
- If `@server` exists → Server Mode
- If `@serverless` exists → Handler Mode
- If `actor` exists → Actor Scheduler Mode
- If no entry but expressions exist → REPL/Script Mode
- If `ml_pipeline` exists → Pipeline Execution Mode

---

## 🔧 Execution Metadata in `starkpkg.json`
```json
{
  "entry": "src/main.stark",
  "execution_mode": "serverless"
}
```
- Compiler reads this to select appropriate execution target
- Optional overrides via CLI: `stark run --mode serverless`

---

## 📁 Runtime Dispatch Internals
- Bytecode header includes metadata for mode:
```text
.bytecode_header
 ├─ entry: main
 ├─ mode: server | serverless | actor
 └─ symbols: {...}
```
- STARKVM bootloader dispatches accordingly

---

## 🔄 Future Enhancements
- Support multiple entrypoints
- Pluggable execution modes
- Inline mode switch annotations
- Execution Profiles for testing and prod

---

## ✅ Summary
STARKLANG’s execution model offers a flexible and scalable way to target CLI tools, serverless functions, HTTP services, actor-based workloads, and ML pipelines — all from a unified language core and runtime engine.

