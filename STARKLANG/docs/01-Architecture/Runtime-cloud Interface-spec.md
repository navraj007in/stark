# STARKLANG Runtime-CloudInterface Specification

> Non-Normative Note: This document provides architectural guidance and may evolve. The normative Core v1 language specification lives in `docs/spec/`.

This document outlines how STARKLANG programs interact with cloud infrastructure during runtime execution. It defines the interface contract between STARKVM and the underlying cloud environment, particularly for serverless functions, containerized services, actor-based workloads, and ML pipelines.

---

## ☁️ Execution Context Overview
STARKVM is cloud-agnostic but cloud-aware. It provides a runtime adapter layer that supports standard execution modes while exposing an extensible interface for seamless cloud interaction.

---

## 🔧 Runtime Boot Process in Cloud Environments
Upon startup, STARKVM performs the following:

1. Loads environment variables from the container or host system.
2. Parses bytecode headers for `execution_mode` and `entrypoint` metadata.
3. Initializes runtime hooks (e.g., telemetry, logging, metrics).
4. Launches the appropriate runtime interface based on execution mode.

---

## 🚀 Serverless Runtime Interface

### Execution Model
- Input is streamed as JSON from `stdin`.
- Output is written to `stdout`.
- Compatible with AWS Lambda, Azure Functions, GCP Cloud Functions.

### Handler Definition
```stark
@serverless
fn handle(input: JSON) -> JSON:
    return { "msg": "Hello Cloud" }
```

### Runtime Adapter
```bash
cat input.json | starkvm handler build/main.sb > response.json
```
- Receives event JSON
- Parses it, routes to appropriate handler
- Emits structured response JSON

### Environment Binding
- ENV variables auto-injected:
  - `CLOUD_REGION`, `FUNC_NAME`, `FUNC_TIMEOUT`, `FUNC_MEMORY`

### Error Mapping
- Runtime errors mapped to standard HTTP status codes (500, 400, etc.)
- Logs captured to `/logs/runtime.log` by default or streamed to cloud logger

---

## 🌐 HTTP Server Mode Interface

### Execution Model
- STARKVM runs an embedded HTTP server if any `@server(path="/xyz")` annotations are detected.

### Endpoint Definition
```stark
@server(path="/predict")
fn predict(input: JSON) -> JSON:
    return { "result": model.predict(input["features"]) }
```

### Routing Table
Generated automatically based on annotations.
```
Route Table:
  POST /predict → predict()
  GET /health → builtin health check
```

### Middleware Support (Future)
- Request headers
- Auth tokens
- Rate limiting
- Request context

---

## 🧠 Actor Runtime Interface

### Execution Model
- STARKVM starts actor scheduler engine if `actor` is found.
- Actors communicate via in-process channels or distributed queue backend.

### Actor Example
```stark
actor EmailWorker:
    fn start():
        listen_queue("email_jobs")
```

### Actor Context
- Injected runtime context:
  - `actor_id`, `instance_id`, `message_queue`, `spawn_time`
- Actor messages encoded in binary or JSON, dispatched via internal queue engine

### Cloud Integration
- Optional backend binding: RabbitMQ, Azure Service Bus, Kafka, Redis
- Future support for autoscaling based on actor throughput metrics

---

## 📡 Cloud Telemetry & Observability Interface

### Runtime Telemetry Hooks
- Emitted automatically from STARKVM:
  - `runtime_start`
  - `request_received`
  - `function_invoked`
  - `error_occurred`
  - `actor_message_processed`
  - `runtime_shutdown`

### Export Formats
- OpenTelemetry JSON
- Prometheus-compatible `/metrics` endpoint (in server mode)
- Log shipping to stdout, file, or cloud log sinks (e.g., Azure Monitor, CloudWatch)

---

## 🔐 Runtime Secrets & Config Injection

### Supported Sources
- ENV variables
- Mounted volume secrets
- `config.stark` file overrides

### Secure Access API (Future DSL Support)
```stark
let secret = get_secret("DB_PASSWORD")
```

---

## 🗃 Deployment Metadata Usage

When built with `@serverless` or `@server`, deployment descriptors are auto-generated:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: stark-service
spec:
  containers:
    - name: starkvm
      image: starklang/runtime:1.0
      env:
        - name: FUNC_TIMEOUT
          value: "30"
        - name: FUNC_MEMORY
          value: "2Gi"
```

---

## 🔄 Future Roadmap
- Native runtime interface for edge environments (Cloudflare Workers, Akash, Deno Deploy)
- Secrets manager SDK (Azure Key Vault, AWS Secrets Manager)
- Full tracing propagation with OpenTelemetry IDs
- Plugin support for runtime adapters

---

## ✅ Summary
STARKLANG’s runtime-cloud interface provides a powerful yet decoupled mechanism for executing compiled code in any cloud environment. With execution mode-aware adapters, standardized telemetry, actor scheduling, and serverless support, it allows seamless packaging and scaling across modern cloud-native ecosystems.
