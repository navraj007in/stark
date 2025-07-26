
# 🎭 STARKLANG — Actor System Specification

The STARKLANG Actor System is a core concurrency abstraction that enables scalable, message-driven, and fault-tolerant applications. It promotes state encapsulation, safe parallelism, and distributed system patterns, aligning perfectly with STARK's goals for cloud-native, AI-first programming.

---

## 📌 Why Actors?

Traditional shared-memory concurrency (threads, mutexes) often leads to complexity, data races, and fragile systems.

The Actor Model offers a cleaner alternative:
- Each actor has private state
- All communication is via messages
- Concurrency is achieved through isolation

---

## 🏗️ Actor Model Architecture

| Component         | Description                                      |
|------------------|--------------------------------------------------|
| Actor             | A self-contained unit with state & mailbox       |
| ActorRef<T>       | Reference to send messages to an actor           |
| Mailbox           | Queue where incoming messages are stored         |
| ActorRegistry     | Global system that tracks active actors          |
| Dispatcher        | Schedules message delivery & actor processing    |
| Supervision Tree  | Manages actor hierarchies & fault recovery (planned) |

---

## 🔹 Declaring Actors

```stark
actor Counter:
    mut state: Int = 0

    fn on_receive(msg: Int):
        state += msg
        print("Counter = " + str(state))
```

---

## ✉ Sending Messages to Actors

```stark
let counter = spawn_actor(Counter)
counter.send(1)
counter.send(10)
```

---

## 🔁 Receiving Messages

```stark
fn on_receive(msg: Command):
    match msg:
        Increment(val) => state += val
        Reset() => state = 0
```

---

## 🔀 ActorRef API

| Method             | Description                                 |
|--------------------|---------------------------------------------|
| send(msg: T)        | Sends a fire-and-forget message             |
| ask(msg) -> Future<R> | Sends a message and expects a response    |
| terminate()         | Gracefully stops an actor                   |
| broadcast(msg)      | Sends message to actor group (planned)     |

---

## 🔄 Actor Lifecycle

| Stage      | Description                         |
|------------|-------------------------------------|
| Spawned    | Actor instance initialized          |
| Active     | Receiving & processing messages     |
| Paused     | (Planned) Actor can suspend itself  |
| Terminated | Removed from registry, GC’d         |

---

## 💥 Actor Failure Handling

| Failure Type     | Handling Behavior                          |
|------------------|--------------------------------------------|
| Panic in handler | Actor terminates (default)                 |
| Supervision Tree | Restart strategy (planned)                 |
| Manual Retry     | Message can be re-sent or logged externally|

---

## 🧠 Ask Pattern (Actor RPC)

```stark
let reply: Future<String> = actor.ask(GetStatus())
let status = await reply
```

---

## 🔐 Actor Isolation & Safety

- Actors cannot share memory
- Cannot access another actor's state directly
- Compiler ensures actor state isn’t leaked via references

---

## ⚙ Actor Scheduler

| Feature            | Behavior                                  |
|--------------------|-------------------------------------------|
| Work Stealing      | Actors balanced across threads/cores      |
| Priority Queues    | (Planned) Prioritize certain message types|
| Backpressure       | Auto-throttle when actor is overloaded    |

---

## 🧩 Advanced Features (Planned)

| Feature                | Description                                 |
|------------------------|---------------------------------------------|
| Actor Pools            | Pre-spawned group of actors for load-spikes |
| Actor Groups           | Broadcast messaging                        |
| Hot Swap Behavior      | Dynamic method replacement                 |
| Actor Mobility         | Relocate actor across nodes                |
| Supervision Trees      | Restart policies: One-for-One, All-for-One |

---

## 🛠 Best Practices

- Keep actors focused and small
- Use enums for typed messages
- Favor `ask()` only for required responses
- Design stateless actors for horizontal scaling

---

## 📌 Example: AI-Pipeline with Actor Workers

```stark
enum Task:
    Inference(data: Tensor<Float32>[128])
    Shutdown()

actor InferenceWorker:
    fn on_receive(msg: Task):
        match msg:
            Inference(data) => 
                let result = model.predict(data)
                print("Prediction: " + str(result))
            Shutdown() => terminate()

let worker = spawn_actor(InferenceWorker)

for data in dataset:
    worker.send(Inference(data))

worker.send(Shutdown())
```

---

## ✅ Summary: Actor System Features

| Capability                | Status   |
|--------------------------|----------|
| Actor Definition          | ✅ Stable |
| Message Passing           | ✅ Stable |
| Mailbox & Dispatcher      | ✅ Stable |
| ask() Pattern (Future)    | ✅ Stable |
| Supervision Tree          | ⏳ Planned |
| Actor Groups/Pools        | ⏳ Planned |
| Message Matching DSL      | ✅ Available |

---

STARKLANG’s actor model is a powerful tool for AI pipelines, cloud-native services, stream processors, and fault-tolerant systems — giving developers tools to build reactive, stateful, and distributed logic with precision and elegance.
