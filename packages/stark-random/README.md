# stark-random

`stark-random` provides secure OS-backed randomness plus deterministic seeded PRNG helpers.

The secure API is backed by the `random` host capability and never falls back to deterministic
state. The deterministic API is pure package code and is reproducible for tests and simulations,
but is not cryptographically secure.

