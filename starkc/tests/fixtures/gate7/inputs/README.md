# Gate 7 inputs

The Gate 7 inputs are **fetched by checksum, not committed** (same policy as
Gate 5 — no large binaries in the tree). `examples/gate7/fetch-artifacts.py`
downloads them into the gitignored `starkc/tmp/gate7/` directory and verifies
each SHA-256.

| Input | Role | SHA-256 |
| --- | --- | --- |
| `dog.jpg` | real detection reference image (PyTorch Hub sample, BSD-3-Clause repo) | `f3f87bb8ab3c26c7ecfd3ac60421d7f32b0503d1d6c5baf8bac42ed93d86351a` |
| `test_data_set_0/input_0.pb` | ONNX-bundled conformance input tensor | `7df6dfefe116d2024a2678a3b015b98256ab8e15205b6b67e5291c8e7889cb5c` |
| `test_data_set_0/output_0.pb` | ONNX-bundled expected output (model-correctness oracle) | `59bd363150b98190fe3957e98905a645feec39b16c7f543e8e96d9e8e595c727` |

To materialise them:

```bash
python ../../../examples/gate7/fetch-artifacts.py
```
