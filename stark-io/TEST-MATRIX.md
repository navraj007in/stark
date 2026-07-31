# stark-io v0.1 test matrix

Status values:

- `blocked`: requires exact public `File` migration or provider surface not yet present.
- `pending`: pure package test can be added after syntax/API checking is enabled for this package.
- `covered`: exercised by `starkc/tests/c788_starkc_build.rs`.

| Area | Case | Status |
| --- | --- | --- |
| OpenOptions | all access modes disabled returns `InvalidInput` | pending |
| OpenOptions | truncate without write returns `InvalidInput` | pending |
| OpenOptions | create without write or append returns `InvalidInput` | pending |
| OpenOptions | create-new without write or append returns `InvalidInput` | pending |
| File construction | open existing file via `open_file` | covered |
| File construction | open missing file maps to `NotFound` | covered |
| File construction | create-new existing file failure | pending |
| Reading | one read reports accepted byte count | covered |
| Reading | `read_to_end` enforces `max_bytes` | pending |
| Reading | invalid UTF-8 maps to `InvalidData` | pending |
| Writing | short write is observable | pending |
| Writing | `write_all` retries until complete | covered |
| Lifecycle | explicit close through `file_close` | covered |
| Lifecycle | failed open does not close | blocked |
| Lifecycle | moved file closes at final owner | blocked |
| Path | `join` rejects absolute child | pending |
| Directory | bounded listing enforces `max_entries` | blocked |
| Directory | recursive deletion enforces entry and depth bounds | blocked |
