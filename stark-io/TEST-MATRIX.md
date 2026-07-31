# stark-io v0.1 test matrix

Status values:

- `blocked`: requires source-level provider binding or provider surface not yet present.
- `pending`: pure package test can be added after syntax/API checking is enabled for this package.

| Area | Case | Status |
| --- | --- | --- |
| OpenOptions | all access modes disabled returns `InvalidInput` | pending |
| OpenOptions | truncate without write returns `InvalidInput` | pending |
| OpenOptions | create without write or append returns `InvalidInput` | pending |
| OpenOptions | create-new without write or append returns `InvalidInput` | pending |
| File construction | open existing file | blocked |
| File construction | open missing file maps to `NotFound` | blocked |
| File construction | create truncates existing file | blocked |
| Reading | one read reports accepted byte count | blocked |
| Reading | `read_to_end` enforces `max_bytes` | blocked |
| Reading | invalid UTF-8 maps to `InvalidData` | blocked |
| Writing | short write is observable | blocked |
| Writing | `write_all` retries until complete | blocked |
| Lifecycle | explicit close prevents implicit close | blocked |
| Lifecycle | failed open does not close | blocked |
| Lifecycle | moved file closes at final owner | blocked |
| Path | `join` rejects absolute child | pending |
| Directory | bounded listing enforces `max_entries` | blocked |
| Directory | recursive deletion enforces entry and depth bounds | blocked |

