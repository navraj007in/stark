# stark-io v0.1 test matrix

Status values:

- `covered`: exercised by `starkc/tests/c788_starkc_build.rs` — either
  `io_minimal_executes_from_source_through_stark_io_package`, or
  `io_expanded_surface_executes_from_source_through_stark_io_package`, which covers seek, sync,
  set-length, metadata, path existence, joining, rename, copy and directory create/list/remove.
- `pending`: a pure package test, writable **now**. This previously said "after syntax/API checking
  is enabled for this package", which was wrong: `stark test` already runs a library package's unit
  tests and needs no `main`. What it needs is the convention — a `tests.stark` module declared from
  `lib.stark`, with functions named `test_*`, taking no parameters and no receiver. STARK has no
  `#[test]` attribute; `#` does not lex. `stark-io` has no `tests.stark` at all yet, which is the
  only reason these are pending.
- `blocked`: provider surface not present. **No row carries this status any more** — CD-292 supplied
  every native symbol the matrix was waiting on. Kept so a future row that needs it has a meaning.
- `out of scope`: deliberately not provided, with the reason on the row.

**No case is blocked on Core `File`'s migration any more.** `NativeFile` binds `io_file`, its own
resource type, so the lifecycle is real: owned, moved, and closed exactly once from a `Drop`
terminator. The two lifecycle rows below were `blocked` for that reason and are now ordinary
`pending` work. What remains genuinely blocked is blocked on missing native symbols, nothing else.

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
| Lifecycle | explicit close through `file_close` (by-value; drop emits the close) | covered |
| Lifecycle | failed open does not close | pending |
| Lifecycle | moved file closes at final owner | pending |
| Path | `join` rejects absolute child | pending |
| Directory | bounded listing enforces `max_entries` | covered |
| Directory | recursive deletion enforces entry and depth bounds | out of scope — no recursive delete is provided |

## Added by the expanded surface

| Area | Case | Status |
| --- | --- | --- |
| OpenOptions | write mode permits `set_length` where a read-only handle cannot | covered |
| Seek | `Start` reports the absolute position | covered |
| Seek | `End(0)` reports the file length | covered |
| Seek | `Start` above `Int64::MAX` is `InvalidInput`, not a trap | pending |
| Metadata | length and `FileType::File` by path | covered |
| Metadata | length through an open handle | covered |
| Metadata | a symlink reports `Symlink`, not its target's kind | pending |
| Metadata | an unavailable timestamp is `None`, not a sentinel | pending |
| Truncate | `set_length` shortens and the change is observable by path | covered |
| Durability | `file_sync` succeeds on a writable handle | covered |
| Path | `path_join` refuses an absolute child | covered |
| Path | `path_join` inserts a separator only when one is missing | pending |
| Effects | rename removes the source and `path_exists` observes it | covered |
| Effects | copy reports the byte count | covered |
| Effects | `remove_file` and `remove_dir` are observable in a later listing | covered |
| Directory | `create_dir` is not recursive | pending |
| Directory | a listing that overruns the buffer is `LimitExceeded`, never short | pending |
| Directory | entry `file_type` distinguishes files from directories | covered |
