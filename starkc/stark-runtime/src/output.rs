//! §9.1: stdout/stderr byte submission, with newline-appending variants.
//!
//! Byte-oriented (not `&str`) because MIR's `PrintlnStr`/`PrintStr` runtime ops (and later
//! integer/bool/float print ops) hand the backend already-formatted UTF-8 bytes; this module
//! does no formatting of its own.

use std::io::Write;

pub fn stdout_bytes(bytes: &[u8]) {
    let mut out = std::io::stdout().lock();
    let _ = out.write_all(bytes);
}

/// Flush any buffered stdout bytes. `std::io::stdout()` is a `LineWriter`, so bytes submitted via
/// `print` (no trailing newline) sit unflushed. `std::process::exit` does NOT flush it, so a trap
/// that aborts mid-output would DROP that prefix — diverging from the HIR/MIR interpreters, which
/// retain their captured prefix. The trap ABI (`trap::abort*`) calls this before exiting so the
/// observable pre-trap output is byte-identical across engines (CD-120 Contract B).
pub fn flush_stdout() {
    let _ = std::io::stdout().lock().flush();
}

pub fn stdout_line(bytes: &[u8]) {
    let mut out = std::io::stdout().lock();
    let _ = out.write_all(bytes);
    let _ = out.write_all(b"\n");
}

pub fn stderr_bytes(bytes: &[u8]) {
    let mut err = std::io::stderr().lock();
    let _ = err.write_all(bytes);
}

/// Flush any buffered stderr bytes (WP-C7.9 Packet D).
///
/// The stderr counterpart of [`flush_stdout`], and needed for the same reason plus one more: the
/// trap ABI writes its own diagnostic to this stream, so a program's unterminated `eprint` prefix
/// must reach the stream *before* that diagnostic or the two would interleave in the wrong order.
pub fn flush_stderr() {
    let _ = std::io::stderr().lock().flush();
}

pub fn stderr_line(bytes: &[u8]) {
    let mut err = std::io::stderr().lock();
    let _ = err.write_all(bytes);
    let _ = err.write_all(b"\n");
}
