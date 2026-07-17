//! Language Server Protocol (LSP) implementation for STARK.
//!
//! Provides a stdio-based LSP server supporting Core v1 diagnostics,
//! type information, navigation, and other IDE features.

pub mod protocol;
pub mod server;
pub mod state;

pub use server::Server;
pub use state::ServerState;

use std::io::{self, BufReader};

/// Run the LSP server on stdin/stdout.
pub fn run() -> io::Result<()> {
    let mut server = Server::new();
    let stdin = io::stdin();
    let stdout = io::stdout();
    let reader = BufReader::new(stdin.lock());

    server.run(reader, stdout.lock())
}
