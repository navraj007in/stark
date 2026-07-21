//! §13.1: compact native-span-ID -> `FileId`/path/line/column/function-instance/MIR-location
//! table, so trap sites carry only a compact ID and the runtime resolves it. Empty until
//! WP-C5.2e wires the trap path -- C5.1b's proof program has no trap sites.
