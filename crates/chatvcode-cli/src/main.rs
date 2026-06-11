//! Binary entry point for the `chatvcode` CLI tool.
//!
//! Provides commands:
//! - `chatvcode index <path>` — Index source files
//! - `chatvcode search <query>` — Semantic search over indexed code
//! - `chatvcode chat "<question>"` — RAG-enhanced AI chat about a codebase

use std::process;

use chatvcode_cli::run;
use chatvcode_core::ErrorSeverity;

fn main() {
    if let Err(e) = run() {
        match e.severity {
            ErrorSeverity::Unrecoverable => {
                log::error!("Fatal error: {e}");
                eprintln!("Error: {e}");
            }
            ErrorSeverity::Recoverable => {
                log::warn!("Error: {e}");
                eprintln!("Error: {e}");
            }
        }
        process::exit(1);
    }
}
