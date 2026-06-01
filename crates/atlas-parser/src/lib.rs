#![allow(
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::missing_const_for_fn,
    clippy::match_same_arms,
    clippy::items_after_statements
)]
pub mod chunk;
pub mod parse;
pub mod parser;

pub use chunk::extract_chunks;
pub use parse::{parse_source, parser_for_language};
pub use parser::{ParserService, language_for};

#[cfg(test)]
mod tests;
