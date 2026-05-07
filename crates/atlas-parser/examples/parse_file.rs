// Run:
//   cargo run --example parse_file -- <file1> [file2] ...
//   cargo run --example parse_file -- crates/atlas-parser/src/lib.rs
//   cargo run --example parse_file -- --help

use atlas_core::SourceFile;
use atlas_parser::parse_source;
use clap::Parser;
use std::{fs, path::PathBuf};

#[derive(Parser)]
#[command(name = "parse_file", about = "Parse source files with atlas-parser")]
struct Args {
    #[arg(required = true)]
    files: Vec<PathBuf>,
}

fn main() {
    let cli = Args::parse();

    for path in &cli.files {
        let source_text = match fs::read_to_string(path) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("error reading {}: {}", path.display(), e);
                continue;
            }
        };

        let file = SourceFile::new(path, &source_text);
        println!(
            "─── {} ({}, {} bytes, {} lines) ───",
            file.path.display(),
            file.language,
            source_text.len(),
            source_text.lines().count(),
        );

        match parse_source(file) {
            Ok(result) => {
                if result.errors.is_empty() {
                    println!("  ✓ no parse errors");
                } else {
                    for e in &result.errors {
                        println!("  ✗ {}", e);
                    }
                }
                println!("  chunks: {}", result.chunks.len());
                for chunk in &result.chunks {
                    println!(
                        "    {} `{}` at lines {}-{}",
                        chunk.kind,
                        chunk.symbol_name.as_deref().unwrap_or("(anonymous)"),
                        chunk.span.start_line + 1,
                        chunk.span.end_line + 1,
                    );
                }
            }
            Err(e) => {
                eprintln!("  ✗ {}", e);
            }
        }
        println!();
    }
}
