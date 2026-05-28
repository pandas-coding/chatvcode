use atlas_core::{
    AtlasResult, ChunkKind, ChunkSpan, CodeChunk, ParseResult, SourceFile, index_path,
};
use clap::Parser;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "index_dir", about = "Index a directory or file with code-atlas")]
struct Args {
    #[arg(default_value = ".")]
    path: PathBuf,
}

fn example_parse_source(source_file: SourceFile) -> AtlasResult<ParseResult> {
    if source_file.source_text.trim().is_empty() {
        return Ok(ParseResult::success(source_file, Vec::new()));
    }

    let end_line = source_file.source_text.lines().count().saturating_sub(1);
    let symbol_name = source_file
        .path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .map(str::to_owned);
    let chunk = CodeChunk {
        id: CodeChunk::generate_id(&source_file.path, ChunkKind::Module, symbol_name.as_deref(), 0),
        file_path: source_file.path.clone(),
        language: source_file.language,
        kind: ChunkKind::Module,
        symbol_name,
        span: ChunkSpan::new(0, source_file.source_text.len(), 0, end_line),
        source_text: source_file.source_text.clone(),
    };

    Ok(ParseResult::success(source_file, vec![chunk]))
}

fn main() {
    let cli = Args::parse();

    // `atlas-core` stays parser-agnostic on purpose, so the example injects
    // a tiny parser instead of depending on `atlas-parser` and creating a cycle.
    match index_path(&cli.path, &example_parse_source) {
        Ok(result) => {
            println!("Index result for: {}", cli.path.display());
            println!(
                "  files: {} parsed, {} skipped, {} total",
                result.stats.parsed_files, result.stats.skipped_files, result.stats.total_files,
            );
            println!("  chunks: {}", result.stats.total_chunks);
            println!("  errors: {}", result.stats.total_errors);

            for file_result in &result.files {
                println!();
                println!(
                    "  ─── {} ({}) ───",
                    file_result.file.path.display(),
                    file_result.file.language,
                );
                for chunk in &file_result.chunks {
                    println!(
                        "    {} `{}` lines {}-{}",
                        chunk.kind,
                        chunk.symbol_name.as_deref().unwrap_or("(anonymous)"),
                        chunk.span.start_line + 1,
                        chunk.span.end_line + 1,
                    );
                }
            }

            if !result.errors.is_empty() {
                println!();
                println!("  errors:");
                for e in &result.errors {
                    println!("    - {e}");
                }
            }
        }
        Err(e) => eprintln!("error: {e}"),
    }
}
