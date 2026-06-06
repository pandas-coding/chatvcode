//! Mock parser for integration tests.
//!
//! Provides a `mock_parse_source` with the same signature as `chatvcode_parser::parse_source`
//! but implemented with simple pattern matching instead of tree-sitter.
//! This eliminates the cyclic dev-dependency between chatvcode-core and chatvcode-parser.

use chatvcode_core::{
    ChatVCodeResult, ChunkKind, ChunkSpan, CodeChunk, FileLanguage, ParseResult, SourceFile,
};

/// A mock `parse_source` that extracts chunks via pattern matching.
///
/// Supports the same languages and chunk types as the real parser,
/// but with simplified heuristics suitable for test fixtures.
pub fn mock_parse_source(source_file: SourceFile) -> ChatVCodeResult<ParseResult> {
    if !source_file.language.is_supported() {
        let err = chatvcode_core::ChatVCodeError::unsupported_language(
            "source file language is not supported yet",
        );
        return Err(err);
    }

    let chunks = extract_chunks(&source_file);
    Ok(ParseResult::success(source_file, chunks))
}

fn extract_chunks(source: &SourceFile) -> Vec<CodeChunk> {
    match source.language {
        FileLanguage::Rust => extract_rust_chunks(source),
        FileLanguage::JavaScript | FileLanguage::Jsx => extract_js_chunks(source),
        FileLanguage::TypeScript | FileLanguage::Tsx => extract_ts_chunks(source),
        FileLanguage::Python => extract_python_chunks(source),
        FileLanguage::Php => extract_php_chunks(source),
        FileLanguage::Unknown => vec![],
    }
}

fn extract_rust_chunks(source: &SourceFile) -> Vec<CodeChunk> {
    let mut chunks = Vec::new();
    let lines: Vec<&str> = source.source_text.lines().collect();
    let line_offsets = line_start_offsets(&source.source_text);

    for (line_idx, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with("//") || trimmed.starts_with("use ") {
            continue;
        }

        let (kind, name) = if let Some(name) = match_rust_kind(trimmed) {
            name
        } else {
            continue;
        };

        let start_byte = line_offsets.get(line_idx).copied().unwrap_or(0);
        let end_line = find_rust_chunk_end_line(&lines, line_idx);
        let end_byte = line_end_byte(&line_offsets, &lines, end_line, source.source_text.len());
        let source_text = source.source_text[start_byte..end_byte].to_string();

        chunks.push(CodeChunk {
            id: CodeChunk::generate_id(&source.path, kind, name.as_deref(), line_idx),
            file_path: source.path.clone(),
            language: source.language,
            kind,
            symbol_name: name,
            span: ChunkSpan::new(start_byte, end_byte, line_idx, end_line),
            source_text,
        });
    }

    chunks
}

fn line_start_offsets(text: &str) -> Vec<usize> {
    let mut offsets = vec![0];
    for (idx, ch) in text.char_indices() {
        if ch == '\n' {
            offsets.push(idx + 1);
        }
    }
    offsets
}

fn line_end_byte(
    line_offsets: &[usize],
    lines: &[&str],
    line_idx: usize,
    fallback: usize,
) -> usize {
    line_offsets
        .get(line_idx)
        .map_or(fallback, |start| start + lines.get(line_idx).map_or(0, |line| line.len()))
}

fn find_rust_chunk_end_line(lines: &[&str], start_line: usize) -> usize {
    let mut brace_depth = 0usize;
    let mut saw_open_brace = false;

    for (line_idx, line) in lines.iter().enumerate().skip(start_line) {
        for ch in line.chars() {
            match ch {
                '{' => {
                    brace_depth += 1;
                    saw_open_brace = true;
                }
                '}' if saw_open_brace => {
                    brace_depth = brace_depth.saturating_sub(1);
                    if brace_depth == 0 {
                        return line_idx;
                    }
                }
                _ => {}
            }
        }
    }

    start_line
}

fn match_rust_kind(line: &str) -> Option<(ChunkKind, Option<String>)> {
    if let Some(rest) = line.strip_prefix("fn ") {
        let name = rest.split(['(', '<']).next()?.trim().to_string();
        if !name.is_empty() {
            return Some((ChunkKind::Function, Some(name)));
        }
    }
    if let Some(rest) = line.strip_prefix("struct ") {
        let name = rest.split(['{', '<', '(', ';']).next()?.trim().to_string();
        if !name.is_empty() {
            return Some((ChunkKind::Struct, Some(name)));
        }
    }
    if let Some(rest) = line.strip_prefix("enum ") {
        let name = rest.split(['{', '<']).next()?.trim().to_string();
        if !name.is_empty() {
            return Some((ChunkKind::Enum, Some(name)));
        }
    }
    if let Some(rest) = line.strip_prefix("trait ") {
        let name = rest.split(['{', '<']).next()?.trim().to_string();
        if !name.is_empty() {
            return Some((ChunkKind::Trait, Some(name)));
        }
    }
    if let Some(rest) = line.strip_prefix("const ") {
        let name = rest.split([':', '=']).next()?.trim().to_string();
        if !name.is_empty() && !name.contains(' ') {
            return Some((ChunkKind::Constant, Some(name)));
        }
    }
    if let Some(rest) = line.strip_prefix("type ") {
        let name = rest.split(['<', '=']).next()?.trim().to_string();
        if !name.is_empty() {
            return Some((ChunkKind::TypeAlias, Some(name)));
        }
    }
    if let Some(rest) = line.strip_prefix("mod ") {
        let name = rest.split(';').next()?.trim().to_string();
        if !name.is_empty() {
            return Some((ChunkKind::Module, Some(name)));
        }
    }
    // Top-level impl block
    if let Some(rest) = line.strip_prefix("impl ") {
        let name = rest.split(['{', '<']).next()?.trim().to_string();
        if !name.is_empty() {
            return Some((ChunkKind::Impl, Some(name)));
        }
    }
    None
}

fn extract_js_chunks(source: &SourceFile) -> Vec<CodeChunk> {
    let mut chunks = Vec::new();
    let lines: Vec<&str> = source.source_text.lines().collect();

    for (line_idx, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with("//") {
            continue;
        }

        let (kind, name) = if let Some(result) = match_js_kind(trimmed) {
            result
        } else {
            continue;
        };

        let start_byte = source.source_text[..]
            .match_indices(line)
            .find(|(pos, _)| source.source_text[..*pos].lines().count() == line_idx)
            .map_or(0, |(pos, _)| pos);

        let end_byte = source.source_text[start_byte..]
            .find('\n')
            .map_or(source.source_text.len(), |n| start_byte + n);

        chunks.push(CodeChunk {
            id: CodeChunk::generate_id(&source.path, kind, name.as_deref(), line_idx),
            file_path: source.path.clone(),
            language: source.language,
            kind,
            symbol_name: name,
            span: ChunkSpan::new(start_byte, end_byte, line_idx, line_idx + 1),
            source_text: line.to_string(),
        });
    }

    chunks
}

fn match_js_kind(line: &str) -> Option<(ChunkKind, Option<String>)> {
    if let Some(rest) = line.strip_prefix("function ") {
        let name = rest.split('(').next()?.trim().to_string();
        if !name.is_empty() {
            return Some((ChunkKind::Function, Some(name)));
        }
    }
    if let Some(rest) = line.strip_prefix("class ") {
        let name = rest.split(['{', ' ']).next()?.trim().to_string();
        if !name.is_empty() {
            return Some((ChunkKind::Class, Some(name)));
        }
    }
    if let Some(rest) = line.strip_prefix("const ") {
        let name = rest.split(['=', ':']).next()?.trim().to_string();
        if !name.is_empty() {
            return Some((ChunkKind::Constant, Some(name)));
        }
    }
    None
}

fn extract_ts_chunks(source: &SourceFile) -> Vec<CodeChunk> {
    let mut chunks = Vec::new();
    let lines: Vec<&str> = source.source_text.lines().collect();

    for (line_idx, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with("//") {
            continue;
        }

        let (kind, name) = if let Some(result) = match_ts_kind(trimmed) {
            result
        } else {
            continue;
        };

        let start_byte = source.source_text[..]
            .match_indices(line)
            .find(|(pos, _)| source.source_text[..*pos].lines().count() == line_idx)
            .map_or(0, |(pos, _)| pos);

        let end_byte = source.source_text[start_byte..]
            .find('\n')
            .map_or(source.source_text.len(), |n| start_byte + n);

        chunks.push(CodeChunk {
            id: CodeChunk::generate_id(&source.path, kind, name.as_deref(), line_idx),
            file_path: source.path.clone(),
            language: source.language,
            kind,
            symbol_name: name,
            span: ChunkSpan::new(start_byte, end_byte, line_idx, line_idx + 1),
            source_text: line.to_string(),
        });
    }

    chunks
}

fn match_ts_kind(line: &str) -> Option<(ChunkKind, Option<String>)> {
    if let Some(rest) = line.strip_prefix("function ") {
        let name = rest.split(['(', '<']).next()?.trim().to_string();
        if !name.is_empty() {
            return Some((ChunkKind::Function, Some(name)));
        }
    }
    if let Some(rest) = line.strip_prefix("class ") {
        let name = rest.split(['{', ' ']).next()?.trim().to_string();
        if !name.is_empty() {
            return Some((ChunkKind::Class, Some(name)));
        }
    }
    if let Some(rest) = line.strip_prefix("interface ") {
        let name = rest.split(['{', ' ']).next()?.trim().to_string();
        if !name.is_empty() {
            return Some((ChunkKind::Interface, Some(name)));
        }
    }
    if let Some(rest) = line.strip_prefix("type ") {
        let name = rest.split(['<', '=']).next()?.trim().to_string();
        if !name.is_empty() {
            return Some((ChunkKind::TypeAlias, Some(name)));
        }
    }
    if let Some(rest) = line.strip_prefix("const ") {
        let name = rest.split(['=', ':']).next()?.trim().to_string();
        if !name.is_empty() {
            return Some((ChunkKind::Constant, Some(name)));
        }
    }
    None
}

fn extract_python_chunks(source: &SourceFile) -> Vec<CodeChunk> {
    let mut chunks = Vec::new();
    let lines: Vec<&str> = source.source_text.lines().collect();

    for (line_idx, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        let (kind, name) = if let Some(result) = match_python_kind(trimmed) {
            result
        } else {
            continue;
        };

        let start_byte = source.source_text[..]
            .match_indices(line)
            .find(|(pos, _)| source.source_text[..*pos].lines().count() == line_idx)
            .map_or(0, |(pos, _)| pos);

        let end_byte = source.source_text[start_byte..]
            .find('\n')
            .map_or(source.source_text.len(), |n| start_byte + n);

        chunks.push(CodeChunk {
            id: CodeChunk::generate_id(&source.path, kind, name.as_deref(), line_idx),
            file_path: source.path.clone(),
            language: source.language,
            kind,
            symbol_name: name,
            span: ChunkSpan::new(start_byte, end_byte, line_idx, line_idx + 1),
            source_text: line.to_string(),
        });
    }

    chunks
}

fn match_python_kind(line: &str) -> Option<(ChunkKind, Option<String>)> {
    if let Some(rest) = line.strip_prefix("def ") {
        let name = rest.split('(').next()?.trim().to_string();
        if !name.is_empty() {
            return Some((ChunkKind::Function, Some(name)));
        }
    }
    if let Some(rest) = line.strip_prefix("class ") {
        let name = rest.split(['(', ':']).next()?.trim().to_string();
        if !name.is_empty() {
            return Some((ChunkKind::Class, Some(name)));
        }
    }
    None
}

fn extract_php_chunks(source: &SourceFile) -> Vec<CodeChunk> {
    let mut chunks = Vec::new();
    let lines: Vec<&str> = source.source_text.lines().collect();

    for (line_idx, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with("//") || trimmed.starts_with("<?php") {
            continue;
        }

        let (kind, name) = if let Some(result) = match_php_kind(trimmed) {
            result
        } else {
            continue;
        };

        let start_byte = source.source_text[..]
            .match_indices(line)
            .find(|(pos, _)| source.source_text[..*pos].lines().count() == line_idx)
            .map_or(0, |(pos, _)| pos);

        let end_byte = source.source_text[start_byte..]
            .find('\n')
            .map_or(source.source_text.len(), |n| start_byte + n);

        chunks.push(CodeChunk {
            id: CodeChunk::generate_id(&source.path, kind, name.as_deref(), line_idx),
            file_path: source.path.clone(),
            language: source.language,
            kind,
            symbol_name: name,
            span: ChunkSpan::new(start_byte, end_byte, line_idx, line_idx + 1),
            source_text: line.to_string(),
        });
    }

    chunks
}

fn match_php_kind(line: &str) -> Option<(ChunkKind, Option<String>)> {
    if let Some(rest) = line.strip_prefix("function ") {
        let name = rest.split('(').next()?.trim().to_string();
        if !name.is_empty() {
            return Some((ChunkKind::Function, Some(name)));
        }
    }
    if let Some(rest) = line.strip_prefix("class ") {
        let name = rest.split(['{', ' ']).next()?.trim().to_string();
        if !name.is_empty() {
            return Some((ChunkKind::Class, Some(name)));
        }
    }
    if let Some(rest) = line.strip_prefix("interface ") {
        let name = rest.split(['{', ' ']).next()?.trim().to_string();
        if !name.is_empty() {
            return Some((ChunkKind::Interface, Some(name)));
        }
    }
    None
}
