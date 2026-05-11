use atlas_core::{AtlasError, AtlasResult, ErrorContext, FileLanguage, ParseResult, SourceFile};
use tree_sitter::Tree;

use crate::ParserService;
use crate::chunk::extract_chunks;

/// Parses a source file and extracts all code chunks.
///
/// This is the main entry point for single-file parsing. It:
/// 1. Validates that the language is supported
/// 2. Parses the source into an AST via tree-sitter
/// 3. Collects any syntax errors as warnings
/// 4. Extracts code chunks from the AST
///
/// Returns a [`ParseResult`] with chunks and errors, or an [`AtlasError`]
/// if the language is unsupported or parsing fails entirely.
pub fn parse_source(source_file: SourceFile) -> AtlasResult<ParseResult> {
    if !source_file.language.is_supported() {
        let err = AtlasError::unsupported_language("source file language is not supported yet")
            .with_context(
                ErrorContext::default()
                    .with_operation("parse_source")
                    .with_path(source_file.path.clone())
                    .with_language(source_file.language),
            );
        log::warn!("{}", err);
        return Err(err);
    }

    log::debug!(
        "Parsing {} ({})",
        source_file.path.display(),
        source_file.language
    );

    let mut service = ParserService::new();
    let tree = service.parse(&source_file)?;

    let errors = collect_parse_errors(&tree, &source_file);

    let root = tree.root_node();
    let chunks = extract_chunks(&root, &source_file);

    if !errors.is_empty() {
        log::warn!(
            "Parse issues in {} ({}): {} error(s)",
            source_file.path.display(),
            source_file.language,
            errors.len()
        );
    }

    log::debug!(
        "Extracted {} chunks from {}",
        chunks.len(),
        source_file.path.display()
    );

    Ok(ParseResult::success(source_file, chunks).with_errors(errors))
}

fn collect_parse_errors(tree: &Tree, source_file: &SourceFile) -> Vec<AtlasError> {
    let mut errors = Vec::new();
    let mut cursor = tree.walk();

    fn walk(
        cursor: &mut tree_sitter::TreeCursor,
        errors: &mut Vec<AtlasError>,
        source_file: &SourceFile,
    ) {
        let node = cursor.node();
        if node.is_error() {
            errors.push(
                AtlasError::parse(format!(
                    "unexpected syntax at row {}",
                    node.start_position().row
                ))
                .with_context(
                    ErrorContext::default()
                        .with_operation("parse_source")
                        .with_path(source_file.path.clone())
                        .with_language(source_file.language),
                ),
            );
        } else if node.is_missing() {
            errors.push(
                AtlasError::parse(format!("missing syntax at row {}", node.start_position().row))
                    .with_context(
                        ErrorContext::default()
                            .with_operation("parse_source")
                            .with_path(source_file.path.clone())
                            .with_language(source_file.language),
                    ),
            );
        }

        if cursor.goto_first_child() {
            loop {
                walk(cursor, errors, source_file);
                if !cursor.goto_next_sibling() {
                    break;
                }
            }
            cursor.goto_parent();
        }
    }

    walk(&mut cursor, &mut errors, source_file);
    errors
}

/// Creates a [`ParserService`] configured for the given language.
///
/// Returns an error if the language is not supported.
pub fn parser_for_language(language: FileLanguage) -> AtlasResult<ParserService> {
    if language.is_supported() {
        Ok(ParserService::new())
    } else {
        let err = AtlasError::unsupported_language("parser is not available for this language")
            .with_context(
                ErrorContext::default()
                    .with_operation("parser_for_language")
                    .with_language(language),
            );
        log::warn!("{}", err);
        Err(err)
    }
}
