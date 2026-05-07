use atlas_core::{AtlasError, AtlasResult, ErrorContext, FileLanguage, ParseResult, SourceFile};
use tree_sitter::Tree;

use crate::ParserService;

pub fn parse_source(source_file: SourceFile) -> AtlasResult<ParseResult> {
    if !source_file.language.is_supported() {
        return Err(AtlasError::unsupported_language("source file language is not supported yet")
            .with_context(
                ErrorContext::default()
                    .with_operation("parse_source")
                    .with_path(source_file.path.clone())
                    .with_language(source_file.language),
            ));
    }

    let mut service = ParserService::new();
    let tree = service.parse(&source_file)?;

    let errors = collect_parse_errors(&tree, &source_file);

    Ok(ParseResult::success(source_file, Vec::new()).with_errors(errors))
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

pub fn parser_for_language(language: FileLanguage) -> AtlasResult<ParserService> {
    if language.is_supported() {
        Ok(ParserService::new())
    } else {
        Err(AtlasError::unsupported_language("parser is not available for this language")
            .with_context(
                ErrorContext::default()
                    .with_operation("parser_for_language")
                    .with_language(language),
            ))
    }
}
