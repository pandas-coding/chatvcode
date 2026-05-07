use atlas_core::{AtlasError, AtlasResult, ErrorContext, FileLanguage, SourceFile};
use tree_sitter::{Language, Parser, Tree};
use tree_sitter_language::LanguageFn;

fn language_fn_for(lang: FileLanguage) -> Option<LanguageFn> {
    match lang {
        FileLanguage::Rust => Some(tree_sitter_rust::LANGUAGE),
        FileLanguage::JavaScript | FileLanguage::Jsx => Some(tree_sitter_javascript::LANGUAGE),
        FileLanguage::TypeScript => Some(tree_sitter_typescript::LANGUAGE_TYPESCRIPT),
        FileLanguage::Tsx => Some(tree_sitter_typescript::LANGUAGE_TSX),
        FileLanguage::Unknown => None,
    }
}

pub fn language_for(lang: FileLanguage) -> Option<Language> {
    language_fn_for(lang).map(Language::new)
}

pub struct ParserService {
    parser: Parser,
}

impl ParserService {
    pub fn new() -> Self {
        Self {
            parser: Parser::new(),
        }
    }

    pub fn parse(&mut self, source_file: &SourceFile) -> AtlasResult<Tree> {
        let lang = language_for(source_file.language).ok_or_else(|| {
            AtlasError::unsupported_language(format!(
                "no tree-sitter grammar for {}",
                source_file.language
            ))
            .with_context(
                ErrorContext::default()
                    .with_operation("parse")
                    .with_path(source_file.path.clone())
                    .with_language(source_file.language),
            )
        })?;

        self.parser
            .set_language(&lang)
            .map_err(|e| AtlasError::internal(e.to_string()))?;

        self.parser
            .parse(&source_file.source_text, None)
            .ok_or_else(|| {
                AtlasError::parse(format!(
                    "tree-sitter returned no tree for {}",
                    source_file.path.display()
                ))
                .with_context(
                    ErrorContext::default()
                        .with_operation("parse")
                        .with_path(source_file.path.clone())
                        .with_language(source_file.language),
                )
            })
    }
}

impl Default for ParserService {
    fn default() -> Self {
        Self::new()
    }
}
