use chatvcode_core::{ChatVCodeError, ChatVCodeResult, ErrorContext, FileLanguage, SourceFile};
use tree_sitter::{Language, Parser, Tree};
use tree_sitter_language::LanguageFn;

const fn language_fn_for(lang: FileLanguage) -> Option<LanguageFn> {
    match lang {
        FileLanguage::Rust => Some(tree_sitter_rust::LANGUAGE),
        FileLanguage::JavaScript | FileLanguage::Jsx => Some(tree_sitter_javascript::LANGUAGE),
        FileLanguage::TypeScript => Some(tree_sitter_typescript::LANGUAGE_TYPESCRIPT),
        FileLanguage::Tsx => Some(tree_sitter_typescript::LANGUAGE_TSX),
        FileLanguage::Python => Some(tree_sitter_python::LANGUAGE),
        FileLanguage::Php => Some(tree_sitter_php::LANGUAGE_PHP),
        FileLanguage::Unknown => None,
    }
}

/// Returns the tree-sitter [`Language`] for the given [`FileLanguage`], if supported.
///
/// Returns `None` for `FileLanguage::Unknown`.
pub fn language_for(lang: FileLanguage) -> Option<Language> {
    language_fn_for(lang).map(Language::new)
}

/// Wrapper around `tree_sitter::Parser` for parsing source files.
///
/// Handles language grammar selection and error conversion. Reuse a single
/// `ParserService` instance across multiple parse calls for efficiency.
pub struct ParserService {
    parser: Parser,
}

impl std::fmt::Debug for ParserService {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ParserService").finish_non_exhaustive()
    }
}

impl ParserService {
    /// Creates a new parser service.
    #[must_use]
    pub fn new() -> Self {
        Self { parser: Parser::new() }
    }

    /// Parses a source file into a tree-sitter [`Tree`].
    ///
    /// Returns an error if the language is unsupported or parsing fails.
    pub fn parse(&mut self, source_file: &SourceFile) -> ChatVCodeResult<Tree> {
        let lang = language_for(source_file.language).ok_or_else(|| {
            let err = ChatVCodeError::unsupported_language(format!(
                "no tree-sitter grammar for {}",
                source_file.language
            ))
            .with_context(
                ErrorContext::default()
                    .with_operation("parse")
                    .with_path(source_file.path.clone())
                    .with_language(source_file.language),
            );
            log::error!("{err}");
            err
        })?;

        self.parser.set_language(&lang).map_err(|e| {
            let err = ChatVCodeError::internal(e.to_string())
                .with_context(
                    ErrorContext::default()
                        .with_operation("parse")
                        .with_path(source_file.path.clone())
                        .with_language(source_file.language),
                )
                .with_source(e.to_string());
            log::error!("{err}");
            err
        })?;

        self.parser
            .parse(&source_file.source_text, None)
            .ok_or_else(|| {
                let err = ChatVCodeError::parse(format!(
                    "tree-sitter returned no tree for {}",
                    source_file.path.display()
                ))
                .with_context(
                    ErrorContext::default()
                        .with_operation("parse")
                        .with_path(source_file.path.clone())
                        .with_language(source_file.language),
                );
                log::error!("{err}");
                err
            })
    }
}

impl Default for ParserService {
    fn default() -> Self {
        Self::new()
    }
}
