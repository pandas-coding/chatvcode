use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use chatvcode_core::model::{ChunkMetadata, SearchOptions, SearchResult};
use chatvcode_core::ParseSource;

use crate::error::AgentError;

pub trait CodeSearchService: Send + Sync {
    fn search(&self, query: &str, options: &SearchOptions) -> Result<Vec<SearchResult>, AgentError>;
}

pub trait ChunkMetadataStoreTrait: Send + Sync {
    fn get_chunks_by_symbol(&self, symbol: &str, kind: Option<&str>) -> Vec<ChunkMetadata>;
    fn get_chunk_by_id(&self, id: &str) -> Option<ChunkMetadata>;
}

pub struct AgentServices {
    pub search: Box<dyn CodeSearchService>,
    pub parser: Box<dyn ParseSource>,
    pub chunk_store: Box<dyn ChunkMetadataStoreTrait>,
}

pub struct ToolContext {
    pub project_path: PathBuf,
    pub timeout: Duration,
    pub token_budget: usize,
    pub services: Arc<AgentServices>,
}

pub struct CoreSearchService {
    project_path: PathBuf,
    parser: Arc<dyn ParseSource>,
    embedding_service: Box<dyn chatvcode_vdb::EmbeddingService>,
}

impl CoreSearchService {
    pub fn new(
        project_path: PathBuf,
        parser: Arc<dyn ParseSource>,
        embedding_service: Box<dyn chatvcode_vdb::EmbeddingService>,
    ) -> Self {
        Self { project_path, parser, embedding_service }
    }
}

impl CodeSearchService for CoreSearchService {
    fn search(&self, query: &str, options: &SearchOptions) -> Result<Vec<SearchResult>, AgentError> {
        chatvcode_core::search_with_service(
            query,
            &self.project_path,
            self.parser.as_ref(),
            options,
            self.embedding_service.as_ref(),
        )
        .map_err(|e| AgentError::ToolError {
            tool_name: "search_code".into(),
            message: e.to_string(),
        })
    }
}

pub struct ChunkMetadataStoreAdapter {
    store: chatvcode_core::model::ChunkMetadataStore,
}

impl ChunkMetadataStoreAdapter {
    pub fn new(store: chatvcode_core::model::ChunkMetadataStore) -> Self {
        Self { store }
    }
}

impl ChunkMetadataStoreTrait for ChunkMetadataStoreAdapter {
    fn get_chunks_by_symbol(&self, symbol: &str, kind: Option<&str>) -> Vec<ChunkMetadata> {
        let symbol_lower = symbol.to_lowercase();
        self.store
            .entries
            .values()
            .filter(|meta| {
                let symbol_match = meta
                    .symbol_name
                    .as_ref()
                    .is_some_and(|name| name.to_lowercase().contains(&symbol_lower));
                let kind_match = kind.is_none_or(|k| {
                    let kind_str = format!("{:?}", meta.kind).to_lowercase();
                    kind_str == k.to_lowercase()
                });
                symbol_match && kind_match
            })
            .cloned()
            .collect()
    }

    fn get_chunk_by_id(&self, id: &str) -> Option<ChunkMetadata> {
        self.store.get(id).cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chatvcode_core::model::{ChunkKind, ChunkMetadata, ChunkMetadataStore};
    use std::path::PathBuf;

    fn make_metadata(chunk_id: &str, symbol: &str, kind: ChunkKind) -> ChunkMetadata {
        ChunkMetadata {
            chunk_id: chunk_id.into(),
            file_path: PathBuf::from("test.rs"),
            language: "rust".into(),
            kind,
            symbol_name: Some(symbol.into()),
            start_line: 1,
            end_line: 10,
            start_byte: 0,
            end_byte: 100,
            source_text: String::new(),
        }
    }

    #[test]
    fn test_adapter_get_chunk_by_id() {
        let mut store = ChunkMetadataStore::new();
        store.insert(make_metadata("c1", "foo", ChunkKind::Function));
        store.insert(make_metadata("c2", "bar", ChunkKind::Struct));

        let adapter = ChunkMetadataStoreAdapter::new(store);

        let result = adapter.get_chunk_by_id("c1");
        assert!(result.is_some());
        assert_eq!(result.unwrap().symbol_name.as_deref(), Some("foo"));

        assert!(adapter.get_chunk_by_id("nonexistent").is_none());
    }

    #[test]
    fn test_adapter_get_chunks_by_symbol() {
        let mut store = ChunkMetadataStore::new();
        store.insert(make_metadata("c1", "foo_bar", ChunkKind::Function));
        store.insert(make_metadata("c2", "baz", ChunkKind::Function));
        store.insert(make_metadata("c3", "FooStruct", ChunkKind::Struct));

        let adapter = ChunkMetadataStoreAdapter::new(store);

        let results = adapter.get_chunks_by_symbol("foo", None);
        assert_eq!(results.len(), 2);

        let results = adapter.get_chunks_by_symbol("foo", Some("function"));
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].chunk_id, "c1");

        let results = adapter.get_chunks_by_symbol("foo", Some("struct"));
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].chunk_id, "c3");
    }

    #[test]
    fn test_adapter_case_insensitive_symbol() {
        let mut store = ChunkMetadataStore::new();
        store.insert(make_metadata("c1", "MyFunction", ChunkKind::Function));

        let adapter = ChunkMetadataStoreAdapter::new(store);

        let results = adapter.get_chunks_by_symbol("myfunction", None);
        assert_eq!(results.len(), 1);

        let results = adapter.get_chunks_by_symbol("MYFUNCTION", None);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_tool_context_arc_sharing() {
        let store = ChunkMetadataStore::new();
        let services = Arc::new(AgentServices {
            search: Box::new(MockSearchService),
            parser: Box::new(|_: chatvcode_core::model::SourceFile| -> chatvcode_core::ChatVCodeResult<chatvcode_core::model::ParseResult> {
                unimplemented!()
            }),
            chunk_store: Box::new(ChunkMetadataStoreAdapter::new(store)),
        });

        let ctx1 = ToolContext {
            project_path: PathBuf::from("/test"),
            timeout: Duration::from_secs(30),
            token_budget: 4096,
            services: Arc::clone(&services),
        };
        let ctx2 = ToolContext {
            project_path: PathBuf::from("/test2"),
            timeout: Duration::from_secs(60),
            token_budget: 8192,
            services: Arc::clone(&services),
        };

        assert!(Arc::ptr_eq(&ctx1.services, &ctx2.services));
    }

    struct MockSearchService;

    impl CodeSearchService for MockSearchService {
        fn search(&self, _query: &str, _options: &SearchOptions) -> Result<Vec<SearchResult>, AgentError> {
            Ok(vec![])
        }
    }
}
