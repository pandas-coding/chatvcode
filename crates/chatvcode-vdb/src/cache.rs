use std::hash::{Hash, Hasher};
use std::num::NonZeroUsize;
use std::sync::Mutex;

use lru::LruCache;

use crate::error::VdbResult;

#[derive(Debug, Clone)]
struct CacheKey {
    query_hash: u64,
    top_k: usize,
    min_score_bits: u64,
}

impl PartialEq for CacheKey {
    fn eq(&self, other: &Self) -> bool {
        self.query_hash == other.query_hash
            && self.top_k == other.top_k
            && self.min_score_bits == other.min_score_bits
    }
}

impl Eq for CacheKey {}

impl Hash for CacheKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.query_hash.hash(state);
        self.top_k.hash(state);
        self.min_score_bits.hash(state);
    }
}

#[derive(Debug, Clone)]
pub struct CachedSearchResult {
    pub chunk_ids: Vec<String>,
    pub scores: Vec<f32>,
}

pub struct SearchCache {
    cache: Mutex<LruCache<CacheKey, CachedSearchResult>>,
}

impl SearchCache {
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        let cap = NonZeroUsize::new(capacity.max(1)).unwrap();
        Self { cache: Mutex::new(LruCache::new(cap)) }
    }

    fn make_key(query: &str, top_k: usize, min_score: Option<f32>) -> CacheKey {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        query.hash(&mut hasher);
        let query_hash = hasher.finish();

        let min_score_bits = min_score.map_or(u64::MAX, |s| u64::from(s.to_bits()));

        CacheKey { query_hash, top_k, min_score_bits }
    }

    pub fn get(
        &self,
        query: &str,
        top_k: usize,
        min_score: Option<f32>,
    ) -> Option<CachedSearchResult> {
        let key = Self::make_key(query, top_k, min_score);
        let mut cache = self.cache.lock().ok()?;
        cache.get(&key).cloned()
    }

    pub fn put(
        &self,
        query: &str,
        top_k: usize,
        min_score: Option<f32>,
        result: CachedSearchResult,
    ) {
        let key = Self::make_key(query, top_k, min_score);
        if let Ok(mut cache) = self.cache.lock() {
            cache.put(key, result);
        }
    }

    pub fn len(&self) -> usize {
        self.cache.lock().map(|c| c.len()).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn clear(&self) {
        if let Ok(mut cache) = self.cache.lock() {
            cache.clear();
        }
    }
}

impl Default for SearchCache {
    fn default() -> Self {
        Self::new(256)
    }
}

pub fn cached_search<F>(
    cache: &SearchCache,
    query: &str,
    top_k: usize,
    min_score: Option<f32>,
    search_fn: F,
) -> VdbResult<Vec<(String, f32)>>
where
    F: FnOnce() -> VdbResult<Vec<(String, f32)>>,
{
    if let Some(cached) = cache.get(query, top_k, min_score) {
        log::debug!("Cache hit for query: {query:?}");
        let results: Vec<(String, f32)> = cached.chunk_ids.into_iter().zip(cached.scores).collect();
        return Ok(results);
    }

    log::debug!("Cache miss for query: {query:?}");
    let results = search_fn()?;

    cache.put(
        query,
        top_k,
        min_score,
        CachedSearchResult {
            chunk_ids: results.iter().map(|(id, _)| id.clone()).collect(),
            scores: results.iter().map(|(_, s)| *s).collect(),
        },
    );

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_hit() {
        let cache = SearchCache::new(10);
        let query = "find error handling";
        let top_k = 5;
        let min_score = Some(0.5);

        assert!(cache.get(query, top_k, min_score).is_none());

        cache.put(
            query,
            top_k,
            min_score,
            CachedSearchResult { chunk_ids: vec!["c1".to_string()], scores: vec![0.9] },
        );

        let cached = cache.get(query, top_k, min_score).unwrap();
        assert_eq!(cached.chunk_ids, vec!["c1"]);
        assert_eq!(cached.scores, vec![0.9]);
    }

    #[test]
    fn test_cache_different_params_miss() {
        let cache = SearchCache::new(10);
        cache.put(
            "query",
            5,
            None,
            CachedSearchResult { chunk_ids: vec!["a".into()], scores: vec![1.0] },
        );

        assert!(cache.get("query", 10, None).is_none());
        assert!(cache.get("query", 5, Some(0.5)).is_none());
        assert!(cache.get("other", 5, None).is_none());
    }

    #[test]
    fn test_cache_clear() {
        let cache = SearchCache::new(10);
        cache.put(
            "q",
            1,
            None,
            CachedSearchResult { chunk_ids: vec!["x".into()], scores: vec![1.0] },
        );
        assert_eq!(cache.len(), 1);
        cache.clear();
        assert_eq!(cache.len(), 0);
    }
}
