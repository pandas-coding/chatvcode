use std::collections::HashMap;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use rayon::prelude::*;

use crate::error::{VdbContext, VdbError, VdbResult};
use crate::model::EmbeddingVector;
use crate::similarity::cosine_similarity;

const MAGIC: [u8; 4] = *b"ATVS";
const VERSION: u32 = 1;

/// Trait for vector storage and similarity search.
///
/// Implementations store [`EmbeddingVector`]s and support:
/// - Adding vectors (with upsert by `chunk_id`)
/// - Top-k similarity search with optional `min_score` filtering
/// - Binary persistence (save/load from disk)
/// - Lookup by `chunk_id`
///
/// The trait requires `Send + Sync` for thread-safe concurrent use.
///
/// # Required methods
///
/// - [`add`](VectorStore::add): Insert one or more vectors.
/// - [`search`](VectorStore::search): Find top-k most similar vectors.
/// - [`save`](VectorStore::save): Serialize to disk.
/// - [`load`](VectorStore::load): Deserialize from disk (associated function).
/// - [`len`](VectorStore::len): Return the number of stored vectors.
/// - [`clear`](VectorStore::clear): Remove all vectors.
/// - [`find`](VectorStore::find): Look up a vector by `chunk_id`.
///
/// # Examples
///
/// ```
/// use chatvcode_vdb::{InMemoryVectorStore, VectorStore, EmbeddingVector};
///
/// let mut store = InMemoryVectorStore::new();
///
/// // Add vectors
/// store.add(vec![
///     EmbeddingVector::new("chunk_a", vec![1.0, 0.0, 0.0]),
///     EmbeddingVector::new("chunk_b", vec![0.0, 1.0, 0.0]),
/// ]).unwrap();
/// assert_eq!(store.len(), 2);
///
/// // Search
/// let results = store.search(&[1.0, 0.0, 0.0], 5, None).unwrap();
/// assert_eq!(results.len(), 2);
/// assert!(results[0].1 > results[1].1); // sorted descending
///
/// // Lookup
/// let found = store.find("chunk_a").unwrap();
/// assert_eq!(found.vector, vec![1.0, 0.0, 0.0]);
/// ```
pub trait VectorStore: Send + Sync {
    fn add(&mut self, vectors: Vec<EmbeddingVector>) -> VdbResult<()>;
    fn remove(&mut self, chunk_ids: &[&str]) -> VdbResult<usize>;
    fn search(
        &self,
        query: &[f32],
        top_k: usize,
        min_score: Option<f32>,
    ) -> VdbResult<Vec<(String, f32)>>;
    fn save(&self, path: &Path) -> VdbResult<()>;
    fn load(path: &Path) -> VdbResult<Self>
    where
        Self: Sized;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn clear(&mut self);
    fn find(&self, chunk_id: &str) -> Option<EmbeddingVector>;
}

/// An in-memory vector store backed by `Vec<EmbeddingVector>` and a `HashMap` index.
///
/// Each vector is stored as a separate [`EmbeddingVector`] struct. A `HashMap`
/// maps chunk IDs to their position in the vector array for O(1) lookup.
/// Search uses parallel iteration via `rayon` for computing cosine similarity.
///
/// # Examples
///
/// ```
/// use chatvcode_vdb::InMemoryVectorStore;
/// use chatvcode_vdb::VectorStore;
///
/// let store = InMemoryVectorStore::new();
/// assert_eq!(store.len(), 0);
/// assert!(store.is_empty());
/// ```
#[derive(Debug)]
pub struct InMemoryVectorStore {
    vectors: Vec<EmbeddingVector>,
    index: HashMap<String, usize>,
    dimension: usize,
}

impl InMemoryVectorStore {
    /// Creates an empty vector store.
    #[must_use]
    pub fn new() -> Self {
        Self { vectors: Vec::new(), index: HashMap::new(), dimension: 0 }
    }

    /// Creates an empty vector store with pre-allocated capacity.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            vectors: Vec::with_capacity(capacity),
            index: HashMap::with_capacity(capacity),
            dimension: 0,
        }
    }

    /// Returns the expected dimension of all vectors in this store.
    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.dimension
    }
}

impl Default for InMemoryVectorStore {
    fn default() -> Self {
        Self::new()
    }
}

impl VectorStore for InMemoryVectorStore {
    fn add(&mut self, vectors: Vec<EmbeddingVector>) -> VdbResult<()> {
        for vector in vectors {
            if self.dimension == 0 {
                self.dimension = vector.dimension;
            } else if vector.dimension != self.dimension {
                return Err(VdbError::invalid_input(format!(
                    "Dimension mismatch: expected {}, got {}",
                    self.dimension, vector.dimension
                ))
                .with_context(VdbContext::default().with_operation("add")));
            }

            if vector.vector.len() != vector.dimension {
                return Err(VdbError::invalid_input(format!(
                    "Vector length {} does not match declared dimension {}",
                    vector.vector.len(),
                    vector.dimension
                ))
                .with_context(VdbContext::default().with_operation("add")));
            }

            if let Some(&existing_idx) = self.index.get(&vector.chunk_id) {
                self.vectors[existing_idx] = vector;
            } else {
                let idx = self.vectors.len();
                self.index.insert(vector.chunk_id.clone(), idx);
                self.vectors.push(vector);
            }
        }
        Ok(())
    }

    fn search(
        &self,
        query: &[f32],
        top_k: usize,
        min_score: Option<f32>,
    ) -> VdbResult<Vec<(String, f32)>> {
        if self.vectors.is_empty() || top_k == 0 {
            return Ok(Vec::new());
        }

        if query.len() != self.dimension {
            return Err(VdbError::invalid_input(format!(
                "Query dimension mismatch: expected {}, got {}",
                self.dimension,
                query.len()
            ))
            .with_context(VdbContext::default().with_operation("search")));
        }

        let query_norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
        if query_norm == 0.0 {
            return Ok(Vec::new());
        }

        let min = min_score.unwrap_or(f32::NEG_INFINITY);

        let mut scores: Vec<(String, f32)> = self
            .vectors
            .par_iter()
            .map(|ev| {
                let score = cosine_similarity(query, &ev.vector);
                (ev.chunk_id.clone(), score)
            })
            .filter(|(_, score)| *score >= min)
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(top_k);

        Ok(scores)
    }

    fn save(&self, path: &Path) -> VdbResult<()> {
        let file = std::fs::File::create(path).map_err(|e| {
            VdbError::io("Failed to create vector store file")
                .with_context(VdbContext::default().with_path(path).with_operation("save"))
                .with_source(e.to_string())
        })?;

        let mut writer = BufWriter::new(file);

        writer.write_all(&MAGIC).map_err(|e| {
            VdbError::io("Failed to write magic bytes")
                .with_context(VdbContext::default().with_path(path).with_operation("save"))
                .with_source(e.to_string())
        })?;

        writer.write_all(&VERSION.to_le_bytes()).map_err(|e| {
            VdbError::io("Failed to write version")
                .with_context(VdbContext::default().with_path(path).with_operation("save"))
                .with_source(e.to_string())
        })?;

        let count = self.vectors.len() as u32;
        writer.write_all(&count.to_le_bytes()).map_err(|e| {
            VdbError::io("Failed to write vector count")
                .with_context(VdbContext::default().with_path(path).with_operation("save"))
                .with_source(e.to_string())
        })?;

        let dim = self.dimension as u32;
        writer.write_all(&dim.to_le_bytes()).map_err(|e| {
            VdbError::io("Failed to write dimension")
                .with_context(VdbContext::default().with_path(path).with_operation("save"))
                .with_source(e.to_string())
        })?;

        for ev in &self.vectors {
            let chunk_id_bytes = ev.chunk_id.as_bytes();
            let chunk_id_len = chunk_id_bytes.len() as u32;
            writer.write_all(&chunk_id_len.to_le_bytes()).map_err(|e| {
                VdbError::io("Failed to write chunk_id length")
                    .with_context(VdbContext::default().with_path(path).with_operation("save"))
                    .with_source(e.to_string())
            })?;
            writer.write_all(chunk_id_bytes).map_err(|e| {
                VdbError::io("Failed to write chunk_id")
                    .with_context(VdbContext::default().with_path(path).with_operation("save"))
                    .with_source(e.to_string())
            })?;

            let vector_bytes: Vec<u8> = ev.vector.iter().flat_map(|f| f.to_le_bytes()).collect();
            writer.write_all(&vector_bytes).map_err(|e| {
                VdbError::io("Failed to write vector data")
                    .with_context(VdbContext::default().with_path(path).with_operation("save"))
                    .with_source(e.to_string())
            })?;
        }

        writer.flush().map_err(|e| {
            VdbError::io("Failed to flush vector store file")
                .with_context(VdbContext::default().with_path(path).with_operation("save"))
                .with_source(e.to_string())
        })?;

        // Ensure data is fully persisted to disk (critical on Windows)
        writer.get_mut().sync_all().map_err(|e| {
            VdbError::io("Failed to sync vector store file to disk")
                .with_context(VdbContext::default().with_path(path).with_operation("save"))
                .with_source(e.to_string())
        })?;

        log::info!("Saved vector store with {} vectors to {}", self.vectors.len(), path.display());

        Ok(())
    }

    fn len(&self) -> usize {
        self.vectors.len()
    }

    fn remove(&mut self, chunk_ids: &[&str]) -> VdbResult<usize> {
        let mut removed = 0;
        let mut indices_to_remove: Vec<usize> = chunk_ids
            .iter()
            .filter_map(|&id| self.index.remove(id))
            .collect();
        indices_to_remove.sort_unstable();
        indices_to_remove.reverse();
        for idx in indices_to_remove {
            if idx < self.vectors.len() {
                self.vectors.remove(idx);
                removed += 1;
            }
        }
        // Rebuild index after removals
        self.index.clear();
        for (i, v) in self.vectors.iter().enumerate() {
            self.index.insert(v.chunk_id.clone(), i);
        }
        Ok(removed)
    }

    fn clear(&mut self) {
        self.vectors.clear();
        self.index.clear();
        self.dimension = 0;
    }

    fn load(path: &Path) -> VdbResult<Self> {
        if !path.exists() {
            return Err(VdbError::io("Vector store file not found")
                .with_context(VdbContext::default().with_path(path).with_operation("load")));
        }

        log::info!("Loading vector store from {}", path.display());

        let file = std::fs::File::open(path).map_err(|e| {
            VdbError::io("Failed to open vector store file")
                .with_context(VdbContext::default().with_path(path).with_operation("load"))
                .with_source(e.to_string())
        })?;

        let mut reader = BufReader::new(file);

        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic).map_err(|e| {
            VdbError::storage("Failed to read magic bytes")
                .with_context(VdbContext::default().with_path(path).with_operation("load"))
                .with_source(e.to_string())
        })?;

        if magic != MAGIC {
            return Err(VdbError::storage(format!(
                "Invalid file format: expected magic {:?}, got {:?}",
                std::str::from_utf8(&MAGIC).unwrap_or("????"),
                std::str::from_utf8(&magic).unwrap_or("????"),
            ))
            .with_context(VdbContext::default().with_path(path).with_operation("load")));
        }

        let mut version_bytes = [0u8; 4];
        reader.read_exact(&mut version_bytes).map_err(|e| {
            VdbError::storage("Failed to read version")
                .with_context(VdbContext::default().with_path(path).with_operation("load"))
                .with_source(e.to_string())
        })?;
        let version = u32::from_le_bytes(version_bytes);

        if version != VERSION {
            return Err(VdbError::storage(format!(
                "Unsupported version: expected {VERSION}, got {version}"
            ))
            .with_context(VdbContext::default().with_path(path).with_operation("load")));
        }

        let mut count_bytes = [0u8; 4];
        reader.read_exact(&mut count_bytes).map_err(|e| {
            VdbError::storage("Failed to read vector count")
                .with_context(VdbContext::default().with_path(path).with_operation("load"))
                .with_source(e.to_string())
        })?;
        let count = u32::from_le_bytes(count_bytes) as usize;

        let mut dim_bytes = [0u8; 4];
        reader.read_exact(&mut dim_bytes).map_err(|e| {
            VdbError::storage("Failed to read dimension")
                .with_context(VdbContext::default().with_path(path).with_operation("load"))
                .with_source(e.to_string())
        })?;
        let dimension = u32::from_le_bytes(dim_bytes) as usize;

        let mut store = Self {
            vectors: Vec::with_capacity(count),
            index: HashMap::with_capacity(count),
            dimension,
        };

        for _ in 0..count {
            let mut chunk_id_len_bytes = [0u8; 4];
            reader.read_exact(&mut chunk_id_len_bytes).map_err(|e| {
                VdbError::storage("Failed to read chunk_id length")
                    .with_context(VdbContext::default().with_path(path).with_operation("load"))
                    .with_source(e.to_string())
            })?;
            let chunk_id_len = u32::from_le_bytes(chunk_id_len_bytes) as usize;

            let mut chunk_id_bytes = vec![0u8; chunk_id_len];
            reader.read_exact(&mut chunk_id_bytes).map_err(|e| {
                VdbError::storage("Failed to read chunk_id")
                    .with_context(VdbContext::default().with_path(path).with_operation("load"))
                    .with_source(e.to_string())
            })?;

            let chunk_id = String::from_utf8(chunk_id_bytes).map_err(|e| {
                VdbError::storage("Invalid chunk_id UTF-8 encoding")
                    .with_context(VdbContext::default().with_path(path).with_operation("load"))
                    .with_source(e.to_string())
            })?;

            let mut vector_bytes = vec![0u8; dimension * 4];
            reader.read_exact(&mut vector_bytes).map_err(|e| {
                VdbError::storage("Failed to read vector data")
                    .with_context(VdbContext::default().with_path(path).with_operation("load"))
                    .with_source(e.to_string())
            })?;

            let vector: Vec<f32> = vector_bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();

            let ev = EmbeddingVector::new(chunk_id.clone(), vector);
            let idx = store.vectors.len();
            store.index.insert(chunk_id, idx);
            store.vectors.push(ev);
        }

        log::info!(
            "Loaded vector store with {} vectors (dimension={})",
            store.vectors.len(),
            dimension
        );

        Ok(store)
    }

    fn find(&self, chunk_id: &str) -> Option<EmbeddingVector> {
        self.index.get(chunk_id).map(|&i| self.vectors[i].clone())
    }
}

/// A memory-efficient vector store using contiguous `Vec<f32>` storage.
///
/// Unlike [`InMemoryVectorStore`], all vector data is stored in a single
/// flat `Vec<f32>` with an offsets array for indexing. Individual vectors
/// are accessed via slices into this contiguous buffer, improving cache
/// locality and reducing per-vector allocation overhead.
///
/// # Examples
///
/// ```
/// use chatvcode_vdb::{CompactVectorStore, VectorStore, EmbeddingVector};
///
/// let mut store = CompactVectorStore::with_capacity(100, 2);
/// store.add(vec![
///     EmbeddingVector::new("c1", vec![1.0, 0.0]),
/// ]).unwrap();
/// assert_eq!(store.len(), 1);
/// ```
#[derive(Debug)]
pub struct CompactVectorStore {
    chunk_ids: Vec<String>,
    vectors: Vec<f32>,   // All vector data stored contiguously
    offsets: Vec<usize>, // Starting offset (in f32 units) for each vector
    dimension: usize,
    index: HashMap<String, usize>, // chunk_id -> index
}

impl CompactVectorStore {
    /// Creates an empty compact vector store.
    #[must_use]
    pub fn new() -> Self {
        Self {
            chunk_ids: Vec::new(),
            vectors: Vec::new(),
            offsets: Vec::new(),
            dimension: 0,
            index: HashMap::new(),
        }
    }

    /// Creates an empty compact vector store with pre-allocated capacity and known dimension.
    #[must_use]
    pub fn with_capacity(capacity: usize, dimension: usize) -> Self {
        Self {
            chunk_ids: Vec::with_capacity(capacity),
            vectors: Vec::with_capacity(capacity * dimension),
            offsets: Vec::with_capacity(capacity),
            dimension,
            index: HashMap::with_capacity(capacity),
        }
    }

    /// Returns the expected dimension of all vectors in this store.
    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.dimension
    }

    fn get_vector(&self, idx: usize) -> &[f32] {
        let start = self.offsets[idx];
        let end = start + self.dimension;
        &self.vectors[start..end]
    }
}

impl Default for CompactVectorStore {
    fn default() -> Self {
        Self::new()
    }
}

impl VectorStore for CompactVectorStore {
    fn add(&mut self, vectors: Vec<EmbeddingVector>) -> VdbResult<()> {
        for vector in vectors {
            if self.dimension == 0 {
                self.dimension = vector.dimension;
            } else if vector.dimension != self.dimension {
                return Err(VdbError::invalid_input(format!(
                    "Dimension mismatch: expected {}, got {}",
                    self.dimension, vector.dimension
                ))
                .with_context(VdbContext::default().with_operation("add")));
            }

            if vector.vector.len() != vector.dimension {
                return Err(VdbError::invalid_input(format!(
                    "Vector length {} does not match declared dimension {}",
                    vector.vector.len(),
                    vector.dimension
                ))
                .with_context(VdbContext::default().with_operation("add")));
            }

            if let Some(&existing_idx) = self.index.get(&vector.chunk_id) {
                // Update existing vector in-place
                let start = self.offsets[existing_idx];
                let end = start + self.dimension;
                self.vectors[start..end].copy_from_slice(&vector.vector);
            } else {
                // Add new vector
                let idx = self.chunk_ids.len();
                let offset = self.vectors.len();
                self.offsets.push(offset);
                self.vectors.extend_from_slice(&vector.vector);
                self.chunk_ids.push(vector.chunk_id.clone());
                self.index.insert(vector.chunk_id, idx);
            }
        }
        Ok(())
    }

    fn search(
        &self,
        query: &[f32],
        top_k: usize,
        min_score: Option<f32>,
    ) -> VdbResult<Vec<(String, f32)>> {
        if self.chunk_ids.is_empty() || top_k == 0 {
            return Ok(Vec::new());
        }

        if query.len() != self.dimension {
            return Err(VdbError::invalid_input(format!(
                "Query dimension mismatch: expected {}, got {}",
                self.dimension,
                query.len()
            ))
            .with_context(VdbContext::default().with_operation("search")));
        }

        let query_norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
        if query_norm == 0.0 {
            return Ok(Vec::new());
        }

        let min = min_score.unwrap_or(f32::NEG_INFINITY);

        let mut scores: Vec<(String, f32)> = (0..self.chunk_ids.len())
            .into_par_iter()
            .map(|idx| {
                let vector = self.get_vector(idx);
                let score = cosine_similarity(query, vector);
                (self.chunk_ids[idx].clone(), score)
            })
            .filter(|(_, score)| *score >= min)
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(top_k);

        Ok(scores)
    }

    fn save(&self, path: &Path) -> VdbResult<()> {
        let file = std::fs::File::create(path).map_err(|e| {
            VdbError::io("Failed to create vector store file")
                .with_context(VdbContext::default().with_path(path).with_operation("save"))
                .with_source(e.to_string())
        })?;

        let mut writer = BufWriter::new(file);

        writer.write_all(&MAGIC).map_err(|e| {
            VdbError::io("Failed to write magic bytes")
                .with_context(VdbContext::default().with_path(path).with_operation("save"))
                .with_source(e.to_string())
        })?;

        writer.write_all(&VERSION.to_le_bytes()).map_err(|e| {
            VdbError::io("Failed to write version")
                .with_context(VdbContext::default().with_path(path).with_operation("save"))
                .with_source(e.to_string())
        })?;

        let count = self.chunk_ids.len() as u32;
        writer.write_all(&count.to_le_bytes()).map_err(|e| {
            VdbError::io("Failed to write vector count")
                .with_context(VdbContext::default().with_path(path).with_operation("save"))
                .with_source(e.to_string())
        })?;

        let dim = self.dimension as u32;
        writer.write_all(&dim.to_le_bytes()).map_err(|e| {
            VdbError::io("Failed to write dimension")
                .with_context(VdbContext::default().with_path(path).with_operation("save"))
                .with_source(e.to_string())
        })?;

        for idx in 0..self.chunk_ids.len() {
            let chunk_id = &self.chunk_ids[idx];
            let chunk_id_bytes = chunk_id.as_bytes();
            let chunk_id_len = chunk_id_bytes.len() as u32;
            writer.write_all(&chunk_id_len.to_le_bytes()).map_err(|e| {
                VdbError::io("Failed to write chunk_id length")
                    .with_context(VdbContext::default().with_path(path).with_operation("save"))
                    .with_source(e.to_string())
            })?;
            writer.write_all(chunk_id_bytes).map_err(|e| {
                VdbError::io("Failed to write chunk_id")
                    .with_context(VdbContext::default().with_path(path).with_operation("save"))
                    .with_source(e.to_string())
            })?;

            let vector = self.get_vector(idx);
            let vector_bytes: Vec<u8> = vector.iter().flat_map(|f| f.to_le_bytes()).collect();
            writer.write_all(&vector_bytes).map_err(|e| {
                VdbError::io("Failed to write vector data")
                    .with_context(VdbContext::default().with_path(path).with_operation("save"))
                    .with_source(e.to_string())
            })?;
        }

        writer.flush().map_err(|e| {
            VdbError::io("Failed to flush vector store file")
                .with_context(VdbContext::default().with_path(path).with_operation("save"))
                .with_source(e.to_string())
        })?;

        // Ensure data is fully persisted to disk (critical on Windows)
        writer.get_mut().sync_all().map_err(|e| {
            VdbError::io("Failed to sync vector store file to disk")
                .with_context(VdbContext::default().with_path(path).with_operation("save"))
                .with_source(e.to_string())
        })?;

        log::info!(
            "Saved compact vector store with {} vectors to {}",
            self.chunk_ids.len(),
            path.display()
        );

        Ok(())
    }

    fn len(&self) -> usize {
        self.chunk_ids.len()
    }

    fn remove(&mut self, chunk_ids: &[&str]) -> VdbResult<usize> {
        let to_remove: std::collections::HashSet<&str> = chunk_ids.iter().copied().collect();

        let keep_indices: Vec<usize> = (0..self.chunk_ids.len())
            .filter(|&i| !to_remove.contains(self.chunk_ids[i].as_str()))
            .collect();

        let removed = self.chunk_ids.len() - keep_indices.len();

        let new_vectors: Vec<f32> = keep_indices
            .iter()
            .flat_map(|&i| {
                let start = self.offsets[i];
                let end = start + self.dimension;
                self.vectors[start..end].to_vec()
            })
            .collect();

        let new_chunk_ids: Vec<String> = keep_indices
            .iter()
            .map(|&i| self.chunk_ids[i].clone())
            .collect();
        let new_offsets: Vec<usize> = (0..keep_indices.len())
            .map(|i| i * self.dimension)
            .collect();

        self.vectors = new_vectors;
        self.chunk_ids = new_chunk_ids;
        self.offsets = new_offsets;
        self.index.clear();
        for (i, id) in self.chunk_ids.iter().enumerate() {
            self.index.insert(id.clone(), i);
        }

        Ok(removed)
    }

    fn clear(&mut self) {
        self.chunk_ids.clear();
        self.vectors.clear();
        self.offsets.clear();
        self.index.clear();
        self.dimension = 0;
    }

    fn load(path: &Path) -> VdbResult<Self> {
        if !path.exists() {
            return Err(VdbError::io("Vector store file not found")
                .with_context(VdbContext::default().with_path(path).with_operation("load")));
        }

        log::info!("Loading compact vector store from {}", path.display());

        let file = std::fs::File::open(path).map_err(|e| {
            VdbError::io("Failed to open vector store file")
                .with_context(VdbContext::default().with_path(path).with_operation("load"))
                .with_source(e.to_string())
        })?;

        let mut reader = BufReader::new(file);

        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic).map_err(|e| {
            VdbError::storage("Failed to read magic bytes")
                .with_context(VdbContext::default().with_path(path).with_operation("load"))
                .with_source(e.to_string())
        })?;

        if magic != MAGIC {
            return Err(VdbError::storage(format!(
                "Invalid file format: expected magic {:?}, got {:?}",
                std::str::from_utf8(&MAGIC).unwrap_or("????"),
                std::str::from_utf8(&magic).unwrap_or("????"),
            ))
            .with_context(VdbContext::default().with_path(path).with_operation("load")));
        }

        let mut version_bytes = [0u8; 4];
        reader.read_exact(&mut version_bytes).map_err(|e| {
            VdbError::storage("Failed to read version")
                .with_context(VdbContext::default().with_path(path).with_operation("load"))
                .with_source(e.to_string())
        })?;
        let version = u32::from_le_bytes(version_bytes);

        if version != VERSION {
            return Err(VdbError::storage(format!(
                "Unsupported version: expected {VERSION}, got {version}"
            ))
            .with_context(VdbContext::default().with_path(path).with_operation("load")));
        }

        let mut count_bytes = [0u8; 4];
        reader.read_exact(&mut count_bytes).map_err(|e| {
            VdbError::storage("Failed to read vector count")
                .with_context(VdbContext::default().with_path(path).with_operation("load"))
                .with_source(e.to_string())
        })?;
        let count = u32::from_le_bytes(count_bytes) as usize;

        let mut dim_bytes = [0u8; 4];
        reader.read_exact(&mut dim_bytes).map_err(|e| {
            VdbError::storage("Failed to read dimension")
                .with_context(VdbContext::default().with_path(path).with_operation("load"))
                .with_source(e.to_string())
        })?;
        let dimension = u32::from_le_bytes(dim_bytes) as usize;

        let mut store = Self::with_capacity(count, dimension);

        for _ in 0..count {
            let mut chunk_id_len_bytes = [0u8; 4];
            reader.read_exact(&mut chunk_id_len_bytes).map_err(|e| {
                VdbError::storage("Failed to read chunk_id length")
                    .with_context(VdbContext::default().with_path(path).with_operation("load"))
                    .with_source(e.to_string())
            })?;
            let chunk_id_len = u32::from_le_bytes(chunk_id_len_bytes) as usize;

            let mut chunk_id_bytes = vec![0u8; chunk_id_len];
            reader.read_exact(&mut chunk_id_bytes).map_err(|e| {
                VdbError::storage("Failed to read chunk_id")
                    .with_context(VdbContext::default().with_path(path).with_operation("load"))
                    .with_source(e.to_string())
            })?;

            let chunk_id = String::from_utf8(chunk_id_bytes).map_err(|e| {
                VdbError::storage("Invalid chunk_id UTF-8 encoding")
                    .with_context(VdbContext::default().with_path(path).with_operation("load"))
                    .with_source(e.to_string())
            })?;

            let mut vector_bytes = vec![0u8; dimension * 4];
            reader.read_exact(&mut vector_bytes).map_err(|e| {
                VdbError::storage("Failed to read vector data")
                    .with_context(VdbContext::default().with_path(path).with_operation("load"))
                    .with_source(e.to_string())
            })?;

            let vector: Vec<f32> = vector_bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();

            // Add to compact store
            let idx = store.chunk_ids.len();
            let offset = store.vectors.len();
            store.offsets.push(offset);
            store.vectors.extend_from_slice(&vector);
            store.chunk_ids.push(chunk_id.clone());
            store.index.insert(chunk_id, idx);
        }

        log::info!(
            "Loaded compact vector store with {} vectors (dimension={})",
            store.chunk_ids.len(),
            dimension
        );

        Ok(store)
    }

    fn find(&self, chunk_id: &str) -> Option<EmbeddingVector> {
        self.index.get(chunk_id).map(|&idx| {
            let vector = self.get_vector(idx).to_vec();
            EmbeddingVector::new(chunk_id, vector)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::EmbeddingVector;

    fn make_vector(id: &str, values: Vec<f32>) -> EmbeddingVector {
        EmbeddingVector::new(id, values)
    }

    #[test]
    fn test_add_and_len() {
        let mut store = InMemoryVectorStore::new();
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);

        store
            .add(vec![make_vector("c1", vec![1.0, 0.0, 0.0])])
            .unwrap();
        assert_eq!(store.len(), 1);
        assert!(!store.is_empty());

        store
            .add(vec![
                make_vector("c2", vec![0.0, 1.0, 0.0]),
                make_vector("c3", vec![0.0, 0.0, 1.0]),
            ])
            .unwrap();
        assert_eq!(store.len(), 3);
    }

    #[test]
    fn test_add_dimension_mismatch() {
        let mut store = InMemoryVectorStore::new();
        store.add(vec![make_vector("c1", vec![1.0, 0.0])]).unwrap();

        let result = store.add(vec![make_vector("c2", vec![1.0, 0.0, 0.0])]);
        assert!(result.is_err());
    }

    #[test]
    fn test_add_upsert_by_chunk_id() {
        let mut store = InMemoryVectorStore::new();
        store.add(vec![make_vector("c1", vec![1.0, 0.0])]).unwrap();
        assert_eq!(store.len(), 1);

        store.add(vec![make_vector("c1", vec![0.0, 1.0])]).unwrap();
        assert_eq!(store.len(), 1);

        let found = store.find("c1").unwrap();
        assert_eq!(found.vector, vec![0.0, 1.0]);
    }

    #[test]
    fn test_find() {
        let mut store = InMemoryVectorStore::new();
        store
            .add(vec![make_vector("c1", vec![1.0, 0.0]), make_vector("c2", vec![0.0, 1.0])])
            .unwrap();

        assert!(store.find("c1").is_some());
        assert!(store.find("c2").is_some());
        assert!(store.find("c3").is_none());

        let found = store.find("c1").unwrap();
        assert_eq!(found.chunk_id, "c1");
        assert_eq!(found.vector, vec![1.0, 0.0]);
    }

    #[test]
    fn test_clear() {
        let mut store = InMemoryVectorStore::new();
        store
            .add(vec![make_vector("c1", vec![1.0, 0.0]), make_vector("c2", vec![0.0, 1.0])])
            .unwrap();
        assert_eq!(store.len(), 2);

        store.clear();
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);
        assert_eq!(store.dimension(), 0);
    }

    #[test]
    fn test_search_top_k() {
        let mut store = InMemoryVectorStore::new();
        store
            .add(vec![
                make_vector("c1", vec![1.0, 0.0, 0.0]),
                make_vector("c2", vec![0.0, 1.0, 0.0]),
                make_vector("c3", vec![0.9, 0.1, 0.0]),
            ])
            .unwrap();

        let results = store.search(&[1.0, 0.0, 0.0], 2, None).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, "c1");
        assert!(results[0].1 > results[1].1);
    }

    #[test]
    fn test_search_empty_store() {
        let store = InMemoryVectorStore::new();
        let results = store.search(&[1.0, 0.0], 5, None).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_query_dimension_mismatch() {
        let mut store = InMemoryVectorStore::new();
        store.add(vec![make_vector("c1", vec![1.0, 0.0])]).unwrap();

        let result = store.search(&[1.0, 0.0, 0.0], 5, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_search_min_score_filter() {
        let mut store = InMemoryVectorStore::new();
        store
            .add(vec![
                make_vector("c1", vec![1.0, 0.0, 0.0]),
                make_vector("c2", vec![0.0, 1.0, 0.0]),
                make_vector("c3", vec![0.9, 0.1, 0.0]),
            ])
            .unwrap();

        let results = store.search(&[1.0, 0.0, 0.0], 10, Some(0.9)).unwrap();
        for (_, score) in &results {
            assert!(*score >= 0.9, "Score {score} below threshold 0.9");
        }
    }

    #[test]
    fn test_search_min_score_filters_all() {
        let mut store = InMemoryVectorStore::new();
        store
            .add(vec![
                make_vector("c1", vec![0.0, 1.0, 0.0]),
                make_vector("c2", vec![0.0, 0.0, 1.0]),
            ])
            .unwrap();

        let results = store.search(&[1.0, 0.0, 0.0], 10, Some(0.5)).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_results_sorted_descending() {
        let mut store = InMemoryVectorStore::new();
        store
            .add(vec![
                make_vector("c1", vec![1.0, 0.0, 0.0]),
                make_vector("c2", vec![0.0, 1.0, 0.0]),
                make_vector(
                    "c3",
                    vec![std::f32::consts::FRAC_1_SQRT_2, std::f32::consts::FRAC_1_SQRT_2, 0.0],
                ),
            ])
            .unwrap();

        let results = store.search(&[1.0, 0.0, 0.0], 10, None).unwrap();
        for i in 1..results.len() {
            assert!(
                results[i - 1].1 >= results[i].1,
                "Results not sorted descending: {} > {}",
                results[i - 1].1,
                results[i].1
            );
        }
    }

    #[test]
    fn test_save_and_load_roundtrip() {
        let dir = std::env::temp_dir().join("chatvcode_vdb_test_save_load");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.vdb");

        let mut store = InMemoryVectorStore::new();
        store
            .add(vec![
                make_vector("c1", vec![1.0, 2.0, 3.0]),
                make_vector("c2", vec![4.0, 5.0, 6.0]),
                make_vector("c3", vec![7.0, 8.0, 9.0]),
            ])
            .unwrap();

        store.save(&path).unwrap();

        let loaded = InMemoryVectorStore::load(&path).unwrap();
        assert_eq!(loaded.len(), 3);
        assert_eq!(loaded.dimension(), 3);

        let v1 = loaded.find("c1").unwrap();
        assert_eq!(v1.vector, vec![1.0, 2.0, 3.0]);

        let v2 = loaded.find("c2").unwrap();
        assert_eq!(v2.vector, vec![4.0, 5.0, 6.0]);

        let v3 = loaded.find("c3").unwrap();
        assert_eq!(v3.vector, vec![7.0, 8.0, 9.0]);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_load_file_not_found() {
        let result = InMemoryVectorStore::load(Path::new("/nonexistent/path/test.vdb"));
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.message.contains("Vector store file not found"));
    }

    #[test]
    fn test_load_invalid_format() {
        let dir = std::env::temp_dir().join("chatvcode_vdb_test_invalid");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bad.vdb");

        std::fs::write(&path, b"NOT_A_VALID_FILE").unwrap();

        let result = InMemoryVectorStore::load(&path);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.message.contains("Invalid file format"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_add_batch_preserves_order() {
        let mut store = InMemoryVectorStore::new();
        let vectors: Vec<EmbeddingVector> = (0..5)
            .map(|i| make_vector(&format!("c{i}"), vec![i as f32, 0.0, 0.0]))
            .collect();
        store.add(vectors).unwrap();
        assert_eq!(store.len(), 5);
        for i in 0..5 {
            let found = store.find(&format!("c{i}")).unwrap();
            assert_eq!(found.vector, vec![i as f32, 0.0, 0.0]);
        }
    }

    #[test]
    fn test_add_and_find_roundtrip() {
        let mut store = InMemoryVectorStore::new();
        let v1 = make_vector("chunk_a", vec![1.0, 2.0, 3.0]);
        let v2 = make_vector("chunk_b", vec![4.0, 5.0, 6.0]);
        store.add(vec![v1, v2]).unwrap();

        let found_a = store.find("chunk_a").unwrap();
        assert_eq!(found_a.chunk_id, "chunk_a");
        assert_eq!(found_a.vector, vec![1.0, 2.0, 3.0]);
        assert_eq!(found_a.dimension, 3);

        let found_b = store.find("chunk_b").unwrap();
        assert_eq!(found_b.chunk_id, "chunk_b");
        assert_eq!(found_b.vector, vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_save_load_preserves_many_vectors() {
        let dir = std::env::temp_dir().join("chatvcode_vdb_test_many_vectors");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("many.vdb");

        let mut store = InMemoryVectorStore::new();
        let dim = 16;
        let count = 50;
        let mut vectors = Vec::with_capacity(count);
        for i in 0..count {
            let vals: Vec<f32> = (0..dim).map(|j| (i * dim + j) as f32 * 0.01).collect();
            vectors.push(make_vector(&format!("vec_{i}"), vals));
        }
        store.add(vectors).unwrap();
        assert_eq!(store.len(), count);

        store.save(&path).unwrap();
        let loaded = InMemoryVectorStore::load(&path).unwrap();
        assert_eq!(loaded.len(), count);
        assert_eq!(loaded.dimension(), dim);

        for i in 0..count {
            let found = loaded.find(&format!("vec_{i}")).unwrap();
            let expected: Vec<f32> = (0..dim).map(|j| (i * dim + j) as f32 * 0.01).collect();
            assert_eq!(found.vector, expected);
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_save_load_with_unicode_chunk_id() {
        let dir = std::env::temp_dir().join("chatvcode_vdb_test_unicode");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("unicode.vdb");

        let mut store = InMemoryVectorStore::new();
        store
            .add(vec![
                make_vector("模块::函数", vec![1.0, 0.0]),
                make_vector("クラス::メソッド", vec![0.0, 1.0]),
            ])
            .unwrap();

        store.save(&path).unwrap();
        let loaded = InMemoryVectorStore::load(&path).unwrap();
        assert_eq!(loaded.len(), 2);
        assert!(loaded.find("模块::函数").is_some());
        assert!(loaded.find("クラス::メソッド").is_some());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_save_load_preserves_search_results() {
        let dir = std::env::temp_dir().join("chatvcode_vdb_test_search_after_load");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("search.vdb");

        let mut store = InMemoryVectorStore::new();
        store
            .add(vec![
                make_vector("c1", vec![1.0, 0.0, 0.0]),
                make_vector("c2", vec![0.0, 1.0, 0.0]),
                make_vector("c3", vec![0.9, 0.1, 0.0]),
            ])
            .unwrap();

        let before = store.search(&[1.0, 0.0, 0.0], 3, None).unwrap();

        store.save(&path).unwrap();
        let loaded = InMemoryVectorStore::load(&path).unwrap();
        let after = loaded.search(&[1.0, 0.0, 0.0], 3, None).unwrap();

        assert_eq!(before.len(), after.len());
        for (b, a) in before.iter().zip(after.iter()) {
            assert_eq!(b.0, a.0);
            assert!((b.1 - a.1).abs() < 1e-6);
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_search_top_k_fewer_than_total() {
        let mut store = InMemoryVectorStore::new();
        store
            .add(vec![
                make_vector("c1", vec![1.0, 0.0]),
                make_vector("c2", vec![0.9, 0.1]),
                make_vector("c3", vec![0.0, 1.0]),
                make_vector("c4", vec![0.1, 0.9]),
            ])
            .unwrap();

        let results = store.search(&[1.0, 0.0], 2, None).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, "c1");
    }

    #[test]
    fn test_search_top_k_greater_than_total() {
        let mut store = InMemoryVectorStore::new();
        store
            .add(vec![make_vector("c1", vec![1.0, 0.0]), make_vector("c2", vec![0.0, 1.0])])
            .unwrap();

        let results = store.search(&[1.0, 0.0], 10, None).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_search_top_k_zero() {
        let mut store = InMemoryVectorStore::new();
        store.add(vec![make_vector("c1", vec![1.0, 0.0])]).unwrap();
        let results = store.search(&[1.0, 0.0], 0, None).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_with_capacity() {
        let store = InMemoryVectorStore::with_capacity(100);
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);
        assert_eq!(store.dimension(), 0);
    }

    #[test]
    fn test_add_vector_length_mismatch_with_dimension() {
        let mut store = InMemoryVectorStore::new();
        let bad_vector =
            EmbeddingVector { chunk_id: "c1".to_string(), vector: vec![1.0, 0.0], dimension: 3 };
        let result = store.add(vec![bad_vector]);
        assert!(result.is_err());
    }
}
