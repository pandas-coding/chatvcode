use std::collections::HashMap;
use std::path::Path;

use crate::error::{VdbContext, VdbError, VdbResult};
use crate::model::EmbeddingVector;
use crate::similarity::cosine_similarity;

const DEFAULT_M: usize = 16;
const DEFAULT_EF_CONSTRUCTION: usize = 200;
const DEFAULT_EF_SEARCH: usize = 50;
const HNSW_MAGIC: [u8; 4] = *b"ATHN";
const HNSW_VERSION: u32 = 1;

#[derive(Debug, Clone)]
struct HnswNode {
    chunk_id: String,
    vector: Vec<f32>,
    neighbors: Vec<Vec<usize>>,
}

#[derive(Debug)]
pub struct HnswVectorStore {
    nodes: Vec<HnswNode>,
    index: HashMap<String, usize>,
    dimension: usize,
    entry_point: Option<usize>,
    m: usize,
    m_max: usize,
    m_max0: usize,
    ml: f64,
    ef_construction: usize,
    ef_search: usize,
}

impl HnswVectorStore {
    #[must_use]
    pub fn new() -> Self {
        Self::with_params(DEFAULT_M, DEFAULT_EF_CONSTRUCTION, DEFAULT_EF_SEARCH)
    }

    #[must_use]
    pub fn with_params(m: usize, ef_construction: usize, ef_search: usize) -> Self {
        let m_max = m;
        let m_max0 = m * 2;
        let ml = 1.0 / (m as f64).ln();
        Self {
            nodes: Vec::new(),
            index: HashMap::new(),
            dimension: 0,
            entry_point: None,
            m,
            m_max,
            m_max0,
            ml,
            ef_construction,
            ef_search,
        }
    }

    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.dimension
    }

    #[must_use]
    pub const fn len(&self) -> usize {
        self.nodes.len()
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    fn random_level(&self) -> usize {
        let r: f64 = rand::random();
        ((-r.ln() * self.ml) as usize).min(self.nodes.len())
    }

    fn select_neighbors_simple(&self, candidates: &[(usize, f32)], m: usize) -> Vec<usize> {
        let mut sorted: Vec<_> = candidates.iter().collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        sorted.truncate(m);
        sorted.into_iter().map(|(idx, _)| *idx).collect()
    }

    fn search_layer(
        &self,
        query: &[f32],
        entry: usize,
        ef: usize,
        layer: usize,
    ) -> Vec<(usize, f32)> {
        use std::collections::BinaryHeap;
        let mut visited = vec![false; self.nodes.len()];

        let score = cosine_similarity(query, &self.nodes[entry].vector);
        let mut candidates = BinaryHeap::new();
        candidates.push((OrderedF32(score), entry));
        let mut results = vec![(entry, score)];
        visited[entry] = true;

        while let Some((_, current)) = candidates.pop() {
            let worst_result = results
                .iter()
                .map(|(_, s)| OrderedF32(*s))
                .min()
                .unwrap_or(OrderedF32(f32::NEG_INFINITY));

            let current_score = cosine_similarity(query, &self.nodes[current].vector);
            if current_score < worst_result.0 && results.len() >= ef {
                break;
            }

            let neighbor_indices = if layer < self.nodes[current].neighbors.len() {
                &self.nodes[current].neighbors[layer]
            } else {
                continue;
            };

            for &neighbor_idx in neighbor_indices {
                if visited[neighbor_idx] {
                    continue;
                }
                visited[neighbor_idx] = true;

                let neighbor_score = cosine_similarity(query, &self.nodes[neighbor_idx].vector);
                let worst = results
                    .iter()
                    .map(|(_, s)| OrderedF32(*s))
                    .min()
                    .unwrap_or(OrderedF32(f32::NEG_INFINITY));

                if neighbor_score > worst.0 || results.len() < ef {
                    candidates.push((OrderedF32(neighbor_score), neighbor_idx));
                    results.push((neighbor_idx, neighbor_score));
                    if results.len() > ef {
                        results.sort_by(|a, b| {
                            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                        });
                        results.truncate(ef);
                    }
                }
            }
        }

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    pub fn add(&mut self, vectors: Vec<EmbeddingVector>) -> VdbResult<()> {
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
                let neighbors = self.nodes[existing_idx].neighbors.clone();
                self.nodes[existing_idx] = HnswNode {
                    chunk_id: vector.chunk_id.clone(),
                    vector: vector.vector.clone(),
                    neighbors,
                };
            } else {
                let level = self.random_level();
                let new_idx = self.nodes.len();
                self.index.insert(vector.chunk_id.clone(), new_idx);

                let mut neighbors = vec![Vec::new(); level + 1];

                if let Some(ep) = self.entry_point {
                    let mut current_ep = ep;
                    for lc in (level + 1..=self.nodes[ep].neighbors.len()).rev() {
                        let layer_results =
                            self.search_layer(&vector.vector, current_ep, 1, lc - 1);
                        if let Some(&(best_idx, _)) = layer_results.first() {
                            current_ep = best_idx;
                        }
                    }

                    for lc in (0..=level.min(self.nodes.len().saturating_sub(1))).rev() {
                        let ef = self.ef_construction;
                        let layer_results = self.search_layer(&vector.vector, current_ep, ef, lc);
                        let m_max = if lc == 0 { self.m_max0 } else { self.m_max };
                        let selected = self.select_neighbors_simple(&layer_results, m_max);
                        neighbors[lc] = selected.clone();

                        for &nidx in &neighbors[lc] {
                            let num_layers = self.nodes[nidx].neighbors.len();
                            if lc >= num_layers {
                                continue;
                            }
                            let already_has = self.nodes[nidx].neighbors[lc].contains(&new_idx);
                            if already_has {
                                continue;
                            }
                            self.nodes[nidx].neighbors[lc].push(new_idx);
                            if self.nodes[nidx].neighbors[lc].len() > m_max {
                                let nvec = &self.nodes[nidx].vector;
                                let neighbor_list: Vec<usize> =
                                    self.nodes[nidx].neighbors[lc].clone();
                                let new_vec = &vector.vector;
                                let candidates: Vec<(usize, f32)> = neighbor_list
                                    .iter()
                                    .map(|&c| {
                                        let v = if c == new_idx {
                                            new_vec
                                        } else {
                                            &self.nodes[c].vector
                                        };
                                        (c, cosine_similarity(nvec, v))
                                    })
                                    .collect();
                                self.nodes[nidx].neighbors[lc] =
                                    self.select_neighbors_simple(&candidates, m_max);
                            }
                        }
                    }
                }

                if neighbors.is_empty() {
                    neighbors.push(Vec::new());
                }

                self.nodes.push(HnswNode {
                    chunk_id: vector.chunk_id.clone(),
                    vector: vector.vector.clone(),
                    neighbors,
                });
                self.entry_point = Some(new_idx);
            }
        }
        Ok(())
    }

    pub fn search(
        &self,
        query: &[f32],
        top_k: usize,
        min_score: Option<f32>,
    ) -> VdbResult<Vec<(String, f32)>> {
        if self.is_empty() || top_k == 0 {
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

        let Some(ep) = self.entry_point else {
            return Ok(Vec::new());
        };

        let mut current_ep = ep;
        let max_layer = self.nodes[ep].neighbors.len().saturating_sub(1);
        for lc in (1..=max_layer).rev() {
            let layer_results = self.search_layer(query, current_ep, 1, lc);
            if let Some(&(best_idx, _)) = layer_results.first() {
                current_ep = best_idx;
            }
        }

        let ef = self.ef_search.max(top_k);
        let results = self.search_layer(query, current_ep, ef, 0);

        let min = min_score.unwrap_or(f32::NEG_INFINITY);
        let filtered: Vec<(String, f32)> = results
            .into_iter()
            .filter(|(_, score)| *score >= min)
            .take(top_k)
            .map(|(idx, score)| (self.nodes[idx].chunk_id.clone(), score))
            .collect();

        Ok(filtered)
    }

    pub fn clear(&mut self) {
        self.nodes.clear();
        self.index.clear();
        self.dimension = 0;
        self.entry_point = None;
    }

    #[must_use]
    pub fn find(&self, chunk_id: &str) -> Option<EmbeddingVector> {
        self.index
            .get(chunk_id)
            .map(|&idx| EmbeddingVector::new(chunk_id, self.nodes[idx].vector.clone()))
    }

    pub fn save(&self, path: &Path) -> VdbResult<()> {
        use std::io::{BufWriter, Write};

        let file = std::fs::File::create(path).map_err(|e| {
            VdbError::io("Failed to create HNSW vector store file")
                .with_context(VdbContext::default().with_path(path).with_operation("save"))
                .with_source(e.to_string())
        })?;
        let mut writer = BufWriter::new(file);

        writer.write_all(&HNSW_MAGIC).map_err(|e| {
            VdbError::io("Failed to write magic bytes")
                .with_context(VdbContext::default().with_path(path).with_operation("save"))
                .with_source(e.to_string())
        })?;
        writer.write_all(&HNSW_VERSION.to_le_bytes()).map_err(|e| {
            VdbError::io("Failed to write version")
                .with_context(VdbContext::default().with_path(path).with_operation("save"))
                .with_source(e.to_string())
        })?;

        let count = self.nodes.len() as u32;
        writer.write_all(&count.to_le_bytes()).map_err(|e| {
            VdbError::io("Failed to write node count")
                .with_context(VdbContext::default().with_path(path).with_operation("save"))
                .with_source(e.to_string())
        })?;
        let dim = self.dimension as u32;
        writer.write_all(&dim.to_le_bytes()).map_err(|e| {
            VdbError::io("Failed to write dimension")
                .with_context(VdbContext::default().with_path(path).with_operation("save"))
                .with_source(e.to_string())
        })?;

        let ep = self.entry_point.map_or(u32::MAX, |e| e as u32);
        writer.write_all(&ep.to_le_bytes()).map_err(|e| {
            VdbError::io("Failed to write entry point")
                .with_context(VdbContext::default().with_path(path).with_operation("save"))
                .with_source(e.to_string())
        })?;

        let m = self.m as u32;
        writer.write_all(&m.to_le_bytes()).map_err(|e| {
            VdbError::io("Failed to write M")
                .with_context(VdbContext::default().with_path(path).with_operation("save"))
                .with_source(e.to_string())
        })?;
        let ef_search = self.ef_search as u32;
        writer.write_all(&ef_search.to_le_bytes()).map_err(|e| {
            VdbError::io("Failed to write ef_search")
                .with_context(VdbContext::default().with_path(path).with_operation("save"))
                .with_source(e.to_string())
        })?;

        for node in &self.nodes {
            let chunk_id_bytes = node.chunk_id.as_bytes();
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

            let vector_bytes: Vec<u8> = node.vector.iter().flat_map(|f| f.to_le_bytes()).collect();
            writer.write_all(&vector_bytes).map_err(|e| {
                VdbError::io("Failed to write vector data")
                    .with_context(VdbContext::default().with_path(path).with_operation("save"))
                    .with_source(e.to_string())
            })?;

            let num_layers = node.neighbors.len() as u32;
            writer.write_all(&num_layers.to_le_bytes()).map_err(|e| {
                VdbError::io("Failed to write num_layers")
                    .with_context(VdbContext::default().with_path(path).with_operation("save"))
                    .with_source(e.to_string())
            })?;

            for layer_neighbors in &node.neighbors {
                let num_neighbors = layer_neighbors.len() as u32;
                writer
                    .write_all(&num_neighbors.to_le_bytes())
                    .map_err(|e| {
                        VdbError::io("Failed to write num_neighbors")
                            .with_context(
                                VdbContext::default().with_path(path).with_operation("save"),
                            )
                            .with_source(e.to_string())
                    })?;
                for &nidx in layer_neighbors {
                    writer
                        .write_all(&(nidx as u32).to_le_bytes())
                        .map_err(|e| {
                            VdbError::io("Failed to write neighbor index")
                                .with_context(
                                    VdbContext::default().with_path(path).with_operation("save"),
                                )
                                .with_source(e.to_string())
                        })?;
                }
            }
        }

        writer.flush().map_err(|e| {
            VdbError::io("Failed to flush HNSW vector store file")
                .with_context(VdbContext::default().with_path(path).with_operation("save"))
                .with_source(e.to_string())
        })?;
        log::info!("Saved HNSW vector store with {} nodes to {}", self.nodes.len(), path.display());
        Ok(())
    }

    pub fn load(path: &Path) -> VdbResult<Self> {
        use std::io::{BufReader, Read};

        if !path.exists() {
            return Err(VdbError::io("HNSW vector store file not found")
                .with_context(VdbContext::default().with_path(path).with_operation("load")));
        }
        log::info!("Loading HNSW vector store from {}", path.display());

        let file = std::fs::File::open(path).map_err(|e| {
            VdbError::io("Failed to open HNSW vector store file")
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
        if magic != HNSW_MAGIC {
            return Err(VdbError::storage(format!(
                "Invalid file format: expected magic {:?}, got {:?}",
                std::str::from_utf8(&HNSW_MAGIC).unwrap_or("????"),
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
        if version != HNSW_VERSION {
            return Err(VdbError::storage(format!(
                "Unsupported version: expected {HNSW_VERSION}, got {version}"
            ))
            .with_context(VdbContext::default().with_path(path).with_operation("load")));
        }

        let mut count_bytes = [0u8; 4];
        reader.read_exact(&mut count_bytes).map_err(|e| {
            VdbError::storage("Failed to read node count")
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

        let mut ep_bytes = [0u8; 4];
        reader.read_exact(&mut ep_bytes).map_err(|e| {
            VdbError::storage("Failed to read entry point")
                .with_context(VdbContext::default().with_path(path).with_operation("load"))
                .with_source(e.to_string())
        })?;
        let ep_raw = u32::from_le_bytes(ep_bytes);
        let entry_point = if ep_raw == u32::MAX { None } else { Some(ep_raw as usize) };

        let mut m_bytes = [0u8; 4];
        reader.read_exact(&mut m_bytes).map_err(|e| {
            VdbError::storage("Failed to read M")
                .with_context(VdbContext::default().with_path(path).with_operation("load"))
                .with_source(e.to_string())
        })?;
        let m = u32::from_le_bytes(m_bytes) as usize;

        let mut ef_bytes = [0u8; 4];
        reader.read_exact(&mut ef_bytes).map_err(|e| {
            VdbError::storage("Failed to read ef_search")
                .with_context(VdbContext::default().with_path(path).with_operation("load"))
                .with_source(e.to_string())
        })?;
        let ef_search = u32::from_le_bytes(ef_bytes) as usize;

        let mut store = Self::with_params(m, DEFAULT_EF_CONSTRUCTION, ef_search);
        store.dimension = dimension;

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

            let mut num_layers_bytes = [0u8; 4];
            reader.read_exact(&mut num_layers_bytes).map_err(|e| {
                VdbError::storage("Failed to read num_layers")
                    .with_context(VdbContext::default().with_path(path).with_operation("load"))
                    .with_source(e.to_string())
            })?;
            let num_layers = u32::from_le_bytes(num_layers_bytes) as usize;

            let mut neighbors = Vec::with_capacity(num_layers);
            for _ in 0..num_layers {
                let mut num_neighbors_bytes = [0u8; 4];
                reader.read_exact(&mut num_neighbors_bytes).map_err(|e| {
                    VdbError::storage("Failed to read num_neighbors")
                        .with_context(VdbContext::default().with_path(path).with_operation("load"))
                        .with_source(e.to_string())
                })?;
                let num_neighbors = u32::from_le_bytes(num_neighbors_bytes) as usize;

                let mut layer_neighbors = Vec::with_capacity(num_neighbors);
                for _ in 0..num_neighbors {
                    let mut nidx_bytes = [0u8; 4];
                    reader.read_exact(&mut nidx_bytes).map_err(|e| {
                        VdbError::storage("Failed to read neighbor index")
                            .with_context(
                                VdbContext::default().with_path(path).with_operation("load"),
                            )
                            .with_source(e.to_string())
                    })?;
                    layer_neighbors.push(u32::from_le_bytes(nidx_bytes) as usize);
                }
                neighbors.push(layer_neighbors);
            }

            let idx = store.nodes.len();
            store
                .nodes
                .push(HnswNode { chunk_id: chunk_id.clone(), vector, neighbors });
            store.index.insert(chunk_id, idx);
        }

        store.entry_point = entry_point;

        log::info!(
            "Loaded HNSW vector store with {} nodes (dimension={})",
            store.nodes.len(),
            dimension
        );
        Ok(store)
    }
}

impl Default for HnswVectorStore {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct OrderedF32(f32);

impl Eq for OrderedF32 {}

impl std::cmp::Ord for OrderedF32 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .partial_cmp(&other.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

impl std::cmp::PartialOrd for OrderedF32 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hnsw_add_and_search() {
        let mut store = HnswVectorStore::with_params(16, 100, 20);

        let vectors: Vec<EmbeddingVector> = (0..100)
            .map(|i| {
                let angle = (i as f32) * 0.1;
                EmbeddingVector::new(format!("v{}", i), vec![angle.cos(), angle.sin()])
            })
            .collect();
        store.add(vectors).unwrap();

        assert_eq!(store.len(), 100);

        let results = store.search(&[1.0, 0.0], 5, None).unwrap();
        assert!(!results.is_empty());
        assert!(results[0].1 > results[1].1);
    }

    #[test]
    fn test_hnsw_save_load_roundtrip() {
        let dir = std::env::temp_dir().join("atlas_vdb_test_hnsw");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.hnsw");

        let mut store = HnswVectorStore::with_params(8, 20, 10);
        let vectors: Vec<EmbeddingVector> = (0..30)
            .map(|i| {
                EmbeddingVector::new(format!("v{}", i), vec![(i as f32).cos(), (i as f32).sin()])
            })
            .collect();
        store.add(vectors).unwrap();

        store.save(&path).unwrap();
        let loaded = HnswVectorStore::load(&path).unwrap();
        assert_eq!(loaded.len(), 30);
        assert_eq!(loaded.dimension(), 2);

        let results_before = store.search(&[1.0, 0.0], 3, None).unwrap();
        let results_after = loaded.search(&[1.0, 0.0], 3, None).unwrap();
        assert_eq!(results_before.len(), results_after.len());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_hnsw_empty_search() {
        let store = HnswVectorStore::new();
        let results = store.search(&[1.0, 0.0], 5, None).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_hnsw_upsert() {
        let mut store = HnswVectorStore::new();
        store
            .add(vec![EmbeddingVector::new("c1", vec![1.0, 0.0])])
            .unwrap();
        assert_eq!(store.len(), 1);
        store
            .add(vec![EmbeddingVector::new("c1", vec![0.0, 1.0])])
            .unwrap();
        assert_eq!(store.len(), 1);

        let found = store.find("c1").unwrap();
        assert!(found.vector[1] > 0.9);
    }
}
