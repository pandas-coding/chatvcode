use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::{VdbContext, VdbError, VdbResult};
use crate::model::EmbeddingVector;

const QMAGIC: [u8; 4] = *b"ATVQ";
const QVERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationParams {
    pub min: f32,
    pub max: f32,
    pub scale: f32,
    pub zero_point: u8,
}

impl QuantizationParams {
    pub fn compute(vectors: &[Vec<f32>]) -> Self {
        let mut min = f32::MAX;
        let mut max = f32::MIN;
        for v in vectors {
            for &x in v {
                if x < min {
                    min = x;
                }
                if x > max {
                    max = x;
                }
            }
        }
        if min >= max {
            min = -1.0;
            max = 1.0;
        }
        let range = max - min;
        let scale = range / 255.0;
        let zero_point = ((-min / scale).round() as i32).clamp(0, 255) as u8;
        Self { min, max, scale, zero_point }
    }

    pub fn quantize(&self, value: f32) -> u8 {
        ((value / self.scale) + f32::from(self.zero_point))
            .round()
            .clamp(0.0, 255.0) as u8
    }

    pub fn dequantize(&self, quantized: u8) -> f32 {
        (f32::from(quantized) - f32::from(self.zero_point)) * self.scale
    }
}

#[derive(Debug, Clone)]
pub struct QuantizedVectorStore {
    chunk_ids: Vec<String>,
    quantized_vectors: Vec<u8>,
    params: QuantizationParams,
    dimension: usize,
    index: std::collections::HashMap<String, usize>,
}

impl QuantizedVectorStore {
    #[must_use]
    pub fn new() -> Self {
        Self {
            chunk_ids: Vec::new(),
            quantized_vectors: Vec::new(),
            params: QuantizationParams { min: 0.0, max: 1.0, scale: 1.0, zero_point: 0 },
            dimension: 0,
            index: std::collections::HashMap::new(),
        }
    }

    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.dimension
    }

    #[must_use]
    pub const fn len(&self) -> usize {
        self.chunk_ids.len()
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.chunk_ids.is_empty()
    }

    pub fn add(&mut self, vectors: Vec<EmbeddingVector>) -> VdbResult<()> {
        if vectors.is_empty() {
            return Ok(());
        }

        let all_raw: Vec<Vec<f32>> = vectors.iter().map(|v| v.vector.clone()).collect();

        let new_params = if self.is_empty() {
            let p = QuantizationParams::compute(&all_raw);
            self.dimension = vectors[0].dimension;
            self.params = p.clone();
            p
        } else {
            self.params.clone()
        };

        for vector in vectors {
            if vector.dimension != self.dimension {
                return Err(VdbError::invalid_input(format!(
                    "Dimension mismatch: expected {}, got {}",
                    self.dimension, vector.dimension
                ))
                .with_context(VdbContext::default().with_operation("add")));
            }

            let quantized: Vec<u8> = vector
                .vector
                .iter()
                .map(|&x| new_params.quantize(x))
                .collect();

            if let Some(&existing_idx) = self.index.get(&vector.chunk_id) {
                let start = existing_idx * self.dimension;
                self.quantized_vectors[start..start + self.dimension].copy_from_slice(&quantized);
            } else {
                let idx = self.chunk_ids.len();
                self.index.insert(vector.chunk_id.clone(), idx);
                self.chunk_ids.push(vector.chunk_id);
                self.quantized_vectors.extend_from_slice(&quantized);
            }
        }
        Ok(())
    }

    #[must_use]
    pub fn find(&self, chunk_id: &str) -> Option<EmbeddingVector> {
        self.index.get(chunk_id).map(|&idx| {
            let start = idx * self.dimension;
            let end = start + self.dimension;
            let vector: Vec<f32> = self.quantized_vectors[start..end]
                .iter()
                .map(|&q| self.params.dequantize(q))
                .collect();
            EmbeddingVector::new(chunk_id, vector)
        })
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

        let min = min_score.unwrap_or(f32::NEG_INFINITY);

        use rayon::prelude::*;
        let mut scores: Vec<(String, f32)> = self
            .chunk_ids
            .par_iter()
            .enumerate()
            .map(|(idx, chunk_id)| {
                let start = idx * self.dimension;
                let end = start + self.dimension;
                let vector: Vec<f32> = self.quantized_vectors[start..end]
                    .iter()
                    .map(|&q| self.params.dequantize(q))
                    .collect();
                let score = crate::similarity::cosine_similarity(query, &vector);
                (chunk_id.clone(), score)
            })
            .filter(|(_, s)| *s >= min)
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(top_k);
        Ok(scores)
    }

    pub fn clear(&mut self) {
        self.chunk_ids.clear();
        self.quantized_vectors.clear();
        self.index.clear();
        self.dimension = 0;
    }

    pub fn save(&self, path: &Path) -> VdbResult<()> {
        let file = std::fs::File::create(path).map_err(|e| {
            VdbError::io("Failed to create quantized vector store file")
                .with_context(VdbContext::default().with_path(path).with_operation("save"))
                .with_source(e.to_string())
        })?;
        let mut writer = BufWriter::new(file);

        writer.write_all(&QMAGIC).map_err(|e| {
            VdbError::io("Failed to write magic bytes")
                .with_context(VdbContext::default().with_path(path).with_operation("save"))
                .with_source(e.to_string())
        })?;
        writer.write_all(&QVERSION.to_le_bytes()).map_err(|e| {
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

        writer
            .write_all(&self.params.min.to_le_bytes())
            .map_err(|e| {
                VdbError::io("Failed to write quant min")
                    .with_context(VdbContext::default().with_path(path).with_operation("save"))
                    .with_source(e.to_string())
            })?;
        writer
            .write_all(&self.params.max.to_le_bytes())
            .map_err(|e| {
                VdbError::io("Failed to write quant max")
                    .with_context(VdbContext::default().with_path(path).with_operation("save"))
                    .with_source(e.to_string())
            })?;
        writer
            .write_all(&self.params.scale.to_le_bytes())
            .map_err(|e| {
                VdbError::io("Failed to write quant scale")
                    .with_context(VdbContext::default().with_path(path).with_operation("save"))
                    .with_source(e.to_string())
            })?;
        writer.write_all(&[self.params.zero_point]).map_err(|e| {
            VdbError::io("Failed to write quant zero_point")
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

            let start = idx * self.dimension;
            let end = start + self.dimension;
            writer
                .write_all(&self.quantized_vectors[start..end])
                .map_err(|e| {
                    VdbError::io("Failed to write quantized vector data")
                        .with_context(VdbContext::default().with_path(path).with_operation("save"))
                        .with_source(e.to_string())
                })?;
        }

        writer.flush().map_err(|e| {
            VdbError::io("Failed to flush quantized vector store file")
                .with_context(VdbContext::default().with_path(path).with_operation("save"))
                .with_source(e.to_string())
        })?;
        log::info!(
            "Saved quantized vector store with {} vectors ({} KB) to {}",
            self.chunk_ids.len(),
            self.quantized_vectors.len() / 1024,
            path.display()
        );
        Ok(())
    }

    pub fn load(path: &Path) -> VdbResult<Self> {
        if !path.exists() {
            return Err(VdbError::io("Quantized vector store file not found")
                .with_context(VdbContext::default().with_path(path).with_operation("load")));
        }
        log::info!("Loading quantized vector store from {}", path.display());

        let file = std::fs::File::open(path).map_err(|e| {
            VdbError::io("Failed to open quantized vector store file")
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
        if magic != QMAGIC {
            return Err(VdbError::storage(format!(
                "Invalid file format: expected magic {:?}, got {:?}",
                std::str::from_utf8(&QMAGIC).unwrap_or("????"),
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
        if version != QVERSION {
            return Err(VdbError::storage(format!(
                "Unsupported version: expected {QVERSION}, got {version}"
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

        let mut min_bytes = [0u8; 4];
        reader.read_exact(&mut min_bytes).map_err(|e| {
            VdbError::storage("Failed to read quant min")
                .with_context(VdbContext::default().with_path(path).with_operation("load"))
                .with_source(e.to_string())
        })?;
        let min = f32::from_le_bytes(min_bytes);

        let mut max_bytes = [0u8; 4];
        reader.read_exact(&mut max_bytes).map_err(|e| {
            VdbError::storage("Failed to read quant max")
                .with_context(VdbContext::default().with_path(path).with_operation("load"))
                .with_source(e.to_string())
        })?;
        let max = f32::from_le_bytes(max_bytes);

        let mut scale_bytes = [0u8; 4];
        reader.read_exact(&mut scale_bytes).map_err(|e| {
            VdbError::storage("Failed to read quant scale")
                .with_context(VdbContext::default().with_path(path).with_operation("load"))
                .with_source(e.to_string())
        })?;
        let scale = f32::from_le_bytes(scale_bytes);

        let mut zp_byte = [0u8; 1];
        reader.read_exact(&mut zp_byte).map_err(|e| {
            VdbError::storage("Failed to read quant zero_point")
                .with_context(VdbContext::default().with_path(path).with_operation("load"))
                .with_source(e.to_string())
        })?;
        let zero_point = zp_byte[0];

        let params = QuantizationParams { min, max, scale, zero_point };

        let mut store = Self {
            chunk_ids: Vec::with_capacity(count),
            quantized_vectors: vec![0u8; count * dimension],
            params,
            dimension,
            index: std::collections::HashMap::with_capacity(count),
        };

        for idx in 0..count {
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

            let start = idx * dimension;
            let end = start + dimension;
            reader
                .read_exact(&mut store.quantized_vectors[start..end])
                .map_err(|e| {
                    VdbError::storage("Failed to read quantized vector data")
                        .with_context(VdbContext::default().with_path(path).with_operation("load"))
                        .with_source(e.to_string())
                })?;

            store.chunk_ids.push(chunk_id.clone());
            store.index.insert(chunk_id, idx);
        }

        log::info!(
            "Loaded quantized vector store with {} vectors (dimension={}, {} KB)",
            store.chunk_ids.len(),
            dimension,
            store.quantized_vectors.len() / 1024
        );
        Ok(store)
    }
}

impl Default for QuantizedVectorStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantization_params_roundtrip() {
        let v = vec![vec![-1.0, 0.0, 0.5, 1.0]];
        let params = QuantizationParams::compute(&v);
        for &val in &v[0] {
            let q = params.quantize(val);
            let dq = params.dequantize(q);
            assert!((dq - val).abs() < params.scale, "value={val} dequant={dq}");
        }
    }

    #[test]
    fn test_quantized_store_add_and_find() {
        let mut store = QuantizedVectorStore::new();
        let v = EmbeddingVector::new("c1", vec![1.0, 0.0, 0.0]);
        store.add(vec![v]).unwrap();
        let found = store.find("c1").unwrap();
        assert_eq!(found.chunk_id, "c1");
        assert!(found.vector[0] > 0.9);
        assert!(found.vector[1].abs() < 0.1);
    }

    #[test]
    fn test_quantized_store_save_load_roundtrip() {
        let dir = std::env::temp_dir().join("chatvcode_vdb_test_quant");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.qvdb");

        let mut store = QuantizedVectorStore::new();
        store
            .add(vec![
                EmbeddingVector::new("c1", vec![1.0, 2.0, 3.0]),
                EmbeddingVector::new("c2", vec![4.0, 5.0, 6.0]),
            ])
            .unwrap();
        store.save(&path).unwrap();

        let loaded = QuantizedVectorStore::load(&path).unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded.dimension(), 3);
        assert!(loaded.find("c1").is_some());
        assert!(loaded.find("c2").is_some());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_quantized_store_search() {
        let mut store = QuantizedVectorStore::new();
        store
            .add(vec![
                EmbeddingVector::new("c1", vec![1.0, 0.0, 0.0]),
                EmbeddingVector::new("c2", vec![0.0, 1.0, 0.0]),
                EmbeddingVector::new("c3", vec![0.9, 0.1, 0.0]),
            ])
            .unwrap();

        let results = store.search(&[1.0, 0.0, 0.0], 3, None).unwrap();
        assert_eq!(results.len(), 3);
        assert!(results[0].1 > results[1].1);
    }

    #[test]
    fn test_quantized_store_upsert() {
        let mut store = QuantizedVectorStore::new();
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
