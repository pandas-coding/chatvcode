# Changelog

All notable changes to this crate will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [2026-06-12]

### Added

- **`--embedding-model` now supports GGUF files**: Previously only accepted ONNX models. The file format is auto-detected by extension (`.gguf` → GGUF, others → ONNX). This allows using a dedicated GGUF embedding model (e.g. Qwen3-Embedding) separately from the LLM inference model for `chat`, `index`, and `search` commands.

### Fixed

- **GGUF embedding extraction used wrong llama.cpp API**: `LlamaEmbeddingContext::embed()` called `llama_get_embeddings_ith(ctx, 0)` (per-token embeddings) instead of `llama_get_embeddings_seq(ctx, 0)` (mean-pooled sequence embeddings). This produced meaningless embedding vectors, causing zero retrieval results in RAG mode. Existing indexes must be rebuilt after this fix.
- **Embedding output dimension now uses `n_embd_out`**: For models whose output embedding dimension differs from the hidden state dimension, the correct `llama_model_n_embd_out()` value is now used instead of `llama_model_n_embd()`.
