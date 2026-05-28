# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [2026-05-29]

### Changed

- **`atlas-llm` switched to building `llama.cpp` via CMake in `crates/atlas-llm/build.rs`**: statically builds `third_party/llama.cpp`, disables tests/examples/apps, and supports optional CUDA / Vulkan / Metal backend switches through Cargo features and `LLAMA_*` environment variables.
- **Improved Windows CMake / linking behavior for `atlas-llm`**: `build.rs` now forces MSVC dynamic CRT (`CMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL`), uses a short temp CMake build directory to avoid long-path / stale-cache issues, sets explicit `CMAKE_CUDA_ARCHITECTURES`, and adds backend-specific link search / system library resolution for CUDA and Vulkan.

### Known Issues

- **`atlas-llm` ↔ `llama.cpp` integration is still incomplete**: the synchronous load / inference path exists, but `infer_stream()` is currently only a placeholder fallback and does not yet provide real shared-context streaming generation.
- **Integration still has upstream-coupling risk**: the crate depends on a local `third_party/llama.cpp` checkout and hand-written FFI declarations, so future `llama.cpp` API or CMake layout changes may still cause build or link breakage.

## [2026-05-17]

### Fixed

- **MSVC CRT linkage conflict in debug builds**: `ort-sys` precompiled static libraries use `/MD` (dynamic CRT), while `esaxx-rs` and `onig_sys` (compiled via the `cc` crate) default to `/MT` (static CRT). MSVC prohibits mixing `/MD` and `/MT` in a single binary, causing `LNK2005` / `LNK1169` errors in debug mode.
  - Root cause: Debug CRT (`/MTd` / `/MDd`) instantiates far more template symbols (checked iterators, debug heap at `_ITERATOR_DEBUG_LEVEL > 0`) than Release CRT, making symbol collisions inevitable. Release mode "passes" only because the overlap is smaller — it is not truly compatible.
  - Fix (`.cargo/config.toml`):
    1. `rustflags = ["-C", "target-feature=-crt-static"]` — makes Rust itself link against the dynamic CRT (`/MD`), matching `ort-sys`
    2. `CFLAGS` / `CXXFLAGS = { value = "/MD", force = true }` — forces the `cc` crate to also compile C/C++ sources with `/MD`, ensuring `esaxx-rs` and `onig_sys` use the same CRT variant
    - `force = true` is required to override any pre-existing `CFLAGS`/`CXXFLAGS` from the shell environment
