//! Raw FFI bindings to the llama.cpp C API.
//!
//! This module provides direct, unsafe bindings to the llama.cpp C library
//! functions declared in `llama.h`. These bindings are hand-written to avoid
//! the `bindgen` dependency and only cover the subset of the API used by
//! `chatvcode-llm`.
//!
//! # Safety
//!
//! All functions in this module are `unsafe` and should only be used through
//! the safe wrappers in [`crate::context`].

#![allow(non_camel_case_types, non_snake_case, dead_code)]
#![allow(clippy::missing_safety_doc)]

use std::ffi::{c_char, c_float, c_int, c_void};

use bitflags::bitflags;

// ---------------------------------------------------------------------------
// Opaque types
// ---------------------------------------------------------------------------

/// Opaque handle to a loaded model.
#[repr(C)]
pub struct llama_model {
    _private: [u8; 0],
}

/// Opaque handle to an inference context (KV cache state etc.).
#[repr(C)]
pub struct llama_context {
    _private: [u8; 0],
}

/// Opaque handle to the vocabulary.
#[repr(C)]
pub struct llama_vocab {
    _private: [u8; 0],
}

/// Opaque handle to the memory (KV cache) of a context.
#[repr(C)]
pub struct llama_memory_i {
    _private: [u8; 0],
}

/// Opaque handle to a sampler (or sampler chain).
#[repr(C)]
pub struct llama_sampler {
    _private: [u8; 0],
}

/// Opaque handle to a `LoRA` adapter.
#[repr(C)]
pub struct llama_adapter_lora {
    _private: [u8; 0],
}

// ---------------------------------------------------------------------------
// Primitive type aliases
// ---------------------------------------------------------------------------

pub type llama_token = i32;
pub type llama_pos = i32;
pub type llama_seq_id = i32;
pub type llama_memory_t = *mut llama_memory_i;

// ---------------------------------------------------------------------------
// Enums — Rust-side as #[repr(C)] enums, mapped 1:1 to llama.h
// ---------------------------------------------------------------------------

/// Vocabulary type.
///
/// Mirrors `enum llama_vocab_type` from `<llama.h>`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlamaVocabType {
    None = 0,
    Spm = 1,
    Bpe = 2,
    Wpm = 3,
    Ugm = 4,
    Rwkv = 5,
    Plamo2 = 6,
}

impl LlamaVocabType {
    /// Convert a raw `i32` value from the C API into a [`LlamaVocabType`].
    ///
    /// Returns `None` if the value does not correspond to any known variant.
    #[must_use]
    pub fn from_i32(v: i32) -> Option<Self> {
        match v {
            0 => Some(Self::None),
            1 => Some(Self::Spm),
            2 => Some(Self::Bpe),
            3 => Some(Self::Wpm),
            4 => Some(Self::Ugm),
            5 => Some(Self::Rwkv),
            6 => Some(Self::Plamo2),
            _ => None,
        }
    }
}

/// RoPE scaling type.
///
/// Mirrors `enum llama_rope_scaling_type` from `<llama.h>`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlamaRopeScalingType {
    Unspecified = -1,
    None = 0,
    Linear = 1,
    Yarn = 2,
    Longrope = 3,
}

impl LlamaRopeScalingType {
    /// Convert a raw `i32` value from the C API into a [`LlamaRopeScalingType`].
    #[must_use]
    pub fn from_i32(v: i32) -> Option<Self> {
        match v {
            -1 => Some(Self::Unspecified),
            0 => Some(Self::None),
            1 => Some(Self::Linear),
            2 => Some(Self::Yarn),
            3 => Some(Self::Longrope),
            _ => None,
        }
    }
}

/// Pooling type for embedding contexts.
///
/// Mirrors `enum llama_pooling_type` from `<llama.h>`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlamaPoolingType {
    Unspecified = -1,
    None = 0,
    Mean = 1,
    Cls = 2,
    Last = 3,
    Rank = 4,
}

impl LlamaPoolingType {
    /// Convert a raw `i32` value from the C API into a [`LlamaPoolingType`].
    #[must_use]
    pub fn from_i32(v: i32) -> Option<Self> {
        match v {
            -1 => Some(Self::Unspecified),
            0 => Some(Self::None),
            1 => Some(Self::Mean),
            2 => Some(Self::Cls),
            3 => Some(Self::Last),
            4 => Some(Self::Rank),
            _ => None,
        }
    }
}

/// Model split mode for multi-GPU.
///
/// Mirrors `enum llama_split_mode` from `<llama.h>`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlamaSplitMode {
    None = 0,
    Layer = 1,
    Row = 2,
    Tensor = 3,
}

impl LlamaSplitMode {
    /// Convert a raw `i32` value from the C API into a [`LlamaSplitMode`].
    #[must_use]
    pub fn from_i32(v: i32) -> Option<Self> {
        match v {
            0 => Some(Self::None),
            1 => Some(Self::Layer),
            2 => Some(Self::Row),
            3 => Some(Self::Tensor),
            _ => None,
        }
    }
}

/// Flash attention configuration.
///
/// Mirrors `enum llama_flash_attn_type` from `<llama.h>`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlamaFlashAttnType {
    Auto = -1,
    Disabled = 0,
    Enabled = 1,
}

impl LlamaFlashAttnType {
    /// Convert a raw `i32` value from the C API into a [`LlamaFlashAttnType`].
    #[must_use]
    pub fn from_i32(v: i32) -> Option<Self> {
        match v {
            -1 => Some(Self::Auto),
            0 => Some(Self::Disabled),
            1 => Some(Self::Enabled),
            _ => None,
        }
    }
}

// Legacy type aliases for use in struct fields that the C API treats as i32.
// The enum values will implicitly convert via `as i32` when assigned.
pub type llama_vocab_type = i32;
pub type llama_rope_scaling_type = i32;
pub type llama_pooling_type = i32;
pub type llama_split_mode = i32;
pub type llama_flash_attn_type = i32;
/// Model file type / quantization format.
///
/// Mirrors `enum llama_ftype` from `<llama.h>`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlamaFtype {
    AllF32 = 0,
    MostlyF16 = 1,
    MostlyQ4_0 = 2,
    MostlyQ4_1 = 3,
    MostlyQ8_0 = 7,
    MostlyQ5_0 = 8,
    MostlyQ5_1 = 9,
    MostlyQ2_K = 10,
    MostlyQ3_K_S = 11,
    MostlyQ3_K_M = 12,
    MostlyQ3_K_L = 13,
    MostlyQ4_K_S = 14,
    MostlyQ4_K_M = 15,
    MostlyQ5_K_S = 16,
    MostlyQ5_K_M = 17,
    MostlyQ6_K = 18,
    MostlyIq2Xxs = 19,
    MostlyIq2Xs = 20,
    MostlyQ2_K_S = 21,
    MostlyIq3Xs = 22,
    MostlyIq3Xxs = 23,
    MostlyIq1S = 24,
    MostlyIq4Nl = 25,
    MostlyIq3S = 26,
    MostlyIq3M = 27,
    MostlyIq2S = 28,
    MostlyIq2M = 29,
    MostlyIq4Xs = 30,
    MostlyIq1M = 31,
    MostlyBf16 = 32,
    MostlyTq1_0 = 36,
    MostlyTq2_0 = 37,
    MostlyMxfp4Moe = 38,
    MostlyNvfp4 = 39,
    MostlyQ1_0 = 40,
    Guessed = 1024,
}

impl LlamaFtype {
    /// Convert a raw `i32` value from the C API into a [`LlamaFtype`].
    ///
    /// Returns `None` if the value does not correspond to any known variant.
    #[must_use]
    pub fn from_i32(v: i32) -> Option<Self> {
        match v {
            0 => Some(Self::AllF32),
            1 => Some(Self::MostlyF16),
            2 => Some(Self::MostlyQ4_0),
            3 => Some(Self::MostlyQ4_1),
            7 => Some(Self::MostlyQ8_0),
            8 => Some(Self::MostlyQ5_0),
            9 => Some(Self::MostlyQ5_1),
            10 => Some(Self::MostlyQ2_K),
            11 => Some(Self::MostlyQ3_K_S),
            12 => Some(Self::MostlyQ3_K_M),
            13 => Some(Self::MostlyQ3_K_L),
            14 => Some(Self::MostlyQ4_K_S),
            15 => Some(Self::MostlyQ4_K_M),
            16 => Some(Self::MostlyQ5_K_S),
            17 => Some(Self::MostlyQ5_K_M),
            18 => Some(Self::MostlyQ6_K),
            19 => Some(Self::MostlyIq2Xxs),
            20 => Some(Self::MostlyIq2Xs),
            21 => Some(Self::MostlyQ2_K_S),
            22 => Some(Self::MostlyIq3Xs),
            23 => Some(Self::MostlyIq3Xxs),
            24 => Some(Self::MostlyIq1S),
            25 => Some(Self::MostlyIq4Nl),
            26 => Some(Self::MostlyIq3S),
            27 => Some(Self::MostlyIq3M),
            28 => Some(Self::MostlyIq2S),
            29 => Some(Self::MostlyIq2M),
            30 => Some(Self::MostlyIq4Xs),
            31 => Some(Self::MostlyIq1M),
            32 => Some(Self::MostlyBf16),
            36 => Some(Self::MostlyTq1_0),
            37 => Some(Self::MostlyTq2_0),
            38 => Some(Self::MostlyMxfp4Moe),
            39 => Some(Self::MostlyNvfp4),
            40 => Some(Self::MostlyQ1_0),
            1024 => Some(Self::Guessed),
            _ => None,
        }
    }
}

/// Legacy C-style type alias for [`LlamaFtype`].
pub type llama_ftype = LlamaFtype;

// ---------------------------------------------------------------------------
// Logging
// ---------------------------------------------------------------------------

/// Log levels used by `ggml_log_callback`.
///
/// Mirrors the C `enum ggml_log_level` from `<ggml.h>`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgmlLogLevel {
    None = 0,
    Debug = 1,
    Info = 2,
    Warn = 3,
    Error = 4,
    Cont = 5,
}

pub type ggml_log_callback =
    Option<unsafe extern "C" fn(level: GgmlLogLevel, text: *const c_char, user_data: *mut c_void)>;

// ---------------------------------------------------------------------------
// Token attributes (bit flags)
// ---------------------------------------------------------------------------

bitflags! {
    /// Token attribute bit flags, combinable with `|` and testable with
    /// [`contains()`](bitflags::bitflags::contains).
    ///
    /// Mirrors `enum llama_token_attr` from `<llama.h>`.
    #[repr(C)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct LlamaTokenAttr: i32 {
        /// No attributes set (the empty set).
        const UNDEFINED = 0;
        const UNKNOWN = 1 << 0;
        const UNUSED = 1 << 1;
        const NORMAL = 1 << 2;
        const CONTROL = 1 << 3;
        const USER_DEFINED = 1 << 4;
        const BYTE = 1 << 5;
        const NORMALIZED = 1 << 6;
        const LSTRIP = 1 << 7;
        const RSTRIP = 1 << 8;
        const SINGLE_WORD = 1 << 9;
    }
}

/// Legacy C-style type alias for [`LlamaTokenAttr`].
pub type llama_token_attr = LlamaTokenAttr;

// ---------------------------------------------------------------------------
// Callback type
// ---------------------------------------------------------------------------

pub type llama_progress_callback =
    Option<unsafe extern "C" fn(progress: c_float, user_data: *mut c_void) -> bool>;

// ---------------------------------------------------------------------------
// Structs — must match C layout exactly
// ---------------------------------------------------------------------------

/// Model loading parameters.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct llama_model_params {
    pub devices: *mut *mut c_void, // ggml_backend_dev_t *
    pub tensor_buft_overrides: *const c_void,
    pub n_gpu_layers: i32,
    pub split_mode: llama_split_mode,
    pub main_gpu: i32,
    pub tensor_split: *const c_float,
    pub progress_callback: llama_progress_callback,
    pub progress_callback_user_data: *mut c_void,
    pub kv_overrides: *const c_void,
    pub vocab_only: bool,
    pub use_mmap: bool,
    pub use_direct_io: bool,
    pub use_mlock: bool,
    pub check_tensors: bool,
    pub use_extra_bufts: bool,
    pub no_host: bool,
    pub no_alloc: bool,
}

/// Context (inference runtime) parameters.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct llama_context_params {
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_ubatch: u32,
    pub n_seq_max: u32,
    pub n_rs_seq: u32,
    pub n_threads: i32,
    pub n_threads_batch: i32,
    pub ctx_type: i32,
    pub rope_scaling_type: llama_rope_scaling_type,
    pub pooling_type: llama_pooling_type,
    pub attention_type: i32,
    pub flash_attn_type: llama_flash_attn_type,
    pub rope_freq_base: c_float,
    pub rope_freq_scale: c_float,
    pub yarn_ext_factor: c_float,
    pub yarn_attn_factor: c_float,
    pub yarn_beta_fast: c_float,
    pub yarn_beta_slow: c_float,
    pub yarn_orig_ctx: u32,
    pub defrag_thold: c_float,
    pub cb_eval: *mut c_void,
    pub cb_eval_user_data: *mut c_void,
    pub type_k: i32,
    pub type_v: i32,
    pub abort_callback: *mut c_void,
    pub abort_callback_data: *mut c_void,
    pub embeddings: bool,
    pub offload_kqv: bool,
    pub no_perf: bool,
    pub op_offload: bool,
    pub swa_full: bool,
    pub kv_unified: bool,
    pub samplers: *mut c_void,
    pub n_samplers: usize,
}

/// A single token's data for sampling.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct llama_token_data {
    pub id: llama_token,
    pub logit: c_float,
    pub p: c_float,
}

/// Array of token candidates for sampling.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct llama_token_data_array {
    pub data: *mut llama_token_data,
    pub size: usize,
    pub selected: i64,
    pub sorted: bool,
}

/// A batch of tokens for encoding/decoding.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct llama_batch {
    pub n_tokens: i32,
    pub token: *mut llama_token,
    pub embd: *mut c_float,
    pub pos: *mut llama_pos,
    pub n_seq_id: *mut i32,
    pub seq_id: *mut *mut llama_seq_id,
    pub logits: *mut i8,
}

/// A chat message for template formatting.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct llama_chat_message {
    pub role: *const c_char,
    pub content: *const c_char,
}

// ---------------------------------------------------------------------------
// FFI function declarations
// ---------------------------------------------------------------------------

unsafe extern "C" {
    // ---- Backend init ----
    pub fn llama_backend_init();
    pub fn llama_backend_free();

    // ---- Default params ----
    pub fn llama_model_default_params() -> llama_model_params;
    pub fn llama_context_default_params() -> llama_context_params;

    // ---- Model ----
    pub fn llama_model_load_from_file(
        path_model: *const c_char,
        params: llama_model_params,
    ) -> *mut llama_model;

    pub fn llama_model_free(model: *mut llama_model);

    pub fn llama_model_desc(model: *const llama_model, buf: *mut c_char, buf_size: usize) -> i32;

    pub fn llama_model_size(model: *const llama_model) -> u64;

    pub fn llama_model_n_params(model: *const llama_model) -> u64;

    pub fn llama_model_n_ctx_train(model: *const llama_model) -> i32;

    pub fn llama_model_n_embd(model: *const llama_model) -> i32;

    pub fn llama_model_n_embd_out(model: *const llama_model) -> i32;

    pub fn llama_model_n_layer(model: *const llama_model) -> i32;

    pub fn llama_model_n_head(model: *const llama_model) -> i32;

    pub fn llama_model_n_head_kv(model: *const llama_model) -> i32;

    pub fn llama_model_rope_type(model: *const llama_model) -> i32;

    pub fn llama_model_get_vocab(model: *const llama_model) -> *const llama_vocab;

    pub fn llama_model_chat_template(
        model: *const llama_model,
        name: *const c_char,
    ) -> *const c_char;

    pub fn llama_model_meta_val_str(
        model: *const llama_model,
        key: *const c_char,
        buf: *mut c_char,
        buf_size: usize,
    ) -> i32;

    pub fn llama_model_meta_count(model: *const llama_model) -> i32;

    pub fn llama_model_meta_key_by_index(
        model: *const llama_model,
        i: i32,
        buf: *mut c_char,
        buf_size: usize,
    ) -> i32;

    pub fn llama_model_meta_val_str_by_index(
        model: *const llama_model,
        i: i32,
        buf: *mut c_char,
        buf_size: usize,
    ) -> i32;

    pub fn llama_model_has_encoder(model: *const llama_model) -> bool;
    pub fn llama_model_has_decoder(model: *const llama_model) -> bool;
    pub fn llama_model_decoder_start_token(model: *const llama_model) -> llama_token;

    // ---- Context ----
    pub fn llama_init_from_model(
        model: *mut llama_model,
        params: llama_context_params,
    ) -> *mut llama_context;

    pub fn llama_free(ctx: *mut llama_context);

    pub fn llama_get_model(ctx: *const llama_context) -> *const llama_model;

    pub fn llama_n_ctx(ctx: *const llama_context) -> u32;
    pub fn llama_n_batch(ctx: *const llama_context) -> u32;
    pub fn llama_n_ubatch(ctx: *const llama_context) -> u32;

    pub fn llama_set_n_threads(ctx: *mut llama_context, n_threads: i32, n_threads_batch: i32);
    pub fn llama_n_threads(ctx: *mut llama_context) -> i32;

    // ---- Vocab ----
    pub fn llama_vocab_n_tokens(vocab: *const llama_vocab) -> i32;
    pub fn llama_vocab_type(vocab: *const llama_vocab) -> llama_vocab_type;

    pub fn llama_vocab_get_text(vocab: *const llama_vocab, token: llama_token) -> *const c_char;

    pub fn llama_vocab_is_eog(vocab: *const llama_vocab, token: llama_token) -> bool;

    pub fn llama_vocab_bos(vocab: *const llama_vocab) -> llama_token;
    pub fn llama_vocab_eos(vocab: *const llama_vocab) -> llama_token;
    pub fn llama_vocab_eot(vocab: *const llama_vocab) -> llama_token;
    pub fn llama_vocab_nl(vocab: *const llama_vocab) -> llama_token;

    pub fn llama_vocab_get_add_bos(vocab: *const llama_vocab) -> bool;
    pub fn llama_vocab_get_add_eos(vocab: *const llama_vocab) -> bool;

    // ---- Tokenization ----
    pub fn llama_tokenize(
        vocab: *const llama_vocab,
        text: *const c_char,
        text_len: i32,
        tokens: *mut llama_token,
        n_tokens_max: i32,
        add_special: bool,
        parse_special: bool,
    ) -> i32;

    pub fn llama_token_to_piece(
        vocab: *const llama_vocab,
        token: llama_token,
        buf: *mut c_char,
        length: i32,
        lstrip: i32,
        special: bool,
    ) -> i32;

    pub fn llama_detokenize(
        vocab: *const llama_vocab,
        tokens: *const llama_token,
        n_tokens: i32,
        text: *mut c_char,
        text_len_max: i32,
        remove_special: bool,
        unparse_special: bool,
    ) -> i32;

    // ---- Chat template ----
    pub fn llama_chat_apply_template(
        tmpl: *const c_char,
        chat: *const llama_chat_message,
        n_msg: usize,
        add_ass: bool,
        buf: *mut c_char,
        length: i32,
    ) -> i32;

    // ---- Batch ----
    pub fn llama_batch_get_one(tokens: *mut llama_token, n_tokens: i32) -> llama_batch;
    pub fn llama_batch_init(n_tokens: i32, embd: i32, n_seq_max: i32) -> llama_batch;
    pub fn llama_batch_free(batch: llama_batch);

    // ---- Decode ----
    pub fn llama_decode(ctx: *mut llama_context, batch: llama_batch) -> i32;

    // ---- Logits ----
    pub fn llama_get_logits(ctx: *mut llama_context) -> *mut c_float;
    pub fn llama_get_logits_ith(ctx: *mut llama_context, i: i32) -> *mut c_float;

    // ---- Embeddings ----
    pub fn llama_get_embeddings(ctx: *mut llama_context) -> *const c_float;
    pub fn llama_get_embeddings_ith(ctx: *mut llama_context, i: i32) -> *const c_float;
    pub fn llama_get_embeddings_seq(ctx: *mut llama_context, seq_id: llama_seq_id) -> *const c_float;
    pub fn llama_n_embd(model: *const llama_model) -> i32;

    // ---- Memory management ----
    pub fn llama_get_memory(ctx: *const llama_context) -> *mut llama_memory_i;
    pub fn llama_memory_clear(mem: *mut llama_memory_i, data: bool);

    // ---- Sampler ----
    pub fn llama_sampler_chain_init(params: llama_sampler_chain_params) -> *mut llama_sampler;
    pub fn llama_sampler_chain_add(chain: *mut llama_sampler, smpl: *mut llama_sampler);
    pub fn llama_sampler_chain_n(chain: *const llama_sampler) -> c_int;

    pub fn llama_sampler_init_greedy() -> *mut llama_sampler;
    pub fn llama_sampler_init_dist(seed: u32) -> *mut llama_sampler;
    pub fn llama_sampler_init_top_k(k: i32) -> *mut llama_sampler;
    pub fn llama_sampler_init_top_p(p: c_float, min_keep: usize) -> *mut llama_sampler;
    pub fn llama_sampler_init_min_p(p: c_float, min_keep: usize) -> *mut llama_sampler;
    pub fn llama_sampler_init_temp(t: c_float) -> *mut llama_sampler;
    pub fn llama_sampler_init_penalties(
        penalty_last_n: i32,
        penalty_repeat: c_float,
        penalty_freq: c_float,
        penalty_present: c_float,
    ) -> *mut llama_sampler;

    pub fn llama_sampler_sample(
        smpl: *mut llama_sampler,
        ctx: *mut llama_context,
        idx: i32,
    ) -> llama_token;

    pub fn llama_sampler_accept(smpl: *mut llama_sampler, token: llama_token);
    pub fn llama_sampler_free(smpl: *mut llama_sampler);
    pub fn llama_sampler_reset(smpl: *mut llama_sampler);

    // ---- Misc ----
    pub fn llama_time_us() -> i64;
    pub fn llama_print_system_info() -> *const c_char;

    // ---- Logging ----
    pub fn ggml_log_set(log_callback: ggml_log_callback, user_data: *mut c_void);
    pub fn llama_log_set(log_callback: ggml_log_callback, user_data: *mut c_void);
    pub fn llama_supports_mmap() -> bool;
    pub fn llama_supports_mlock() -> bool;
    pub fn llama_supports_gpu_offload() -> bool;

    // ---- GGML backend registry ----
    pub fn ggml_backend_reg_count() -> usize;
    pub fn ggml_backend_reg_get(index: usize) -> *mut c_void;
    pub fn ggml_backend_reg_name(reg: *mut c_void) -> *const c_char;
    pub fn ggml_backend_reg_dev_count(reg: *mut c_void) -> usize;
    pub fn ggml_backend_reg_dev_get(reg: *mut c_void, index: usize) -> *mut c_void;
    pub fn ggml_backend_dev_name(device: *mut c_void) -> *const c_char;
    pub fn ggml_backend_dev_description(device: *mut c_void) -> *const c_char;
}

/// Sampler chain params (much simpler than the other param structs).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct llama_sampler_chain_params {
    pub no_perf: bool,
}
