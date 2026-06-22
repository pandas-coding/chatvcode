//! Sampler chain configuration for text generation.
//!
//! This module provides functions to build llama.cpp sampler chains
//! based on [`GenerationParams`](crate::types::GenerationParams).
//!
//! Samplers control how tokens are selected during generation:
//! - **Penalties**: Repeat penalty, frequency penalty, presence penalty
//! - **Top-k**: Limits sampling to the k most likely tokens
//! - **Top-p (nucleus)**: Limits sampling to tokens with cumulative probability <= p
//! - **Min-p**: Filters tokens with probability < min_p * max_probability
//! - **Temperature**: Controls randomness (0.0 = greedy, higher = more random)
//!
//! # Example
//!
//! ```ignore
//! use chatvcode_llm::sampler::build_sampler_chain;
//! use chatvcode_llm::GenerationParams;
//!
//! let params = GenerationParams::default()
//!     .with_temperature(0.7)
//!     .with_top_p(0.9)
//!     .with_top_k(40);
//!
//! let sampler = build_sampler_chain(&params);
//! // Use sampler with llama.cpp inference...
//! ```

use crate::ffi;
use crate::types::GenerationParams;

/// Build a sampler chain from generation parameters.
///
/// Creates a llama.cpp sampler chain that applies the following samplers
/// in order (when enabled):
///
/// 1. **Repeat penalty** — penalizes tokens that appeared recently
/// 2. **Top-k** — keeps only the k most likely tokens
/// 3. **Top-p** — keeps tokens with cumulative probability <= p
/// 4. **Min-p** — filters tokens below min_p * max_probability
/// 5. **Temperature + sampling** — applies temperature and samples from distribution
///    (or greedy if temperature <= 0.0)
///
/// # Arguments
///
/// * `params` — Generation parameters controlling sampling behavior
///
/// # Returns
///
/// A pointer to the created sampler chain. The caller is responsible for
/// freeing it with `llama_sampler_free`.
pub fn build_sampler_chain(params: &GenerationParams) -> *mut ffi::llama_sampler {
    unsafe {
        let chain = ffi::llama_sampler_chain_init(ffi::llama_sampler_chain_params { no_perf: false });

        if params.repeat_penalty != 1.0 {
            let penalties = ffi::llama_sampler_init_penalties(
                params.repeat_last_n,
                params.repeat_penalty,
                0.0,
                0.0,
            );
            ffi::llama_sampler_chain_add(chain, penalties);
        }

        if params.top_k > 0 {
            let top_k = ffi::llama_sampler_init_top_k(params.top_k);
            ffi::llama_sampler_chain_add(chain, top_k);
        }

        if params.top_p < 1.0 {
            let top_p = ffi::llama_sampler_init_top_p(params.top_p, 1);
            ffi::llama_sampler_chain_add(chain, top_p);
        }

        if params.min_p > 0.0 {
            let min_p = ffi::llama_sampler_init_min_p(params.min_p, 1);
            ffi::llama_sampler_chain_add(chain, min_p);
        }

        if params.temperature <= 0.0 {
            let greedy = ffi::llama_sampler_init_greedy();
            ffi::llama_sampler_chain_add(chain, greedy);
        } else {
            let temp = ffi::llama_sampler_init_temp(params.temperature);
            ffi::llama_sampler_chain_add(chain, temp);

            let dist = ffi::llama_sampler_init_dist(params.seed);
            ffi::llama_sampler_chain_add(chain, dist);
        }

        chain
    }
}

/// Create a default sampler chain with standard parameters.
///
/// Uses the following defaults:
/// - Top-k: 40
/// - Top-p: 0.9
/// - Temperature: 0.7
/// - Seed: random (u32::MAX)
///
/// # Returns
///
/// A pointer to the created sampler chain. The caller is responsible for
/// freeing it with `llama_sampler_free`.
pub fn default_sampler_chain() -> *mut ffi::llama_sampler {
    unsafe {
        let chain = ffi::llama_sampler_chain_init(ffi::llama_sampler_chain_params { no_perf: false });
        let top_k = ffi::llama_sampler_init_top_k(40);
        ffi::llama_sampler_chain_add(chain, top_k);
        let top_p = ffi::llama_sampler_init_top_p(0.9, 1);
        ffi::llama_sampler_chain_add(chain, top_p);
        let temp = ffi::llama_sampler_init_temp(0.7);
        ffi::llama_sampler_chain_add(chain, temp);
        let dist = ffi::llama_sampler_init_dist(u32::MAX);
        ffi::llama_sampler_chain_add(chain, dist);
        chain
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_sampler_chain_greedy() {
        crate::backend::init();
        let params = GenerationParams::greedy();
        let sampler = build_sampler_chain(&params);
        assert!(!sampler.is_null());
        unsafe { ffi::llama_sampler_free(sampler) };
    }

    #[test]
    fn test_build_sampler_chain_default() {
        crate::backend::init();
        let params = GenerationParams::default();
        let sampler = build_sampler_chain(&params);
        assert!(!sampler.is_null());
        unsafe { ffi::llama_sampler_free(sampler) };
    }

    #[test]
    fn test_default_sampler_chain() {
        crate::backend::init();
        let sampler = default_sampler_chain();
        assert!(!sampler.is_null());
        unsafe { ffi::llama_sampler_free(sampler) };
    }
}
