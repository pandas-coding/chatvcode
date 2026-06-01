#![allow(clippy::too_many_lines)]
/// Build script for atlas-llm.
///
/// Compiles `llama.cpp` from `third_party/llama.cpp` using `CMake` and
/// links the resulting static library into the crate.
///
/// # Build Options
///
/// Environment variables that influence the build:
/// - `LLAMA_CUDA=1` — enable CUDA acceleration
/// - `LLAMA_CUDA_ARCH` — CUDA architecture(s), e.g. "86" (default: auto-detect)
/// - `LLAMA_VULKAN=1` — enable Vulkan acceleration
/// - `LLAMA_METAL=1` — enable Metal acceleration (macOS)
/// - `LLAMA_NATIVE=1` — enable `-march=native` optimizations
/// - `LLAMA_DEBUG=1` — build in debug mode
///
/// # Feature Flags
///
/// Cargo features map to the same options:
/// - `cuda` → `LLAMA_CUDA`
/// - `vulkan` → `LLAMA_VULKAN`
/// - `metal` → `LLAMA_METAL`
///
use std::env;
use std::path::{Path, PathBuf};

fn cmake_config_name(profile: &str) -> &'static str {
    if profile == "release" { "Release" } else { "Debug" }
}

fn cmake_cache_value(cache_path: &Path, key: &str) -> Option<String> {
    let cache = std::fs::read_to_string(cache_path).ok()?;
    for line in cache.lines() {
        if let Some((lhs, rhs)) = line.split_once('=')
            && let Some((name, _ty)) = lhs.split_once(':')
            && name == key
        {
            return Some(rhs.trim().to_string());
        }
    }
    None
}

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let third_party = manifest_dir
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("third_party")
        .join("llama.cpp");

    assert!(
        third_party.exists(),
        "llama.cpp source not found at {}. \
         Please run: git clone --depth 1 https://github.com/ggerganov/llama.cpp.git third_party/llama.cpp",
        third_party.display()
    );

    let mut config = cmake::Config::new(&third_party);

    // Configure the build type
    let profile = env::var("PROFILE").unwrap_or_else(|_| "debug".to_string());
    let cmake_cfg = cmake_config_name(&profile);
    config.profile(cmake_cfg);

    // Shared library is not compatible with static linking for a single binary.
    // Always build static.
    config.define("BUILD_SHARED_LIBS", "OFF");
    // Force release CRT (/MD) even in debug builds to match Rust's MSVC runtime.
    // Debug CRT (/MDd) symbols like _CrtDbgReport are not available when linking with Rust.
    if cfg!(target_os = "windows") {
        config.define("CMAKE_MSVC_RUNTIME_LIBRARY", "MultiThreadedDLL");
    }
    config.define("LLAMA_BUILD_TESTS", "OFF");
    config.define("LLAMA_BUILD_EXAMPLES", "OFF");
    config.define("LLAMA_BUILD_SERVER", "OFF");
    // Keep GGML backends but minimize what we build
    config.define("GGML_BUILD_EXAMPLES", "OFF");
    // Disable building CLI tools and apps — we only need the library
    config.define("LLAMA_BUILD_APPS", "OFF");

    // ---- Accelerator flags ----

    let use_cuda = env::var("LLAMA_CUDA")
        .map(|v| v == "1")
        .unwrap_or(cfg!(feature = "cuda"));
    let use_vulkan = env::var("LLAMA_VULKAN")
        .map(|v| v == "1")
        .unwrap_or(cfg!(feature = "vulkan"));
    let use_metal = env::var("LLAMA_METAL")
        .map(|v| v == "1")
        .unwrap_or(cfg!(feature = "metal"));

    if use_cuda {
        println!("cargo:warning=Enabling CUDA acceleration for llama.cpp");
        config.define("GGML_CUDA", "ON");
        // Set explicit CUDA architectures so CMake does not default to "native",
        // which can fail when the compiler test cannot detect a GPU (e.g. in
        // some CI / WSL environments).
        let cuda_arch = env::var("LLAMA_CUDA_ARCH").unwrap_or_else(|_| {
            // RTX 3060 = sm_86
            "86".to_string()
        });
        println!("cargo:warning=Using CUDA architecture: {cuda_arch}");
        config.define("CMAKE_CUDA_ARCHITECTURES", &cuda_arch);

        // Suppress a noisy NVCC warning from upstream ggml-cuda template
        // instantiations on Windows (warning #177-D: declared but never referenced).
        // This is not a functional error, but it can obscure the real build failure.
        if cfg!(target_os = "windows") {
            let cuda_flags =
                env::var("LLAMA_CUDA_FLAGS").unwrap_or_else(|_| "--diag-suppress=177".to_string());
            config.define("CMAKE_CUDA_FLAGS", &cuda_flags);
        }
    }

    // Vulkan is cross-platform (Windows, Linux, Android)
    if use_vulkan {
        println!("cargo:warning=Enabling Vulkan acceleration for llama.cpp");
        config.define("GGML_VULKAN", "ON");
    }

    // Metal is macOS-only
    if use_metal {
        if cfg!(target_os = "macos") {
            println!("cargo:warning=Enabling Metal acceleration for llama.cpp");
            config.define("GGML_METAL", "ON");
        } else {
            println!("cargo:warning=Metal is only available on macOS, skipping");
        }
    }

    // Enable native CPU optimizations
    if env::var("LLAMA_NATIVE").map(|v| v == "1").unwrap_or(false) {
        println!("cargo:warning=Enabling native CPU optimizations (-march=native)");
        config.define("GGML_NATIVE", "ON");
    }

    if env::var("LLAMA_DEBUG").map(|v| v == "1").unwrap_or(false) {
        println!("cargo:warning=Enabling llama.cpp debug mode");
        config.define("LLAMA_DEBUG", "ON");
    }

    // ---- Build ----
    // On Windows, CMake's ExternalProject (vulkan-shaders-gen) creates very deep
    // directory trees that can exceed MAX_PATH (260 chars). Build in a short temp
    // directory to avoid this.
    let build_dir = if cfg!(target_os = "windows") {
        // Use a short but unique temp dir per build-script invocation.
        // A fixed shared temp dir can cause stale CMake cache / parallel build
        // interference, which leads to misleading compiler-detection failures.
        let short = std::env::temp_dir().join(format!("atlas-llm-cmake-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&short);
        println!("cargo:warning=Using CMake build dir: {}", short.display());
        config.out_dir(&short);
        short
    } else {
        PathBuf::new()
    };

    // Build only the llama library target, not the full install
    // (install target may fail if apps/server have dependency issues)
    config.build_target("llama");
    let dst = config.build();
    let dst = if build_dir.as_os_str().is_empty() { dst } else { build_dir };

    // Print link directives
    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    println!("cargo:rustc-link-search=native={}/build/lib", dst.display());
    println!("cargo:rustc-link-search=native={}/build/src/{}", dst.display(), cmake_cfg);
    println!("cargo:rustc-link-search=native={}/build/ggml/src/{}", dst.display(), cmake_cfg);

    let cache_path = dst.join("build").join("CMakeCache.txt");

    // Link llama
    println!("cargo:rustc-link-lib=static=llama");

    // On Windows, we also need to link against the C++ runtime and some system libs.
    // When GPU backends are built as static libs, we must also explicitly link their
    // corresponding backend libraries and SDK import libs from Rust.
    if cfg!(target_os = "windows") {
        println!("cargo:rustc-link-lib=static=ggml");
        println!("cargo:rustc-link-lib=static=ggml-base");
        println!("cargo:rustc-link-lib=static=ggml-cpu");
        println!("cargo:rustc-link-lib=dylib=ole32");
        println!("cargo:rustc-link-lib=dylib=oleaut32");
        println!("cargo:rustc-link-lib=dylib=advapi32");

        if use_cuda {
            println!(
                "cargo:rustc-link-search=native={}/build/ggml/src/ggml-cuda/{}",
                dst.display(),
                cmake_cfg
            );
            println!("cargo:rustc-link-lib=static=ggml-cuda");

            if let Some(cuda_lib_dir) =
                cmake_cache_value(&cache_path, "_cmake_CUDAToolkit_implicit_link_directories")
            {
                for dir in cuda_lib_dir.split(';').filter(|s| !s.is_empty()) {
                    println!("cargo:rustc-link-search=native={dir}");
                }
            }

            println!("cargo:rustc-link-lib=dylib=cudart");
            println!("cargo:rustc-link-lib=dylib=cublas");
            println!("cargo:rustc-link-lib=dylib=cuda");
        }

        if use_vulkan {
            println!(
                "cargo:rustc-link-search=native={}/build/ggml/src/ggml-vulkan/{}",
                dst.display(),
                cmake_cfg
            );
            println!("cargo:rustc-link-lib=static=ggml-vulkan");

            if let Some(vulkan_lib) = cmake_cache_value(&cache_path, "Vulkan_LIBRARY")
                && let Some(parent) = Path::new(&vulkan_lib).parent()
            {
                println!("cargo:rustc-link-search=native={}", parent.display());
            }

            println!("cargo:rustc-link-lib=dylib=vulkan-1");
        }
    }

    // On macOS, link against system frameworks
    if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=framework=Accelerate");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=MetalKit");
    }

    // On Linux, we typically need pthread
    if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-lib=pthread");
        println!("cargo:rustc-link-lib=dl");
    }

    // Re-run build.rs if third_party/llama.cpp changes
    println!("cargo:rerun-if-changed={}", third_party.display());
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=LLAMA_CUDA");
    println!("cargo:rerun-if-env-changed=LLAMA_CUDA_ARCH");
    println!("cargo:rerun-if-env-changed=LLAMA_CUDA_FLAGS");
    println!("cargo:rerun-if-env-changed=LLAMA_VULKAN");
    println!("cargo:rerun-if-env-changed=LLAMA_METAL");
    println!("cargo:rerun-if-env-changed=LLAMA_NATIVE");
    println!("cargo:rerun-if-env-changed=LLAMA_DEBUG");
}
