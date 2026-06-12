# chatvcode-llm

基于 `llama.cpp` FFI 绑定的 LLM 推理引擎，为 ChatVCode 提供模型加载、文本生成和嵌入能力。

## 模块结构

| 模块 | 说明 |
|------|------|
| `ffi` | `llama.cpp` C API 的原始 FFI 绑定 |
| `context` | 模型加载与推理的安全 Rust 封装 |
| `service` | 高层服务抽象（`LlamaService`、`LlamaEmbeddingService`） |
| `types` | 配置、生成参数、响应等数据模型 |
| `log` | ggml/llama.cpp C 层日志到 Rust `log` crate 的桥接 |
| `gguf` | GGUF 文件格式解析与模型自动发现 |
| `error` | 错误类型定义 |

## 快速开始

```rust
use chatvcode_llm::{LlmConfig, LlamaService, LlmService as _, GenerationParams};

let config = LlmConfig::new("~/.chatvcode/models/qwen2.5-coder-7b.gguf")
    .with_n_ctx(8192)
    .with_n_gpu_layers(-1);

let service = LlamaService::new(&config)?;

let response = service.infer(
    "Explain Rust lifetimes",
    &GenerationParams::default(),
    None,
)?;
println!("{}", response.text);
```

## 配置参数（`LlmConfig`）

通过 builder 模式配置模型加载与推理参数：

```rust
let config = LlmConfig::new("/path/to/model.gguf")
    .with_n_ctx(4096)
    .with_n_threads(8)
    .with_n_gpu_layers(-1)
    .with_mmap(true)
    .with_verbose_log(false);
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model_path` | `PathBuf` | — | GGUF 模型文件路径 |
| `n_ctx` | `u32` | `8192` | 上下文窗口大小（最大 token 数），设为 0 使用模型默认值 |
| `n_batch` | `u32` | `8192` | prompt 处理的最大批大小，自动保证 `>= n_ctx` |
| `n_ubatch` | `u32` | `512` | 物理微批大小 |
| `n_threads` | `i32` | CPU 核心数 | 单 token 生成使用的线程数 |
| `n_threads_batch` | `i32` | CPU 核心数 | 批处理/prompt 阶段使用的线程数 |
| `n_gpu_layers` | `i32` | `0` | 卸载到 GPU 的模型层数，`-1` 表示全部，`0` 表示纯 CPU |
| `use_mmap` | `bool` | `true` | 是否使用内存映射 I/O 加载模型 |
| `use_mlock` | `bool` | `false` | 是否将模型页面锁定在 RAM 中（防止被换出到磁盘） |
| `chat_template` | `Option<String>` | `None` | 聊天模板覆盖，`None` 时从 GGUF 元数据自动检测 |
| `verbose_log` | `bool` | `false` | 是否输出 llama.cpp/ggml 的详细日志（见下方说明） |

## 日志控制（`verbose_log`）

### 问题背景

llama.cpp 和 ggml 在模型加载和推理过程中会通过 C 层的 `ggml_log_callback` 输出大量诊断信息，包括：

- 后端注册信息（`register_backend`、`register_device`）
- 模型元数据（`llama_model_loader`、KV 键值对）
- 张量创建日志（`create_tensor: loading tensor blk.X.*`，通常有数百行）

这些日志默认直接写入 stderr，在索引和对话时会产生大量噪音。

### 工作原理

`chatvcode-llm` 通过 `log` 模块安装自定义 C 回调，将 ggml/llama.cpp 的日志桥接到 Rust 的 `log` crate：

- **安静模式**（`verbose_log = false`，默认）：仅转发 `WARN` 和 `ERROR` 级别的消息，`INFO`/`DEBUG`/`CONT` 被静默丢弃
- **详细模式**（`verbose_log = true`）：转发所有级别的消息到 Rust `log`，保留原始日志级别

回调在 `llama_backend_init()` **之前**安装，确保后端注册和张量加载的所有 C 层日志都被正确拦截。该回调使用无锁原子操作，在多线程并行张量加载时是安全的。

### 使用方式

```rust
// 默认安静模式 — 仅显示警告和错误
let config = LlmConfig::new("/path/to/model.gguf");

// 开启详细日志 — 显示模型加载细节、张量创建、后端注册等
let config = LlmConfig::new("/path/to/model.gguf")
    .with_verbose_log(true);
```

也可以通过 `chatvcode_llm::init()` 初始化后端时自动安装安静模式的日志回调：

```rust
chatvcode_llm::init(); // 内部已调用 setup_ggml_logging(false)
```

### 日志级别映射

| ggml 级别 | Rust log 级别 | 安静模式 | 详细模式 |
|-----------|--------------|---------|---------|
| `ERROR` | `error!` | 显示 | 显示 |
| `WARN` | `warn!` | 显示 | 显示 |
| `INFO` | `info!` | 静默丢弃 | 显示 |
| `DEBUG` | `debug!` | 静默丢弃 | 显示 |
| `CONT` | `info!` | 静默丢弃 | 显示 |
| `NONE` | `trace!` | 静默丢弃 | 显示 |
