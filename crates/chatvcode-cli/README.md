# chatvcode-cli

ChatVCode 的命令行入口，提供代码索引、语义搜索和 AI 对话功能。

## 快速开始

### 准备模型

将 GGUF 格式的大语言模型放入默认目录：

```bash
mkdir -p ~/.chatvcode/models

# 推荐模型（任选一个）：
# Qwen2.5-Coder-7B-Instruct（编码能力强，推荐大多数用户）
curl -Lo ~/.chatvcode/models/qwen2.5-coder-7b.gguf "<GGUF下载链接>"
```

> 模型须为 GGUF v2/v3 格式。如果 `~/.chatvcode/models/` 下仅有一个 `.gguf` 文件，CLI 会自动发现并使用它。

### 构建项目

```bash
# 在项目根目录
cargo build --release

# 二进制文件位于 target/release/chatvcode
```

## 命令行对话大模型

### 模式一：纯 LLM 对话（`--retrieval=false`）

不依赖索引和向量库，直接与大模型对话。适合测试模型加载、推理是否正常，或进行通用问答。

```bash
# 流式输出（默认，逐 token 打印）
chatvcode chat "What is Rust?" --retrieval=false

# 非流式输出（等生成完毕后一次性输出）
chatvcode chat "What is 2+2?" --retrieval=false --stream=false

# 指定模型文件
chatvcode chat "Explain closures" --retrieval=false --model=/path/to/model.gguf

# 自定义生成参数
chatvcode chat "Write a quicksort" --retrieval=false \
    --temperature=0.3 \
    --max-tokens=1024 \
    --top-k=40 \
    --top-p=0.9

# 自定义系统提示
chatvcode chat "Explain this regex" --retrieval=false \
    --system-prompt="You are a regex expert."

# 指定 chat 模板
chatvcode chat "hello" --retrieval=false --template=chatml

# JSON 格式输出
chatvcode chat "What is ownership in Rust?" --retrieval=false --json

# GPU 加速（将模型层卸载到 GPU）
chatvcode chat "hello" --retrieval=false --n-gpu-layers=-1

# 调试模式（显示 llama.cpp/ggml 详细日志：tensor 创建、backend 注册等）
chatvcode chat "hello" --retrieval=false --llm-verbose-log
```

### 模式二：RAG 增强对话（需要先建索引）

先对项目建索引，再基于代码库上下文进行问答：

```bash
# 第一步：索引项目（使用 GGUF 模型做嵌入，自动发现 ~/.chatvcode/models/ 下的模型）
chatvcode index ./my-project

# 指定 GGUF 模型
chatvcode index ./my-project --model=/path/to/model.gguf

# 启用 GPU 加速
chatvcode index ./my-project --model=/path/to/model.gguf --n-gpu-layers=-1

# 调试模式（显示 llama.cpp/ggml 详细日志）
chatvcode index ./my-project --llm-verbose-log

# 第二步：基于代码库问答
chatvcode chat "What does the main function do?" --path=./my-project

# --- 或者使用独立的 GGUF 嵌入模型（推荐用于专用嵌入模型如 Qwen3-Embedding）---
chatvcode index ./my-project --embedding-model=/path/to/embedding.gguf
chatvcode chat "Explain error handling" --path=./my-project \
    --embedding-model=/path/to/embedding.gguf

# --- 或者使用 ONNX 嵌入模型 ---
chatvcode index ./my-project \
    --embedding-model=/path/to/embedding.onnx \
    --embedding-tokenizer=/path/to/tokenizer.json
chatvcode chat "Explain error handling" --path=./my-project \
    --embedding-model=/path/to/embedding.onnx \
    --embedding-tokenizer=/path/to/tokenizer.json
```

> **使用独立嵌入模型**：如果索引时使用了专用嵌入模型（如 Qwen3-Embedding），`chat` 时必须通过 `--embedding-model` 指定同一模型，否则向量维度不匹配会报错。`--model` 用于 LLM 推理，`--embedding-model` 用于嵌入查询，两者可以不同：
> ```bash
> chatvcode chat "question" --path=./my-project \
>     --model=/path/to/llm.gguf \
>     --embedding-model=/path/to/embedding.gguf
> ```

### 模式三：交互式多轮对话（`--interactive`）

在单次命令基础上增加 `--interactive` 标志，进入多轮 REPL 模式。对话历史在多次问答间保留，支持 KV cache 复用加速后续轮次推理。

```bash
# 纯 LLM 交互式对话
chatvcode chat "hello" --retrieval=false --interactive

# RAG 增强交互式对话（需先建索引）
chatvcode chat "hello" --path=./my-project --interactive

# 指定模型和参数
chatvcode chat "hello" --retrieval=false --interactive \
    --model=/path/to/model.gguf \
    --temperature=0.5 \
    --max-tokens=1024
```

进入交互模式后，终端显示 `💬 >` 提示符，直接输入问题即可提问。输入斜杠命令（以 `/` 开头）执行控制操作。

#### 斜杠命令

| 命令              | 缩写       | 说明                                                              |
| ----------------- | ---------- | ----------------------------------------------------------------- |
| `/quit`           | `/q`       | 退出交互式模式                                                    |
| `/help`           | `/h`、`/?` | 显示所有可用命令                                                  |
| `/clear`          | —          | 清空对话历史（保留系统提示）                                      |
| `/sources`        | `/src`     | 显示上一次回答引用的代码来源（文件路径、行号、符号名、相似度分数） |
| `/retry`          | `/r`       | 重新发送上一条问题                                                |
| `/save [path]`    | —          | 将当前会话保存为 JSON 文件（默认路径 `~/.chatvcode/session.json`） |
| `/load [path]`    | —          | 从 JSON 文件恢复会话（默认路径 `~/.chatvcode/session.json`）       |
| `/history`        | —          | 显示对话历史摘要（轮数、预估 token 数、消息预览）                 |
| `/model list`     | `/model ls`| 列出所有可用模型（按优先级扫描本地和全局目录）                    |
| `/model info <path>` | —       | 显示指定模型的详细元数据（架构、参数量、上下文长度等）            |
| `/model switch <path>` | —     | 运行时切换到不同模型（清空对话历史，保留系统提示）                |
| `/model memory <path>` | —     | 估算指定模型的内存使用量（模型权重 + KV cache + 运行时开销）     |
| `/model gpu <path>`    | —     | 根据模型大小推荐 GPU 层卸载数量                                   |

#### 交互模式示例

```
🎤 Interactive chat mode (type `/quit` to exit, `/help` for commands)
📂 Project: ./my-project
   Mode: RAG (with code context)

💬 > What does the main function do?
--- Response ---
The main function initializes the application and starts the event loop.
--- End ---
📎 Sources (2):
  [1] src/main.rs:10 (main) [score: 0.920]
  [2] src/app.rs:42 (run) [score: 0.850]

💬 > /sources
📎 Sources (2):
  [1] src/main.rs:10 (main) [score: 0.920]
  [2] src/app.rs:42 (run) [score: 0.850]

💬 > /model list
Discovered 2 model(s):

[1] qwen2.5-coder-7b.gguf, arch=qwen2, params=7.62B, ctx=32768, quant=Q4_K_M, size=4.4 GB, source=global
    Path: /home/user/.chatvcode/models/qwen2.5-coder-7b.gguf

[2] deepseek-coder-6.7b.gguf, arch=deepseek, params=6.70B, ctx=16384, quant=Q5_K_M, size=4.6 GB, source=local
    Path: ./my-project/.chatvcode/models/deepseek-coder-6.7b.gguf

Use `/model switch <path>` to switch models.

💬 > /model memory /home/user/.chatvcode/models/qwen2.5-coder-7b.gguf
Memory Estimation for /home/user/.chatvcode/models/qwen2.5-coder-7b.gguf
Context size: 8192 tokens

  Model weights : 4.4 GB
  KV cache      : 1.0 GB
  Overhead      : 512.0 MB
  Total (est.)  : 5.9 GB
  Status        : ✓ Expected to fit in available RAM

💬 > /model switch /home/user/.chatvcode/models/qwen2.5-coder-7b.gguf
⚙ Switching to model: /home/user/.chatvcode/models/qwen2.5-coder-7b.gguf
  Memory estimate: 5.9 GB
  Architecture: qwen2
  Context size:  32768
  Parameters:    7.62B
  GPU layers:    0
✓ Model switched successfully.
  Conversation history cleared.

💬 > /save
✓ Session saved to /home/user/.chatvcode/session.json

💬 > /quit
👋 Goodbye!
```

#### 会话持久化

`/save` 和 `/load` 命令支持将对话历史序列化为 JSON 并在后续会话中恢复：

```bash
# 在交互模式中保存到自定义路径
💬 > /save ./my-session.json

# 下次启动时恢复（进入交互模式后执行 /load）
chatvcode chat "hello" --retrieval=false --interactive
💬 > /load ./my-session.json
✓ Session loaded from ./my-session.json (3 turns, ~450 tokens)
```

> 会话 JSON 保存对话消息和系统提示，但不保存 chat 模板和 KV cache 状态。恢复时需使用相同的 `--template` 参数。

#### 快捷键

| 按键    | 说明                              |
| ------- | --------------------------------- |
| `↑`/`↓` | 浏览输入历史                      |
| `Ctrl+C`| 清除当前输入行（不退出交互模式）  |
| `Ctrl+D`| 退出交互式模式（EOF）             |

输入历史自动保存到 `~/.chatvcode/history`，跨会话保留。

## 所有 `chat` 命令参数

| 参数                     | 默认值      | 说明                                                        |
| ------------------------ | ----------- | ----------------------------------------------------------- |
| `<QUESTION>`             | —           | 要提问的问题                                                |
| `-p, --path`             | `.`         | 项目目录路径                                                |
| `-m, --model`            | 自动发现    | GGUF 模型文件路径                                           |
| `-t, --temperature`      | `0.7`       | 生成温度                                                    |
| `--max-tokens`           | `512`       | 最大生成 token 数                                           |
| `--top-k`                | `40`        | Top-k 采样参数                                              |
| `--top-p`                | `0.9`       | Top-p 采样参数                                              |
| `--template`             | `auto`      | Chat 模板：`auto` / `raw` / `chatml` / `llama3` / 自定义    |
| `--system-prompt`        | 内置默认    | 自定义系统提示                                              |
| `--stream`               | `true`      | 启用流式输出（用 `--stream=false` 禁用）                    |
| `--json`                 | 关          | 以 JSON 格式输出结果                                        |
| `--n-ctx`                | `2048`      | 模型上下文窗口大小                                          |
| `--n-threads`            | CPU 核心数  | 推理线程数                                                  |
| `--n-gpu-layers`         | `0`         | GPU 卸载层数（`-1` 表示全部层）                             |
| `--embedding-model`      | —           | 嵌入模型路径（GGUF 或 ONNX，优先于 `--model`）              |
| `--embedding-tokenizer`  | —           | ONNX 嵌入模型分词器路径（GGUF 模型不需要）                  |
| `--embedding-dimension`  | `0`（自动） | 嵌入向量维度                                                |
| `--embedding-max-tokens` | `512`       | 嵌入输入最大 token 数                                       |
| `--top-k-retrieval`      | `16`        | 检索返回的代码片段数                                        |
| `--min-score`            | —           | 最小相似度阈值（0.0–1.0）                                   |
| `--context-token-budget` | `0`（不限） | 分配给上下文的 token 预算                                   |
| `--retrieval`            | `true`      | 启用 RAG 检索（用 `--retrieval=false` 切换为纯 LLM 模式）   |
| `--interactive`          | `false`     | 启用交互式多轮对话 REPL 模式                                |
| `--llm-verbose-log`      | `false`     | 启用 llama.cpp/ggml 详细日志（tensor 创建、backend 注册等） |
| `--config`               | —           | 配置文件路径（默认按优先级搜索：本地 > 全局）               |

> **传参格式**：所有参数的统一传参格式为 `--<arg>=<value>`。布尔参数（`--retrieval`、`--stream`、`--llm-verbose-log` 等）还可使用 `--arg`（等价于 `--arg=true`）的简写形式。

> **配置优先级**：CLI 参数 > 本地配置文件（`<cwd>/.chatvcode/config.json`）> 全局配置文件（`~/.chatvcode/config.json`）> 内置默认值。使用 `--config=<path>` 指定自定义配置文件，或依赖默认的搜索路径。当本地和全局配置文件同时存在时，两者的值会合并，本地文件的值优先。

## 所有 `index` 命令参数

| 参数                      | 默认值                          | 说明                                                        |
| ------------------------- | ------------------------------- | ----------------------------------------------------------- |
| `<PATH>`                  | —                               | 项目目录或源文件路径                                        |
| `--model`                 | 自动发现                        | GGUF 模型文件路径（用于生成嵌入向量）                       |
| `--n-threads`             | CPU 核心数                      | GGUF 嵌入计算线程数                                         |
| `--n-gpu-layers`          | `0`                             | GGUF 嵌入 GPU 卸载层数（`-1` 表示全部层）                   |
| `--embedding-model`       | —                               | 嵌入模型路径（GGUF 或 ONNX，设置后优先使用，忽略 `--model`） |
| `--embedding-tokenizer`   | —                               | ONNX 嵌入模型分词器路径（GGUF 模型不需要）                  |
| `--embedding-dimension`   | `0`（自动）                     | 嵌入向量维度                                                |
| `--embedding-max-tokens`  | `512`                           | 嵌入输入最大 token 数                                       |
| `--embedding-batch-size`  | `32`                            | 嵌入批处理大小                                              |
| `--vector-store-path`     | `<path>/.chatvcode/vectors.vdb` | 向量存储文件路径                                            |
| `--state-file`            | —                               | 增量索引状态文件路径                                        |
| `--large-file-threshold`  | `1048576`                       | 大文件阈值（字节）                                          |
| `--large-file-max-lines`  | `500`                           | 大文件最大读取行数                                          |
| `--chunk-split-threshold` | `3000`                          | 分块字符数阈值（0 禁用）                                    |
| `--llm-verbose-log`       | `false`                         | 启用 llama.cpp/ggml 详细日志（tensor 创建、backend 注册等） |

> **嵌入模型优先级**：`--embedding-model`（GGUF 或 ONNX）> `--model`（GGUF）> 自动发现。`--embedding-model` 根据文件扩展名自动检测格式（`.gguf` → GGUF，其他 → ONNX）。

## 输出说明

### 流式输出（默认）

```
💬 Question: What is Rust?
   Mode: LLM only (no RAG context)

--- Response ---
Rust is a systems programming language focused on safety, speed, and concurrency.
--- End ---

⏱ Time: 5.2s
📊 Tokens: ~42
📚 Mode: LLM only (no RAG context)
```

### 非流式输出（`--stream=false`）

```
4

⏱ Time: 4.8s
📊 Tokens: 78 prompt + 1 completion = 79 total
🛑 Stop reason: Eos
📚 Mode: LLM only (no RAG context)
```

### JSON 输出（`--json`）

```json
{
  "answer": "Rust is a systems programming language.",
  "sources": [],
  "token_usage": {
    "prompt_tokens": 0,
    "completion_tokens": 6,
    "total_tokens": 6
  },
  "stop_reason": "Eos",
  "duration_ms": 120,
  "retrieved_count": 0,
  "used_count": 0,
  "no_context": true
}
```

RAG 模式下，`sources` 字段会包含代码引用来源（文件路径、行号、符号名、相似度分数）。

## `model` 子命令：模型管理

`chatvcode model` 提供模型发现、元数据查看、内存估算、GPU 层推荐和配置文件管理功能。

```bash
# 列出所有可用模型（按优先级扫描本地和全局目录）
chatvcode model list

# 查看模型详细元数据
chatvcode model info /path/to/model.gguf

# 估算模型内存使用量（指定上下文窗口大小）
chatvcode model memory /path/to/model.gguf --n-ctx=8192

# 推荐 GPU 层卸载数量（可选指定 VRAM 大小）
chatvcode model gpu /path/to/model.gguf
chatvcode model gpu /path/to/model.gguf --vram-gb=8

# 查看当前配置（合并配置文件和默认值）
chatvcode model config show

# 创建默认配置文件
chatvcode model config init

# 验证配置文件
chatvcode model config validate
chatvcode model config validate --path=/custom/config.json
```

### 模型搜索优先级

`model list` 和自动发现按以下优先级扫描模型目录：

| 优先级 | 目录                                | 来源标记 | 说明                       |
| ------ | ----------------------------------- | -------- | -------------------------- |
| 1      | `<cwd>/.chatvcode/models/`          | local    | 当前项目的本地模型目录     |
| 2      | `~/.chatvcode/models/`              | global   | 用户级全局模型目录         |

> 本地项目模型优先于全局模型。相同路径的模型不会重复列出。

### 内存估算

`model memory` 根据 GGUF 元数据估算峰值内存使用：

```
Memory Estimation for /home/user/.chatvcode/models/qwen2.5-coder-7b.gguf
Context size: 8192 tokens

  Model weights : 4.4 GB
  KV cache      : 1.0 GB
  Overhead      : 512.0 MB
  Total (est.)  : 5.9 GB
  Status        : ✓ Expected to fit in available RAM
```

- **Model weights**：模型文件大小（GGUF 文件内存映射）
- **KV cache**：`2 × n_head_kv × head_dim × n_ctx × 2 bytes × n_layer`
- **Overhead**：运行时开销（激活值、缓冲区等），启发式估算为模型大小的 10%（256MB–2GB）
- 当预估内存超过可用 RAM 时，会显示警告信息

### GPU 层推荐

`model gpu` 根据模型大小和可用 VRAM 推荐 GPU 层卸载数量：

```
GPU Layer Recommendation for /home/user/.chatvcode/models/qwen2.5-coder-7b.gguf
  Total layers     : 28
  Recommended      : 25 layers
  Est. VRAM usage  : 7.2 GB
  Note             : 25/28 layers fit in available VRAM (8.0 GB)
```

- 推荐值 `-1` 表示所有层均可放入 VRAM
- 自动预留 512MB GPU 开销和 10% 安全余量
- 可使用 `--vram-gb=N` 显式指定可用 VRAM

### 配置文件

配置文件支持本地（项目级）和全局两个层级，按以下优先级加载：

1. **本地配置**：`<cwd>/.chatvcode/config.json`（当前工作目录下的项目级配置）
2. **全局配置**：`~/.chatvcode/config.json`（用户级全局配置）
3. **内置默认值**

当本地和全局配置文件同时存在时，两者的值会合并，**本地文件的值优先**。所有字段均为可选，未设置的字段自动回退到下一级。

配置文件为 JSON 格式，支持三个配置段：

```json
{
  "model": {
    "path": "/home/user/.chatvcode/models/qwen2.5-coder-7b.gguf",
    "n_gpu_layers": 0,
    "n_ctx": 8192,
    "n_threads": 8,
    "template": "auto",
    "use_mmap": true,
    "verbose_log": false
  },
  "generation": {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 40,
    "max_tokens": 2048
  },
  "chat": {
    "system_prompt": "You are a helpful coding assistant.",
    "stream": true,
    "retrieval": true,
    "top_k_retrieval": 16,
    "context_token_budget": 0
  }
}
```

**完整优先级链**：CLI 参数 > 本地配置文件 > 全局配置文件 > 内置默认值。

**配置管理命令**：
- 使用 `chatvcode model config init` 创建默认全局配置文件
- 使用 `chatvcode model config show` 查看合并后的完整配置（显示哪些文件存在及其优先级）
- 使用 `chatvcode model config validate` 验证配置文件格式
- 使用 `--config=<path>` 指定自定义配置文件路径（覆盖默认搜索）

**典型用法**：
- 在 `~/.chatvcode/config.json` 中设置全局偏好（如默认模型路径、GPU 层数）
- 在项目目录的 `.chatvcode/config.json` 中设置项目特定配置（如上下文大小、系统提示）
- 本地配置只需包含需要覆盖的字段，其余字段自动继承全局配置

## 模型自动发现逻辑

当不指定 `--model` 时，CLI 按以下规则自动发现模型：

1. 按优先级扫描模型目录（本地 `.chatvcode/models/` > 全局 `~/.chatvcode/models/`）
2. 验证每个 `.gguf` 文件的魔数和版本
3. 若恰好有一个有效模型 → 自动使用
4. 若没有模型 → 输出友好引导信息，提示如何下载
5. 若有多个模型 → 列出所有模型并选择第一个作为默认（建议使用 `--model` 显式指定）

> 加载前会自动显示内存估算信息。若预估内存可能超出可用 RAM，会显示警告。

## Chat 模板

`--template` 参数控制 prompt 格式化方式：

| 模板       | 说明                                                            |
| ---------- | --------------------------------------------------------------- |
| `auto`     | 自动从 GGUF 元数据推断，推断失败回退到 `chatml`                 |
| `raw`      | 原始文本，不做格式化                                            |
| `chatml`   | ChatML 格式（`<\|im_start\|>` / `<\|im_end\|>`）                |
| `llama3`   | Llama 3 格式（`<\|start_header_id\|>` / `<\|end_header_id\|>`） |
| 其他字符串 | 作为自定义模板传递给 llama.cpp 的 jinja 引擎                    |

大多数主流模型在 `auto` 模式下即可正确工作，无需手动指定。

## 测试与调试

### LLM 详细日志（`--llm-verbose-log`）

当需要排查模型加载问题时，可启用 llama.cpp/ggml 的详细日志输出。默认情况下这些日志（tensor 创建、backend 注册、KV 元数据等）会被静默过滤，开启后会转发到 Rust 的 `log` 输出。

```bash
# chat 命令启用
chatvcode chat "hello" --retrieval=false --llm-verbose-log

# index 命令启用
chatvcode index ./my-project --llm-verbose-log

# 也可使用 --llm-verbose-log=true 的完整写法
chatvcode chat "hello" --retrieval=false --llm-verbose-log=true
```

> 详细日志会产生大量输出（数百行 tensor 创建日志），建议仅在调试时使用。更多信息参见 [chatvcode-llm README](../chatvcode-llm/README.md) 中的日志控制说明。

### Mock LLM 模式

无需真实模型即可测试整个 chat 流程：

```bash
# 使用内置 mock 响应
chatvcode chat "hello" --retrieval=false --mock-llm

# 自定义 mock 响应
chatvcode chat "What is Rust?" --retrieval=false \
    --mock-llm \
    --mock-llm-response="Rust is a systems programming language."
```

### 示例程序

```bash
# Mock LLM 对话示例（无需模型）
cargo run --example basic_chat
```

## 常见问题

### 模型加载失败

```
✗ Failed to load model: ...
```

**建议**：

- 确认文件为有效的 GGUF v2/v3 格式
- 尝试 `--n-gpu-layers=0` 强制纯 CPU 模式
- 尝试降低 `--n-ctx`（默认 2048）
- 确保有足够内存（7B Q4_K_M 模型约需 5GB RAM）

### 没有找到模型

```
✗ Could not auto-discover a model.
```

**解决**：将 GGUF 模型放入 `~/.chatvcode/models/`，或使用 `--model=<path>` 显式指定。

### 无索引文件

```
⚠ No vector store found at ...
```

**解决**：先运行 `chatvcode index <path>` 建索引，或使用 `--retrieval=false` 跳过检索直接对话。
