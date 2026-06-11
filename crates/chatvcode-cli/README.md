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

# 第二步：基于代码库问答
chatvcode chat "What does the main function do?" --path=./my-project

# --- 或者使用 ONNX 嵌入模型 ---
chatvcode index ./my-project \
    --embedding-model=/path/to/embedding.onnx \
    --embedding-tokenizer=/path/to/tokenizer.json
chatvcode chat "Explain error handling" --path=./my-project \
    --embedding-model=/path/to/embedding.onnx \
    --embedding-tokenizer=/path/to/tokenizer.json
```

## 所有 `chat` 命令参数

| 参数                     | 默认值      | 说明                                                      |
| ------------------------ | ----------- | --------------------------------------------------------- |
| `<QUESTION>`             | —           | 要提问的问题                                              |
| `-p, --path`             | `.`         | 项目目录路径                                              |
| `-m, --model`            | 自动发现    | GGUF 模型文件路径                                         |
| `-t, --temperature`      | `0.7`       | 生成温度                                                  |
| `--max-tokens`           | `512`       | 最大生成 token 数                                         |
| `--top-k`                | `40`        | Top-k 采样参数                                            |
| `--top-p`                | `0.9`       | Top-p 采样参数                                            |
| `--template`             | `auto`      | Chat 模板：`auto` / `raw` / `chatml` / `llama3` / 自定义  |
| `--system-prompt`        | 内置默认    | 自定义系统提示                                            |
| `--stream`               | `true`      | 启用流式输出（用 `--stream=false` 禁用）                  |
| `--json`                 | 关          | 以 JSON 格式输出结果                                      |
| `--n-ctx`                | `2048`      | 模型上下文窗口大小                                        |
| `--n-threads`            | CPU 核心数  | 推理线程数                                                |
| `--n-gpu-layers`         | `0`         | GPU 卸载层数（`-1` 表示全部层）                           |
| `--embedding-model`      | —           | ONNX 嵌入模型路径（RAG 模式需要）                         |
| `--embedding-tokenizer`  | —           | 嵌入模型分词器路径                                        |
| `--embedding-dimension`  | `0`（自动） | 嵌入向量维度                                              |
| `--embedding-max-tokens` | `512`       | 嵌入输入最大 token 数                                     |
| `--top-k-retrieval`      | `8`         | 检索返回的代码片段数                                      |
| `--min-score`            | —           | 最小相似度阈值（0.0–1.0）                                 |
| `--context-token-budget` | `0`（不限） | 分配给上下文的 token 预算                                 |
| `--retrieval`            | `true`      | 启用 RAG 检索（用 `--retrieval=false` 切换为纯 LLM 模式） |

> **传参格式**：所有参数的统一传参格式为 `--<arg>=<value>`。布尔参数（`--retrieval`、`--stream` 等）还可使用 `--arg`（等价于 `--arg=true`）的简写形式。

## 所有 `index` 命令参数

| 参数                      | 默认值                          | 说明                                                     |
| ------------------------- | ------------------------------- | -------------------------------------------------------- |
| `<PATH>`                  | —                               | 项目目录或源文件路径                                     |
| `--model`                 | 自动发现                        | GGUF 模型文件路径（用于生成嵌入向量）                    |
| `--n-threads`             | CPU 核心数                      | GGUF 嵌入计算线程数                                      |
| `--n-gpu-layers`          | `0`                             | GGUF 嵌入 GPU 卸载层数（`-1` 表示全部层）                |
| `--embedding-model`       | —                               | ONNX 嵌入模型路径（设置后优先使用 ONNX，忽略 `--model`） |
| `--embedding-tokenizer`   | —                               | ONNX 嵌入模型分词器路径                                  |
| `--embedding-dimension`   | `0`（自动）                     | 嵌入向量维度                                             |
| `--embedding-max-tokens`  | `512`                           | 嵌入输入最大 token 数                                    |
| `--embedding-batch-size`  | `32`                            | 嵌入批处理大小                                           |
| `--vector-store-path`     | `<path>/.chatvcode/vectors.vdb` | 向量存储文件路径                                         |
| `--state-file`            | —                               | 增量索引状态文件路径                                     |
| `--large-file-threshold`  | `1048576`                       | 大文件阈值（字节）                                       |
| `--large-file-max-lines`  | `500`                           | 大文件最大读取行数                                       |
| `--chunk-split-threshold` | `3000`                          | 分块字符数阈值（0 禁用）                                 |

> **嵌入模型优先级**：`--embedding-model`（ONNX）> `--model`（GGUF）> 自动发现。两者都设置时优先使用 ONNX。

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

## 模型自动发现逻辑

当不指定 `--model` 时，CLI 按以下规则自动发现模型：

1. 扫描 `~/.chatvcode/models/` 目录
2. 验证每个 `.gguf` 文件的魔数和版本
3. 若恰好有一个有效模型 → 自动使用
4. 若没有模型 → 输出友好引导信息，提示如何下载
5. 若有多个模型 → 报错并列出所有模型，要求用 `--model` 指定

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
