# 同步推理引擎测试指南

## 快速验证

```bash
# 运行所有测试（单元测试 + 验收测试）
cargo test -p atlas-llm

# 只运行验收测试
cargo test -p atlas-llm --test inference_acceptance

# 运行单元测试并显示输出
cargo test -p atlas-llm --lib -- --nocapture
```

---

## 验收标准测试

### 标准1: 给定 prompt，可返回完整文本响应

| 测试 | 验证内容 |
|------|---------|
| `criterion_1_returns_complete_text_response` | 响应文本非空且完整 |
| `criterion_1_handles_various_prompts` | 处理各种 prompt（包括空 prompt） |
| `criterion_1_response_not_truncated_when_within_limits` | 未超限时不截断 |

### 标准2: 生成参数可配置且生效

| 测试 | 验证内容 |
|------|---------|
| `criterion_2_temperature_is_configurable` | temperature 参数设置生效 |
| `criterion_2_top_p_is_configurable` | top_p 参数设置生效 |
| `criterion_2_top_k_is_configurable` | top_k 参数设置生效 |
| `criterion_2_max_tokens_limits_generation` | max_tokens 限制生效 |
| `criterion_2_repeat_penalty_is_configurable` | repeat_penalty 参数设置生效 |
| `criterion_2_seed_affects_generation` | seed 参数设置生效 |
| `criterion_2_greedy_params_work` | 贪心解码参数正确 |

### 标准3: 返回结果包含停止原因、token 统计与耗时信息

| 测试 | 验证内容 |
|------|---------|
| `criterion_3_stop_reason_eos` | EOS 停止原因正确 |
| `criterion_3_stop_reason_max_tokens` | MaxTokens 停止原因正确 |
| `criterion_3_stop_reason_cancelled` | Cancelled 停止原因正确 |
| `criterion_3_token_usage_present` | token 统计存在且正确 |
| `criterion_3_timing_information_present` | 耗时统计存在 |
| `criterion_3_response_structure_complete` | 完整响应结构验证 |

---

## 场景测试

| 测试 | 场景描述 |
|------|---------|
| `scenario_coding_question` | 代码生成场景 |
| `scenario_constrained_generation` | 受限生成（低 max_tokens） |
| `scenario_cancelled_mid_generation` | 中途取消生成 |

---

## 集成测试（需要真实模型）

如果有 GGUF 模型文件，可以运行真实推理测试：

```bash
# 运行集成测试（需要模型文件）
cargo test -p atlas-llm --test real_inference -- --ignored
```

### 前置条件

1. 下载 GGUF 模型文件
2. 放置到 `~/.codeatlas/models/` 目录
3. 确保模型文件扩展名为 `.gguf`

### 集成测试内容

| 测试 | 验证内容 |
|------|---------|
| `test_real_model_inference` | 真实模型推理、响应结构、性能统计 |
| `test_real_model_with_different_params` | 不同参数下的推理、贪心解码确定性 |

---

## 测试覆盖率

当前测试覆盖：
- ✅ 43 个单元测试
- ✅ 19 个验收测试
- ✅ 2 个集成测试（需真实模型）

验收标准 100% 覆盖。

---

## 常见问题

### Q: 测试失败怎么办？

1. 检查编译错误：`cargo check -p atlas-llm`
2. 查看详细输出：`cargo test -p atlas-llm -- --nocapture`
3. 运行单个测试：`cargo test -p atlas-llm test_name`

### Q: 如何验证真实模型？

```bash
# 下载小型测试模型（约 4GB）
# 推荐：Qwen2.5-Coder-1.5B-Instruct-GGUF
# 放置到 ~/.codeatlas/models/

# 运行集成测试
cargo test -p atlas-llm --test real_inference -- --ignored --nocapture
```

### Q: 如何查看测试报告？

```bash
# 生成测试报告（需要 nightly）
cargo +nightly test -p atlas-llm -- -Z unstable-options --report-time
```
