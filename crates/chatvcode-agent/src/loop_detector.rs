use std::collections::HashMap;

use chatvcode_llm::ToolCall;
use serde_json::Value;

use crate::types::AgentStep;

/// 循环检测结果
#[derive(Debug, Clone, PartialEq)]
pub enum LoopDetectionResult {
    /// 未检测到循环
    NoLoop,
    /// 软循环：相同的工具调用模式，但参数略有不同
    SimilarLoop { pattern: String, occurrences: usize },
    /// 硬循环：完全相同的工具调用（名称 + 参数）
    ExactLoop { tool_name: String, occurrences: usize },
}

/// Agent 循环检测器，分析最近若干步内的工具调用模式，区分硬循环与软循环。
pub struct LoopDetector {
    /// 滑动窗口大小（考察的最近步数）
    window_size: usize,
    /// 参数相似度阈值（0.0 ~ 1.0）
    similarity_threshold: f64,
    /// 触发软循环所需的最少相似调用次数
    min_similar_occurrences: usize,
}

impl Default for LoopDetector {
    fn default() -> Self {
        Self { window_size: 6, similarity_threshold: 0.8, min_similar_occurrences: 3 }
    }
}

impl LoopDetector {
    pub fn new() -> Self {
        Self::default()
    }

    /// 自定义参数构造，用于测试或调优。
    pub fn with_params(
        window_size: usize,
        similarity_threshold: f64,
        min_similar_occurrences: usize,
    ) -> Self {
        Self {
            window_size: window_size.max(1),
            similarity_threshold: similarity_threshold.clamp(0.0, 1.0),
            min_similar_occurrences: min_similar_occurrences.max(2),
        }
    }

    pub fn window_size(&self) -> usize {
        self.window_size
    }

    pub fn similarity_threshold(&self) -> f64 {
        self.similarity_threshold
    }

    /// 检测最近的工具调用是否构成循环。
    pub fn detect(&self, steps: &[AgentStep]) -> LoopDetectionResult {
        if steps.len() < self.window_size {
            return LoopDetectionResult::NoLoop;
        }

        let recent_calls: Vec<&ToolCall> = steps
            .iter()
            .rev()
            .take(self.window_size)
            .flat_map(|s| s.tool_calls.iter())
            .collect();

        if recent_calls.is_empty() {
            return LoopDetectionResult::NoLoop;
        }

        if let Some(exact) = self.detect_exact_loop(&recent_calls) {
            return exact;
        }

        if let Some(similar) = self.detect_similar_loop(&recent_calls) {
            return similar;
        }

        LoopDetectionResult::NoLoop
    }

    /// 检测完全相同的工具调用（名称 + 参数）。
    fn detect_exact_loop(&self, calls: &[&ToolCall]) -> Option<LoopDetectionResult> {
        let mut call_counts: HashMap<String, usize> = HashMap::new();

        for call in calls {
            let key = canonical_call_key(call);
            *call_counts.entry(key).or_default() += 1;
        }

        let mut best: Option<(String, usize)> = None;
        for (key, count) in call_counts {
            if count >= 2 {
                match &best {
                    Some((_, c)) if *c >= count => {}
                    _ => best = Some((key, count)),
                }
            }
        }

        best.map(|(key, count)| {
            let tool_name = key.split('\u{0}').next().unwrap_or("unknown").to_string();
            LoopDetectionResult::ExactLoop { tool_name, occurrences: count }
        })
    }

    /// 检测软循环：相同工具的参数彼此高度相似。
    fn detect_similar_loop(&self, calls: &[&ToolCall]) -> Option<LoopDetectionResult> {
        let mut by_tool: HashMap<&str, Vec<String>> = HashMap::new();
        for call in calls {
            by_tool
                .entry(call.name.as_str())
                .or_default()
                .push(canonical_args(&call.arguments));
        }

        let mut best: Option<(String, usize)> = None;
        for (name, args_list) in by_tool {
            if args_list.len() >= self.min_similar_occurrences {
                let similarities = compute_pairwise_similarity(&args_list);
                if similarities.is_empty() {
                    continue;
                }
                let avg = similarities.iter().sum::<f64>() / similarities.len() as f64;
                if avg >= self.similarity_threshold {
                    match &best {
                        Some((_, c)) if *c >= args_list.len() => {}
                        _ => best = Some((name.to_string(), args_list.len())),
                    }
                }
            }
        }

        best.map(|(pattern, occurrences)| LoopDetectionResult::SimilarLoop { pattern, occurrences })
    }
}

/// 构造工具调用的规范化唯一键（工具名 + 规范化参数），用于精确匹配。
fn canonical_call_key(call: &ToolCall) -> String {
    let args = canonical_args(&call.arguments);
    format!("{}\u{0}{}", call.name, args)
}

/// 将参数映射规范化为确定顺序的 JSON 字符串，避免 HashMap 迭代顺序导致的不稳定。
fn canonical_args(args: &HashMap<String, Value>) -> String {
    let mut entries: Vec<(String, Value)> =
        args.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
    entries.sort_by(|a, b| a.0.cmp(&b.0));
    let mut map = serde_json::Map::new();
    for (k, v) in entries {
        map.insert(k, v);
    }
    Value::Object(map).to_string()
}

/// 计算参数列表中两两之间的相似度（基于 JSON 字符串的归一化编辑距离）。
fn compute_pairwise_similarity(args: &[String]) -> Vec<f64> {
    let mut similarities = Vec::new();
    for i in 0..args.len() {
        for j in (i + 1)..args.len() {
            similarities.push(json_similarity(&args[i], &args[j]));
        }
    }
    similarities
}

/// 基于归一化 Levenshtein 距离计算两个 JSON 字符串的相似度（0.0 ~ 1.0）。
fn json_similarity(a: &str, b: &str) -> f64 {
    let max_len = a.len().max(b.len());
    if max_len == 0 {
        return 1.0;
    }
    let dist = levenshtein(a.as_bytes(), b.as_bytes());
    1.0 - (dist as f64 / max_len as f64)
}

/// Levenshtein 编辑距离，动态规划实现，空间复杂度 O(min(m, n))。
fn levenshtein(a: &[u8], b: &[u8]) -> usize {
    let (a, b) = if a.len() <= b.len() { (a, b) } else { (b, a) };
    let m = a.len();
    let n = b.len();

    if m == 0 {
        return n;
    }

    let mut prev = (0..=n).collect::<Vec<usize>>();
    let mut curr = vec![0usize; n + 1];

    for i in 1..=m {
        curr[0] = i;
        for j in 1..=n {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            curr[j] = (prev[j] + 1).min(curr[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    prev[n]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{AgentState, AgentStep, ThinkingPhase, TokenUsage};
    use chatvcode_llm::ToolCall;
    use serde_json::json;
    use std::collections::HashMap;

    fn make_step(tool_calls: Vec<ToolCall>) -> AgentStep {
        AgentStep {
            step_number: 0,
            state: AgentState::Acting,
            thinking_phase: Some(ThinkingPhase::Observing),
            thought: None,
            tool_calls,
            tool_results: vec![],
            duration_ms: 0,
            token_usage: TokenUsage::default(),
        }
    }

    fn make_call(name: &str, args: &[(&str, Value)]) -> ToolCall {
        let mut map = HashMap::new();
        for (k, v) in args {
            map.insert((*k).to_string(), v.clone());
        }
        ToolCall { name: name.to_string(), arguments: map, id: None }
    }

    #[test]
    fn no_loop_without_window() {
        let detector = LoopDetector::new();
        let steps = (0..3)
            .map(|_| make_step(vec![make_call("read_file", &[("path", json!("a.rs"))])]))
            .collect::<Vec<_>>();
        assert_eq!(detector.detect(&steps), LoopDetectionResult::NoLoop);
    }

    #[test]
    fn no_loop_with_varied_tools() {
        let detector = LoopDetector::new();
        let tools = [
            "read_file",
            "list_files",
            "grep_code",
            "search_code",
            "search_symbol",
            "get_file_structure",
        ];
        let steps = tools
            .iter()
            .map(|t| make_step(vec![make_call(t, &[])]))
            .collect::<Vec<_>>();
        assert_eq!(detector.detect(&steps), LoopDetectionResult::NoLoop);
    }

    #[test]
    fn detects_exact_loop() {
        let detector = LoopDetector::new();
        let call_factory = || make_step(vec![make_call("read_file", &[("path", json!("foo.rs"))])]);
        let steps = (0..6).map(|_| call_factory()).collect::<Vec<_>>();
        match detector.detect(&steps) {
            LoopDetectionResult::ExactLoop { tool_name, occurrences } => {
                assert_eq!(tool_name, "read_file");
                assert!(occurrences >= 2);
            }
            other => panic!("expected ExactLoop, got {:?}", other),
        }
    }

    #[test]
    fn exact_loop_distinct_args_not_flagged_as_exact() {
        let detector = LoopDetector::with_params(6, 0.8, 3);
        let paths = [
            "src/main.rs",
            "docs/architecture.md",
            "crates/core/lib.rs",
            "tests/integration.rs",
            "config/settings.toml",
            "scripts/build.sh",
        ];
        let steps = paths
            .iter()
            .map(|p| make_step(vec![make_call("read_file", &[("path", json!(p))])]))
            .collect::<Vec<_>>();
        // 6 个完全不同的参数既不构成 ExactLoop，也不构成 SimilarLoop
        match detector.detect(&steps) {
            LoopDetectionResult::NoLoop => {}
            other => panic!("expected NoLoop, got {:?}", other),
        }
    }

    #[test]
    fn detects_similar_loop() {
        let detector = LoopDetector::with_params(6, 0.5, 3);
        // 六个查询互不相同（避免 ExactLoop），但彼此高度相似
        let queries = [
            "search for foo implementation",
            "search for foo definition",
            "search for foo usage",
            "search for foo reference",
            "search for foo declaration",
            "search for foo signature",
        ];
        let steps = queries
            .iter()
            .map(|q| make_step(vec![make_call("search_code", &[("query", json!(q))])]))
            .collect::<Vec<_>>();
        match detector.detect(&steps) {
            LoopDetectionResult::SimilarLoop { pattern, occurrences } => {
                assert_eq!(pattern, "search_code");
                assert!(occurrences >= 3);
            }
            other => panic!("expected SimilarLoop, got {:?}", other),
        }
    }

    #[test]
    fn empty_steps_no_loop() {
        let detector = LoopDetector::new();
        assert_eq!(detector.detect(&[]), LoopDetectionResult::NoLoop);
    }

    #[test]
    fn steps_without_tool_calls_no_loop() {
        let detector = LoopDetector::new();
        let steps = (0..6).map(|_| make_step(vec![])).collect::<Vec<_>>();
        assert_eq!(detector.detect(&steps), LoopDetectionResult::NoLoop);
    }

    #[test]
    fn exact_loop_takes_priority_over_similar() {
        // 全部完全相同，应优先返回 ExactLoop
        let detector = LoopDetector::with_params(6, 0.1, 3);
        let step = make_step(vec![make_call("grep_code", &[("pattern", json!("fn"))])]);
        let steps = (0..6).map(|_| step.clone()).collect::<Vec<_>>();
        match detector.detect(&steps) {
            LoopDetectionResult::ExactLoop { tool_name, .. } => {
                assert_eq!(tool_name, "grep_code");
            }
            other => panic!("expected ExactLoop priority, got {:?}", other),
        }
    }

    #[test]
    fn canonical_args_is_stable() {
        let mut a = HashMap::new();
        a.insert("b".to_string(), json!(2));
        a.insert("a".to_string(), json!(1));

        let mut b = HashMap::new();
        b.insert("a".to_string(), json!(1));
        b.insert("b".to_string(), json!(2));

        assert_eq!(canonical_args(&a), canonical_args(&b));
    }

    #[test]
    fn levenshtein_basic() {
        assert_eq!(levenshtein(b"kitten", b"sitting"), 3);
        assert_eq!(levenshtein(b"", b"abc"), 3);
        assert_eq!(levenshtein(b"abc", b"abc"), 0);
    }

    #[test]
    fn json_similarity_identical_is_one() {
        assert!((json_similarity("{\"a\":1}", "{\"a\":1}") - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn default_params_sensible() {
        let d = LoopDetector::default();
        assert_eq!(d.window_size(), 6);
        assert!((d.similarity_threshold() - 0.8).abs() < f64::EPSILON);
    }
}
