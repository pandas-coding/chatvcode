//! Agent 状态机
//!
//! 实现简化的 4 状态设计（`Thinking` ↔ `Acting` → `Done`/`Failed`），
//! 通过 [`TransitionEvent`] 驱动状态转移，并维护当前 [`ThinkingPhase`]。
//!
//! 设计要点：
//! - `ThinkingPhase` 仅在 `Thinking` 状态下有意义，影响 prompt 构建与日志输出
//! - 终态（`Done`/`Failed`）一旦进入不可再转移
//! - 不可恢复事件（`MaxStepsReached`/`Timeout`/`UnrecoverableError`）可从任意非终态触发 `Failed`

use chatvcode_llm::{ToolCall, ToolResult};

use crate::types::{AgentState, ThinkingPhase};

/// 状态转移事件。
///
/// 由 `AgentLoop` 在每一步的推理/执行结束后产生，喂给 [`AgentStateMachine::transition`]。
#[derive(Debug, Clone)]
pub enum TransitionEvent {
    /// LLM 输出中包含工具调用。
    ToolCallsDetected(Vec<ToolCall>),
    /// LLM 给出最终回答（无工具调用）。
    FinalAnswer(String),
    /// 工具执行完成，回到思考阶段以观察结果。
    ToolsExecuted(Vec<ToolResult>),
    /// LLM 输出格式无法解析（保留当前状态，记录错误）。
    ParseError(String),
    /// 达到最大步数限制。
    MaxStepsReached,
    /// 执行超时。
    Timeout,
    /// 不可恢复的错误。
    UnrecoverableError(String),
    /// 强制进入总结阶段（由循环检测触发）。
    ForceConclusion,
    /// 继续当前阶段（软循环仅注入提示，不改状态）。
    Continue,
}

/// Agent 状态机：跟踪当前状态、思考阶段、待执行工具调用与执行结果，并按事件驱动状态转移。
#[derive(Debug, Clone)]
pub struct AgentStateMachine {
    /// 当前 Agent 状态。
    state: AgentState,
    /// 当前思考阶段（仅在 `Thinking` 状态下被解释使用）。
    thinking_phase: ThinkingPhase,
    /// `Thinking -> Acting` 时缓存的待执行工具调用。
    pending_tool_calls: Vec<ToolCall>,
    /// `Acting -> Thinking` 时缓存的最近一轮工具执行结果。
    last_tool_results: Vec<ToolResult>,
    /// `Thinking -> Done` 时缓存的最终回答。
    final_answer: Option<String>,
    /// 最近一次错误信息（解析失败或不可恢复错误）。
    last_error: Option<String>,
    /// 累计解析失败次数。
    parse_error_count: usize,
    /// 进入终态的原因。
    termination_reason: Option<String>,
    /// 已发生的有效状态转移次数（用于观测/调试）。
    transitions: usize,
}

impl Default for AgentStateMachine {
    fn default() -> Self {
        Self::new()
    }
}

impl AgentStateMachine {
    /// 创建初始状态机：状态 `Thinking`，思考阶段 `Planning`。
    pub fn new() -> Self {
        Self {
            state: AgentState::Thinking,
            thinking_phase: ThinkingPhase::Planning,
            pending_tool_calls: Vec::new(),
            last_tool_results: Vec::new(),
            final_answer: None,
            last_error: None,
            parse_error_count: 0,
            termination_reason: None,
            transitions: 0,
        }
    }

    /// 当前状态。
    pub fn state(&self) -> AgentState {
        self.state
    }

    /// 当前思考阶段。
    pub fn thinking_phase(&self) -> ThinkingPhase {
        self.thinking_phase
    }

    /// 待执行的工具调用（只读视图）。
    pub fn pending_tool_calls(&self) -> &[ToolCall] {
        &self.pending_tool_calls
    }

    /// 取走待执行的工具调用，清空缓存。
    pub fn take_pending_tool_calls(&mut self) -> Vec<ToolCall> {
        std::mem::take(&mut self.pending_tool_calls)
    }

    /// 最近一次工具执行结果（只读视图）。
    pub fn last_tool_results(&self) -> &[ToolResult] {
        &self.last_tool_results
    }

    /// 取走最近一次工具执行结果，清空缓存。
    pub fn take_last_tool_results(&mut self) -> Vec<ToolResult> {
        std::mem::take(&mut self.last_tool_results)
    }

    /// 缓存的最终回答。
    pub fn final_answer(&self) -> Option<&str> {
        self.final_answer.as_deref()
    }

    /// 取走最终回答，清空缓存。
    pub fn take_final_answer(&mut self) -> Option<String> {
        self.final_answer.take()
    }

    /// 最近一次错误描述。
    pub fn last_error(&self) -> Option<&str> {
        self.last_error.as_deref()
    }

    /// 累计解析失败次数。
    pub fn parse_error_count(&self) -> usize {
        self.parse_error_count
    }

    /// 已发生的有效状态转移次数。
    pub fn transitions(&self) -> usize {
        self.transitions
    }

    /// 是否处于终态（`Done` 或 `Failed`）。
    pub fn is_terminal(&self) -> bool {
        matches!(self.state, AgentState::Done | AgentState::Failed)
    }

    /// 进入终态的原因（如 `completed`、`max_steps`、`timeout`、错误消息等）。
    pub fn termination_reason(&self) -> Option<&str> {
        self.termination_reason.as_deref()
    }

    /// 设置思考阶段（仅应在 `Thinking` 状态下调用）。
    ///
    /// 在非 `Thinking` 状态下调用会被忽略，以避免破坏状态一致性。
    pub fn set_thinking_phase(&mut self, phase: ThinkingPhase) {
        if self.state == AgentState::Thinking {
            self.thinking_phase = phase;
        }
    }

    /// 应用一个状态转移事件，返回转移后的状态。
    ///
    /// 终态（`Done`/`Failed`）将拒绝任何后续转移并保持原状态。
    pub fn transition(&mut self, event: TransitionEvent) -> AgentState {
        // 终态不可再转移
        if self.is_terminal() {
            return self.state;
        }

        let prev = self.state;
        match (self.state, event) {
            // Thinking -> Acting: 检测到工具调用
            (AgentState::Thinking, TransitionEvent::ToolCallsDetected(calls)) => {
                self.pending_tool_calls = calls;
                if self.pending_tool_calls.is_empty() {
                    // 空调用不应进入 Acting，保持 Thinking
                    self.last_error = Some(
                        "ToolCallsDetected 携带空调用列表，状态保持 Thinking".to_string(),
                    );
                } else {
                    self.state = AgentState::Acting;
                }
            }
            // Thinking -> Done: LLM 给出最终回答
            (AgentState::Thinking, TransitionEvent::FinalAnswer(answer)) => {
                self.final_answer = Some(answer);
                self.state = AgentState::Done;
                self.termination_reason = Some("completed".to_string());
            }
            // Thinking 中解析失败：保留状态，记录错误
            (AgentState::Thinking, TransitionEvent::ParseError(msg)) => {
                self.parse_error_count += 1;
                self.last_error = Some(msg);
            }
            // Thinking 中强制总结：切换思考阶段为 Concluding，状态不变
            (AgentState::Thinking, TransitionEvent::ForceConclusion) => {
                self.thinking_phase = ThinkingPhase::Concluding;
            }
            // Acting -> Thinking: 工具执行完成，进入观察阶段
            (AgentState::Acting, TransitionEvent::ToolsExecuted(results)) => {
                self.last_tool_results = results;
                self.thinking_phase = ThinkingPhase::Observing;
                self.state = AgentState::Thinking;
            }
            // 任意非终态 -> Failed: 达到最大步数
            (_, TransitionEvent::MaxStepsReached) => {
                self.termination_reason = Some("max_steps".to_string());
                self.state = AgentState::Failed;
            }
            // 任意非终态 -> Failed: 超时
            (_, TransitionEvent::Timeout) => {
                self.termination_reason = Some("timeout".to_string());
                self.state = AgentState::Failed;
            }
            // 任意非终态 -> Failed: 不可恢复错误
            (_, TransitionEvent::UnrecoverableError(msg)) => {
                self.last_error = Some(msg.clone());
                self.termination_reason = Some(msg);
                self.state = AgentState::Failed;
            }
            // Continue 与其它未知组合：保持当前状态
            (_, TransitionEvent::Continue) => {}
            (_, _) => {}
        }

        if prev != self.state {
            self.transitions += 1;
        }
        self.state
    }

    /// 重置为初始状态（用于测试或会话复用）。
    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::collections::HashMap;

    fn make_call(name: &str) -> ToolCall {
        let mut args = HashMap::new();
        args.insert("path".to_string(), json!("src/main.rs"));
        ToolCall { name: name.to_string(), arguments: args, id: None }
    }

    fn make_result(success: bool) -> ToolResult {
        ToolResult {
            call_id: None,
            success,
            value: if success { json!("ok") } else { json!("err") },
        }
    }

    #[test]
    fn initial_state_is_thinking_planning() {
        let sm = AgentStateMachine::new();
        assert_eq!(sm.state(), AgentState::Thinking);
        assert_eq!(sm.thinking_phase(), ThinkingPhase::Planning);
        assert!(sm.pending_tool_calls().is_empty());
        assert!(!sm.is_terminal());
        assert_eq!(sm.transitions(), 0);
    }

    #[test]
    fn thinking_to_acting_on_tool_calls() {
        let mut sm = AgentStateMachine::new();
        let calls = vec![make_call("read_file"), make_call("grep_code")];
        let next = sm.transition(TransitionEvent::ToolCallsDetected(calls.clone()));
        assert_eq!(next, AgentState::Acting);
        assert_eq!(sm.state(), AgentState::Acting);
        assert_eq!(sm.pending_tool_calls().len(), 2);
        assert_eq!(sm.transitions(), 1);
    }

    #[test]
    fn thinking_to_done_on_final_answer() {
        let mut sm = AgentStateMachine::new();
        let next = sm.transition(TransitionEvent::FinalAnswer("answer".to_string()));
        assert_eq!(next, AgentState::Done);
        assert_eq!(sm.final_answer(), Some("answer"));
        assert!(sm.is_terminal());
        assert_eq!(sm.termination_reason(), Some("completed"));
    }

    #[test]
    fn acting_to_thinking_on_tools_executed() {
        let mut sm = AgentStateMachine::new();
        sm.transition(TransitionEvent::ToolCallsDetected(vec![make_call("read_file")]));
        assert_eq!(sm.state(), AgentState::Acting);

        let next = sm.transition(TransitionEvent::ToolsExecuted(vec![make_result(true)]));
        assert_eq!(next, AgentState::Thinking);
        assert_eq!(sm.thinking_phase(), ThinkingPhase::Observing);
        assert_eq!(sm.last_tool_results().len(), 1);
    }

    #[test]
    fn full_cycle_thinking_acting_thinking_done() {
        let mut sm = AgentStateMachine::new();
        sm.transition(TransitionEvent::ToolCallsDetected(vec![make_call("read_file")]));
        sm.transition(TransitionEvent::ToolsExecuted(vec![make_result(true)]));
        assert_eq!(sm.state(), AgentState::Thinking);
        assert_eq!(sm.thinking_phase(), ThinkingPhase::Observing);

        sm.set_thinking_phase(ThinkingPhase::Planning);
        assert_eq!(sm.thinking_phase(), ThinkingPhase::Planning);

        sm.transition(TransitionEvent::FinalAnswer("done".to_string()));
        assert_eq!(sm.state(), AgentState::Done);
    }

    #[test]
    fn max_steps_from_any_state_leads_to_failed() {
        for initial in [AgentState::Thinking, AgentState::Acting] {
            let mut sm = AgentStateMachine::new();
            if initial == AgentState::Acting {
                sm.transition(TransitionEvent::ToolCallsDetected(vec![make_call("x")]));
            }
            assert_eq!(sm.state(), initial);
            let next = sm.transition(TransitionEvent::MaxStepsReached);
            assert_eq!(next, AgentState::Failed);
            assert_eq!(sm.termination_reason(), Some("max_steps"));
        }
    }

    #[test]
    fn timeout_from_any_state_leads_to_failed() {
        let mut sm = AgentStateMachine::new();
        assert_eq!(sm.transition(TransitionEvent::Timeout), AgentState::Failed);

        let mut sm = AgentStateMachine::new();
        sm.transition(TransitionEvent::ToolCallsDetected(vec![make_call("x")]));
        assert_eq!(sm.transition(TransitionEvent::Timeout), AgentState::Failed);
    }

    #[test]
    fn unrecoverable_error_stores_message() {
        let mut sm = AgentStateMachine::new();
        let next = sm.transition(TransitionEvent::UnrecoverableError("boom".to_string()));
        assert_eq!(next, AgentState::Failed);
        assert_eq!(sm.last_error(), Some("boom"));
        assert_eq!(sm.termination_reason(), Some("boom"));
    }

    #[test]
    fn parse_error_keeps_thinking_and_counts() {
        let mut sm = AgentStateMachine::new();
        let next = sm.transition(TransitionEvent::ParseError("invalid json".to_string()));
        assert_eq!(next, AgentState::Thinking);
        assert_eq!(sm.parse_error_count(), 1);
        assert_eq!(sm.last_error(), Some("invalid json"));

        sm.transition(TransitionEvent::ParseError("again".to_string()));
        assert_eq!(sm.parse_error_count(), 2);
    }

    #[test]
    fn parse_error_in_acting_is_ignored() {
        let mut sm = AgentStateMachine::new();
        sm.transition(TransitionEvent::ToolCallsDetected(vec![make_call("x")]));
        let next = sm.transition(TransitionEvent::ParseError("oops".to_string()));
        assert_eq!(next, AgentState::Acting);
        assert_eq!(sm.parse_error_count(), 0);
    }

    #[test]
    fn force_conclusion_sets_concluding_phase() {
        let mut sm = AgentStateMachine::new();
        let next = sm.transition(TransitionEvent::ForceConclusion);
        assert_eq!(next, AgentState::Thinking);
        assert_eq!(sm.thinking_phase(), ThinkingPhase::Concluding);
    }

    #[test]
    fn force_conclusion_only_in_thinking() {
        let mut sm = AgentStateMachine::new();
        sm.transition(TransitionEvent::ToolCallsDetected(vec![make_call("x")]));
        let next = sm.transition(TransitionEvent::ForceConclusion);
        // Acting 状态下 ForceConclusion 不匹配任何规则，保持当前状态
        assert_eq!(next, AgentState::Acting);
        assert_eq!(sm.thinking_phase(), ThinkingPhase::Planning);
    }

    #[test]
    fn continue_event_is_noop() {
        let mut sm = AgentStateMachine::new();
        assert_eq!(sm.transition(TransitionEvent::Continue), AgentState::Thinking);
        assert_eq!(sm.transitions(), 0);

        let mut sm = AgentStateMachine::new();
        sm.transition(TransitionEvent::ToolCallsDetected(vec![make_call("x")]));
        assert_eq!(sm.transition(TransitionEvent::Continue), AgentState::Acting);
        assert_eq!(sm.transitions(), 1);
    }

    #[test]
    fn empty_tool_calls_does_not_enter_acting() {
        let mut sm = AgentStateMachine::new();
        let next = sm.transition(TransitionEvent::ToolCallsDetected(vec![]));
        assert_eq!(next, AgentState::Thinking);
        assert!(sm.last_error().is_some());
        assert_eq!(sm.transitions(), 0);
    }

    #[test]
    fn terminal_done_rejects_all_transitions() {
        let mut sm = AgentStateMachine::new();
        sm.transition(TransitionEvent::FinalAnswer("done".to_string()));
        let before = sm.state();
        let next = sm.transition(TransitionEvent::ToolCallsDetected(vec![make_call("x")]));
        assert_eq!(next, AgentState::Done);
        assert_eq!(before, AgentState::Done);
        assert!(sm.pending_tool_calls().is_empty());
        // 之前的转移（1 次）不应变化
        assert_eq!(sm.transitions(), 1);
    }

    #[test]
    fn terminal_failed_rejects_all_transitions() {
        let mut sm = AgentStateMachine::new();
        sm.transition(TransitionEvent::MaxStepsReached);
        let next = sm.transition(TransitionEvent::FinalAnswer("late answer".to_string()));
        assert_eq!(next, AgentState::Failed);
        assert!(sm.final_answer().is_none());
        assert_eq!(sm.termination_reason(), Some("max_steps"));
    }

    #[test]
    fn set_thinking_phase_only_in_thinking_state() {
        let mut sm = AgentStateMachine::new();
        sm.set_thinking_phase(ThinkingPhase::Concluding);
        assert_eq!(sm.thinking_phase(), ThinkingPhase::Concluding);

        // 进入 Acting 后更改思考阶段应被忽略
        sm.transition(TransitionEvent::ToolCallsDetected(vec![make_call("x")]));
        sm.set_thinking_phase(ThinkingPhase::Planning);
        assert_eq!(sm.thinking_phase(), ThinkingPhase::Concluding);
    }

    #[test]
    fn take_methods_drain_caches() {
        let mut sm = AgentStateMachine::new();
        sm.transition(TransitionEvent::ToolCallsDetected(vec![make_call("x")]));
        let drained = sm.take_pending_tool_calls();
        assert_eq!(drained.len(), 1);
        assert!(sm.pending_tool_calls().is_empty());

        sm.transition(TransitionEvent::ToolsExecuted(vec![make_result(true)]));
        let drained = sm.take_last_tool_results();
        assert_eq!(drained.len(), 1);
        assert!(sm.last_tool_results().is_empty());

        sm.transition(TransitionEvent::FinalAnswer("ans".into()));
        assert_eq!(sm.take_final_answer().as_deref(), Some("ans"));
        assert!(sm.final_answer().is_none());
    }

    #[test]
    fn reset_returns_to_initial() {
        let mut sm = AgentStateMachine::new();
        sm.transition(TransitionEvent::ToolCallsDetected(vec![make_call("x")]));
        sm.transition(TransitionEvent::ToolsExecuted(vec![make_result(true)]));
        sm.transition(TransitionEvent::FinalAnswer("done".into()));
        sm.reset();
        assert_eq!(sm.state(), AgentState::Thinking);
        assert_eq!(sm.thinking_phase(), ThinkingPhase::Planning);
        assert_eq!(sm.transitions(), 0);
        assert!(!sm.is_terminal());
    }

    #[test]
    fn all_normal_transition_paths_covered() {
        // Thinking -> Acting
        let mut sm = AgentStateMachine::new();
        assert_eq!(sm.transition(TransitionEvent::ToolCallsDetected(vec![make_call("x")])), AgentState::Acting);
        // Acting -> Thinking
        assert_eq!(sm.transition(TransitionEvent::ToolsExecuted(vec![make_result(true)])), AgentState::Thinking);
        // Thinking -> Done
        assert_eq!(sm.transition(TransitionEvent::FinalAnswer("ans".into())), AgentState::Done);

        // Thinking -> Failed (max steps)
        let mut sm = AgentStateMachine::new();
        assert_eq!(sm.transition(TransitionEvent::MaxStepsReached), AgentState::Failed);

        // Thinking -> Failed (timeout)
        let mut sm = AgentStateMachine::new();
        assert_eq!(sm.transition(TransitionEvent::Timeout), AgentState::Failed);

        // Thinking -> Failed (unrecoverable)
        let mut sm = AgentStateMachine::new();
        assert_eq!(
            sm.transition(TransitionEvent::UnrecoverableError("err".into())),
            AgentState::Failed
        );

        // Acting -> Failed 路径
        let mut sm = AgentStateMachine::new();
        sm.transition(TransitionEvent::ToolCallsDetected(vec![make_call("x")]));
        assert_eq!(sm.transition(TransitionEvent::MaxStepsReached), AgentState::Failed);
    }
}