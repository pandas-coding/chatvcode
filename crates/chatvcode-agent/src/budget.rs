use chatvcode_llm::ChatMessage;

pub trait TokenEstimator: Send + Sync {
    fn estimate_text(&self, text: &str) -> usize;

    fn estimate_messages(&self, messages: &[ChatMessage]) -> usize;
}

pub struct SimpleTokenEstimator;

impl SimpleTokenEstimator {
    pub fn new() -> Self {
        Self
    }
}

impl Default for SimpleTokenEstimator {
    fn default() -> Self {
        Self::new()
    }
}

impl TokenEstimator for SimpleTokenEstimator {
    fn estimate_text(&self, text: &str) -> usize {
        if text.is_empty() {
            return 0;
        }
        (text.len() + 3) / 4
    }

    fn estimate_messages(&self, messages: &[ChatMessage]) -> usize {
        const OVERHEAD_PER_MSG: usize = 4;
        messages
            .iter()
            .map(|m| self.estimate_text(&m.content) + OVERHEAD_PER_MSG)
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimate_text_empty() {
        let est = SimpleTokenEstimator::new();
        assert_eq!(est.estimate_text(""), 0);
    }

    #[test]
    fn estimate_text_short() {
        let est = SimpleTokenEstimator::new();
        assert_eq!(est.estimate_text("hi"), 1);
    }

    #[test]
    fn estimate_text_exact_multiple_of_4() {
        let est = SimpleTokenEstimator::new();
        assert_eq!(est.estimate_text("hello!"), 2);
        assert_eq!(est.estimate_text("abcdefgh"), 2);
    }

    #[test]
    fn estimate_text_rounds_up() {
        let est = SimpleTokenEstimator::new();
        assert_eq!(est.estimate_text("a"), 1);
        assert_eq!(est.estimate_text("abc"), 1);
        assert_eq!(est.estimate_text("abcde"), 2);
    }

    #[test]
    fn estimate_messages_empty() {
        let est = SimpleTokenEstimator::new();
        assert_eq!(est.estimate_messages(&[]), 0);
    }

    #[test]
    fn estimate_messages_single() {
        let est = SimpleTokenEstimator::new();
        let msgs = vec![ChatMessage::user("hello world!")];
        let tokens = est.estimate_messages(&msgs);
        assert_eq!(tokens, est.estimate_text("hello world!") + 4);
    }

    #[test]
    fn estimate_messages_multiple() {
        let est = SimpleTokenEstimator::new();
        let msgs = vec![
            ChatMessage::system("You are a helpful assistant."),
            ChatMessage::user("Hi"),
            ChatMessage::assistant("Hello! How can I help?"),
        ];
        let tokens = est.estimate_messages(&msgs);
        let expected: usize = msgs
            .iter()
            .map(|m| est.estimate_text(&m.content) + 4)
            .sum();
        assert_eq!(tokens, expected);
    }

    #[test]
    fn simple_token_estimator_default() {
        let est = SimpleTokenEstimator::default();
        assert_eq!(est.estimate_text("test"), 1);
    }
}
