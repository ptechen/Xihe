//! 反思：把转录蒸馏为摘要，再把摘要蒸馏为下一步。
//!
//! 这里有两个刻意不相连的独立职责：
//!
//!   * [`Summariser`] 压缩对话文本。自带实现是确定性的、
//!     无需模型，因此没有模型的部署仍能折叠长窗口；
//!     由 LLM 支撑的摘要器可由智能体层插入，提供方本就住在那里。
//!   * [`reflect`] 把*最近的*摘要变成关注点，随后
//!     [`apply_reflection`]（[`crate::components`]）把它们写为观察，
//!     并从中推导计划步骤。
//!
//! 顺序由信息依赖决定：反思派生自摘要后的对话，计划派生自反思——
//! 因此摘要总是喂给步骤，绝不反过来。

use async_trait::async_trait;

use crate::error::{MemoryError, MemoryResult};

/// 把转录压缩为更短的一段文本。
#[async_trait]
pub trait Summariser: Send + Sync {
    /// 把转录压缩为比原先更少的行。
    async fn summarise(&self, transcript: &str) -> MemoryResult<String>;

    /// 用于日志与缓存键的稳定名字。
    fn id(&self) -> &str {
        "summariser"
    }
}

/// 挑出含最显著词的句子，并保持其顺序。
///
/// 显著性即文档频率：被许多行使用的词很常见、价值更低。
/// 与*任务*的相关性是调用者的问题——向提示词加入
/// [`MemoryScope::LongTerm`] 事实搜索以偏置结果。
pub struct HeuristicSummariser {
    max_items: usize,
    item_max_chars: usize,
}

impl HeuristicSummariser {
    /// 保留最好的 `max_items` 行，每行截到 `item_max_chars`。
    pub fn new(max_items: usize, item_max_chars: usize) -> Self {
        Self {
            max_items: max_items.max(1),
            item_max_chars: item_max_chars.max(16),
        }
    }

    /// 一个摘要最多可含多少行。
    pub fn max_items(&self) -> usize {
        self.max_items
    }

    /// 给每行打分；空输入产出空而非占位符。
    pub fn rank(&self, transcript: &str) -> Vec<(String, f64)> {
        let lines: Vec<String> = transcript
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .map(str::to_string)
            .collect();
        let mut frequency = std::collections::HashMap::<String, usize>::new();
        for line in &lines {
            let mut seen = std::collections::HashSet::new();
            for token in tokens(line) {
                if seen.insert(token.clone()) {
                    *frequency.entry(token).or_insert(0) += 1;
                }
            }
        }
        let n = lines.len().max(1) as f64;
        let mut scored: Vec<(String, f64)> = lines
            .into_iter()
            .map(|line| {
                let score: f64 = tokens(&line)
                    .iter()
                    .map(|token| {
                        let count = *frequency.get(token).unwrap_or(&1) as f64;
                        (n / count).ln() + 1.0
                    })
                    .sum::<f64>()
                    / (tokens(&line).len().max(1) as f64).sqrt();
                (line, score)
            })
            .collect();
        scored.sort_by(|a, b| b.1.total_cmp(&a.1));
        scored.truncate(self.max_items);
        // 呈现顺序跟随对话，而非排名。
        scored.sort_by(|a, b| {
            let ai = transcript.find(&a.0).unwrap_or(0);
            let bi = transcript.find(&b.0).unwrap_or(0);
            ai.cmp(&bi)
        });
        scored
    }
}

impl Default for HeuristicSummariser {
    fn default() -> Self {
        Self::new(6, 300)
    }
}

#[async_trait]
impl Summariser for HeuristicSummariser {
    async fn summarise(&self, transcript: &str) -> MemoryResult<String> {
        let ranked = self.rank(transcript);
        if ranked.is_empty() {
            return Ok(String::new());
        }
        let mut out = String::new();
        for (line, _) in ranked {
            let trimmed = trim_chars(&line, self.item_max_chars);
            if !out.is_empty() {
                out.push('\n');
            }
            out.push_str("- ");
            out.push_str(&trimmed);
        }
        Ok(out)
    }

    fn id(&self) -> &str {
        "heuristic"
    }
}

fn tokens(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|word| word.chars().count() > 2)
        .map(|word| word.to_lowercase())
        .collect()
}

fn trim_chars(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    let cut = text
        .char_indices()
        .nth(max_chars.saturating_sub(1))
        .map(|(i, _)| i)
        .unwrap_or(text.len());
    format!("{}…", &text[..cut])
}

/// 摘要建议智能体接下来做什么。
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReflectionStep {
    /// 值得优化之事——成为一条关注记录。
    Focus(String),
    /// 现已知晓之事——成为一条长期事实。
    Knowledge(String),
    /// 一条告诫；纠正类还会关闭当前计划步骤。
    Note {
        /// 这是哪类告诫，例如 `correction` 或 `plan`。
        kind: String,
        /// 告诫本身。
        text: String,
    },
}

impl ReflectionStep {
    /// 给定类别的一条注释，例如 `correction` 或 `plan`。
    pub fn note(kind: impl Into<String>, text: impl Into<String>) -> Self {
        Self::Note {
            kind: kind.into(),
            text: text.into(),
        }
    }
}

/// 读取最新摘要并从中推导该做什么。
///
/// `tool_activity` 与 `failures` 是摘要所覆盖同一窗口上的计数；
/// `has_plan` 避免在已有一个计划步骤未结时再叠加第二个。
pub fn reflect(
    summary: &str,
    existing_focus: &[String],
    has_plan: bool,
    tool_activity: usize,
    failures: usize,
) -> Vec<ReflectionStep> {
    let mut steps = Vec::new();
    let existing: Vec<String> = existing_focus
        .iter()
        .map(|text| normalize(text))
        .collect();
    for line in summary
        .lines()
        .map(|line| line.trim().trim_start_matches(['-', '*', '•', ' ']).trim())
        .filter(|line| !line.is_empty())
    {
        let text = trim_chars(line, 240);
        let key = normalize(&text);
        if existing.iter().any(|seen| similar(seen, &key)) {
            continue;
        }
        steps.push(ReflectionStep::Focus(text));
    }
    if tool_activity > 0 && failures > 0 {
        steps.push(ReflectionStep::note(
            "correction",
            format!(
                "{failures} of {tool_activity} tool results failed in the summarised window; \
                 check inputs before retrying."
            ),
        ));
    }
    for sentence in sentences_needing_knowledge(summary) {
        steps.push(ReflectionStep::Knowledge(sentence));
    }
    if has_plan {
        return steps;
    }
    steps.push(if failures > 0 {
        ReflectionStep::note(
            "plan",
            "plan a repair: re-check the failed tool call before planning new work.",
        )
    } else if tool_activity > 0 {
        ReflectionStep::note(
            "plan",
            "plan the next step from the tools used in the summarised window.",
        )
    } else {
        ReflectionStep::note("plan", "plan the next step from the reflected summary.")
    });
    steps
}

/// 陈述持久之事而非请求某事的行。
fn sentences_needing_knowledge(summary: &str) -> Vec<String> {
    summary
        .lines()
        .flat_map(|line| line.split(['.', '!', '?']))
        .map(str::trim)
        .filter(|sentence| {
            let lower = sentence.to_lowercase();
            !sentence.is_empty()
                && sentence.chars().count() > 24
                && !lower.contains("next step")
                && !lower.contains("plan")
                && !lower.contains("should")
                && (lower.contains("because")
                    || lower.contains("requires")
                    || lower.contains("always")
                    || lower.contains("never"))
        })
        .map(|sentence| trim_chars(sentence, 200))
        .collect()
}

/// 小写化的内容词，连接起来——比较所用的形式。
pub fn normalize(text: &str) -> String {
    tokens(text).join(" ")
}

/// 当任一短语包含另一个、或它们共享大部分内容词时为真。
pub fn similar(a: &str, b: &str) -> bool {
    if a.is_empty() || b.is_empty() {
        return false;
    }
    if a.contains(b) || b.contains(a) {
        return true;
    }
    let left: std::collections::HashSet<&str> = a.split(' ').collect();
    let right: std::collections::HashSet<&str> = b.split(' ').collect();
    let smaller = left.len().min(right.len()).max(1);
    let shared = left.intersection(&right).count();
    // 关于同一件事的两个短短语通常共享一半的词。
    shared * 2 >= smaller && shared >= 2
}

/// 一次反思被允许消耗多少额度。
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReflectionBudget {
    /// 一次反思后保留的最大关注点数。
    pub max_focus: usize,
    /// 一次反思新增的最大关注点数。
    pub max_new_focus: usize,
    /// 一次反思新增的最大注释数。
    pub max_notes: usize,
    /// 一次反思新增的最大事实数。
    pub max_knowledge: usize,
}

impl Default for ReflectionBudget {
    fn default() -> Self {
        Self {
            max_focus: 5,
            max_new_focus: 3,
            max_notes: 3,
            max_knowledge: 4,
        }
    }
}

impl ReflectionBudget {
    /// 拒绝会停止反思或无界增长的预算。
    pub fn validate(&self) -> MemoryResult<()> {
        if self.max_focus == 0 || self.max_new_focus == 0 {
            return Err(MemoryError::config(
                "reflection budget needs at least one focus point",
            ));
        }
        if self.max_notes > 64 || self.max_knowledge > 64 {
            return Err(MemoryError::config(
                "reflection budget caps notes and knowledge at 64 per pass",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TRANSCRIPT: &str = "user: the report needs the per-region totals\n\
         assistant: the query aggregates by region and requires a date filter\n\
         user: please use the cached export\n\
         assistant: the cached export always has yesterday's data";

    #[tokio::test]
    async fn summarising_keeps_the_informative_lines() {
        let summary = HeuristicSummariser::default()
            .summarise(TRANSCRIPT)
            .await
            .unwrap();
        assert!(summary.contains("aggregates by region"), "{summary}");
        assert!(summary.starts_with("- "), "{summary}");
        assert!(!summary.is_empty());
    }

    #[tokio::test]
    async fn summarising_empty_transcripts_returns_empty() {
        assert_eq!(
            HeuristicSummariser::default()
                .summarise("   \n  \n")
                .await
                .unwrap(),
            ""
        );
    }

    #[tokio::test]
    async fn long_lines_are_cut_at_a_char_boundary() {
        let long = format!("user: {}", "é".repeat(500));
        let summary = HeuristicSummariser::new(2, 60)
            .summarise(&long)
            .await
            .unwrap();
        assert!(summary.chars().count() < 80, "{summary}");
        assert!(summary.ends_with('…'), "{summary}");
    }

    #[test]
    fn ranked_output_follows_conversation_order() {
        let summariser = HeuristicSummariser::new(3, 200);
        let ranked = summariser.rank(TRANSCRIPT);
        let positions: Vec<usize> = ranked
            .iter()
            .map(|(line, _)| TRANSCRIPT.find(line).unwrap())
            .collect();
        let mut sorted = positions.clone();
        sorted.sort_unstable();
        assert_eq!(positions, sorted);
    }

    #[test]
    fn reflection_derives_focus_and_a_plan() {
        let summary = "- the export always includes yesterday\n- totals require a date filter";
        let steps = reflect(summary, &[], false, 0, 0);
        assert!(
            steps
                .iter()
                .any(|step| matches!(step, ReflectionStep::Focus(text) if text.contains("require"))),
            "{steps:?}"
        );
        assert!(steps.iter().any(|step| matches!(
            step,
            ReflectionStep::Note { kind, .. } if kind == "plan"
        )));
        assert!(
            steps
                .iter()
                .any(|step| matches!(step, ReflectionStep::Knowledge(text) if text.contains("always"))),
            "{steps:?}"
        );
    }

    #[test]
    fn reflection_skips_focus_already_tracked() {
        let summary = "- totals require a date filter";
        let steps = reflect(
            summary,
            &["The totals require a DATE filter.".to_string()],
            true,
            0,
            0,
        );
        assert!(
            !steps
                .iter()
                .any(|step| matches!(step, ReflectionStep::Focus(_))),
            "{steps:?}"
        );
    }

    #[test]
    fn failures_become_a_corrective_note() {
        let steps = reflect("- nothing interesting", &[], true, 4, 2);
        let note = steps
            .iter()
            .find_map(|step| match step {
                ReflectionStep::Note { kind, text } if kind == "correction" => Some(text.clone()),
                _ => None,
            })
            .unwrap();
        assert!(note.contains("2 of 4"), "{note}");
    }

    #[test]
    fn an_open_plan_suppresses_a_second_planning_step() {
        let with_plan = reflect("- a line about results", &[], true, 1, 0);
        let without = reflect("- a line about results", &[], false, 1, 0);
        assert!(
            !with_plan
                .iter()
                .any(|step| matches!(step, ReflectionStep::Note { kind, .. } if kind == "plan")),
            "{with_plan:?}"
        );
        assert!(without.iter().any(|step| matches!(
            step,
            ReflectionStep::Note { kind, .. } if kind == "plan"
        )));
    }

    #[test]
    fn budgets_are_validated() {
        assert!(ReflectionBudget::default().validate().is_ok());
        let zero = ReflectionBudget {
            max_focus: 0,
            ..Default::default()
        };
        assert!(zero.validate().is_err());
        let huge = ReflectionBudget {
            max_knowledge: 100,
            ..Default::default()
        };
        assert!(huge.validate().is_err());
    }

}
