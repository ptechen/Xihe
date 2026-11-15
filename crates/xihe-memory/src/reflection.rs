//! Reflection: distilling a transcript into a summary, and a summary into the
//! next steps.
//!
//! Two separate jobs live here, deliberately unconnected:
//!
//!   * [`Summariser`] compresses dialogue text. The bundled implementation is
//!     deterministic and needs no model, so a deployment without one still
//!     folds long windows; an LLM-backed summariser can be dropped in by the
//!     agent layer, which is where providers already live.
//!   * [`reflect`] turns the *most recent* summary into focus points, then
//!     [`apply_reflection`] ([`crate::components`]) writes them as observations
//!     and derives plan steps from them.
//!
//! The order is set by information dependency: reflection is derived from
//! summarised dialogue, and the plan is derived from reflection — so a summary
//! always feeds the steps, never the other way round.

use async_trait::async_trait;

use crate::error::{MemoryError, MemoryResult};

/// Compresses a transcript into a shorter piece of text.
#[async_trait]
pub trait Summariser: Send + Sync {
    /// Compress a transcript into fewer lines than it started with.
    async fn summarise(&self, transcript: &str) -> MemoryResult<String>;

    /// A stable name for logs and cache keys.
    fn id(&self) -> &str {
        "summariser"
    }
}

/// Picks the sentences with the most salient words and keeps them in order.
///
/// Salience is document frequency: a term used by many lines is common and
/// worth less. Relevance to the *task* is the caller's problem — add a
/// [`MemoryScope::LongTerm`] fact search to the prompt to bias the result.
pub struct HeuristicSummariser {
    max_items: usize,
    item_max_chars: usize,
}

impl HeuristicSummariser {
    /// Keep the best `max_items` lines, each cut to `item_max_chars`.
    pub fn new(max_items: usize, item_max_chars: usize) -> Self {
        Self {
            max_items: max_items.max(1),
            item_max_chars: item_max_chars.max(16),
        }
    }

    /// How many lines one summary may contain.
    pub fn max_items(&self) -> usize {
        self.max_items
    }

    /// Score each line; empty input yields nothing rather than a placeholder.
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
        // Presentation follows the dialogue, not the ranking.
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

/// What a summary suggests the agent should do next.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReflectionStep {
    /// A thing worth optimising — becomes a focus record.
    Focus(String),
    /// A thing that is now known — becomes a long-term fact.
    Knowledge(String),
    /// A caution; a corrective kind also closes the current plan step.
    Note {
        /// What kind of caution this is, e.g. `correction` or `plan`.
        kind: String,
        /// The caution itself.
        text: String,
    },
}

impl ReflectionStep {
    /// A note of the given kind, e.g. `correction` or `plan`.
    pub fn note(kind: impl Into<String>, text: impl Into<String>) -> Self {
        Self::Note {
            kind: kind.into(),
            text: text.into(),
        }
    }
}

/// Read the newest summary and derive what to do from it.
///
/// `tool_activity` and `failures` are counts over the same window the summary
/// covers; `has_plan` avoids stacking a second planning step while one is open.
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

/// Lines that state something durable rather than ask for something.
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

/// Lowercased content words, joined — the form comparisons use.
pub fn normalize(text: &str) -> String {
    tokens(text).join(" ")
}

/// True when either phrase contains the other, or they share most content words.
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
    // Two short phrases about the same thing usually share half their words.
    shared * 2 >= smaller && shared >= 2
}

/// How much of a scope a reflection is allowed to consume.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReflectionBudget {
    /// Maximum focus points kept after a reflection.
    pub max_focus: usize,
    /// Maximum new focus points added by one reflection.
    pub max_new_focus: usize,
    /// Maximum notes added by one reflection.
    pub max_notes: usize,
    /// Maximum facts added by one reflection.
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
    /// Reject a budget that would either stop reflecting or grow without bound.
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
