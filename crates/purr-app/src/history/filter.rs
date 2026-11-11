//! Searching the history by title and transcript text.

use purr_common::history::entry::HistorySummary;

/// The entries whose title or transcript contains `query`, ignoring case; all of them when blank.
#[must_use]
pub fn matching<'a>(summaries: &'a [HistorySummary], query: &str) -> Vec<&'a HistorySummary> {
    let wanted = query.trim().to_lowercase();
    if wanted.is_empty() {
        return summaries.iter().collect();
    }
    summaries
        .iter()
        .filter(|summary| {
            summary.title.to_lowercase().contains(&wanted) || summary.search_text.contains(&wanted)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use purr_common::history::entry::{AudioAvailability, HistoryId, LanguageOrigin};

    fn summary(id: &str, title: &str, text: &str) -> HistorySummary {
        HistorySummary {
            id: HistoryId::parse(id).unwrap(),
            title: title.to_owned(),
            file_name: title.to_owned(),
            created_at_ms: 0,
            audio_seconds: 1.0,
            language: None,
            language_origin: LanguageOrigin::Chosen,
            model_id: "tiny".to_owned(),
            model_label: "Tiny".to_owned(),
            processing_seconds: 1.0,
            preview: String::new(),
            segment_count: 1,
            search_text: text.to_lowercase(),
            audio: AudioAvailability::NotKept,
        }
    }

    fn ids<'a>(found: &[&'a HistorySummary]) -> Vec<&'a str> {
        found.iter().map(|summary| summary.id.as_str()).collect()
    }

    fn sample() -> Vec<HistorySummary> {
        vec![
            summary("a", "Réunion équipe.ogg", "On commence par le budget"),
            summary("b", "memo.m4a", "Bring the signed forms"),
            summary("c", "BUDGET-2026.wav", "Rien à signaler"),
        ]
    }

    #[test]
    fn a_blank_query_keeps_everything_in_order() {
        assert_eq!(ids(&matching(&sample(), "  ")), vec!["a", "b", "c"]);
    }

    #[test]
    fn titles_and_transcripts_are_both_searched() {
        assert_eq!(ids(&matching(&sample(), "budget")), vec!["a", "c"]);
        assert_eq!(ids(&matching(&sample(), "FORMS")), vec!["b"]);
    }

    #[test]
    fn non_ascii_letters_match_by_case() {
        assert_eq!(ids(&matching(&sample(), "ÉQUIPE")), vec!["a"]);
    }

    #[test]
    fn nothing_matches_an_absent_word() {
        assert_eq!(ids(&matching(&sample(), "zebra")), Vec::<&str>::new());
    }
}
