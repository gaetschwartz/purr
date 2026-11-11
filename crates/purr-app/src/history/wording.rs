//! The words a History row, card or page shows about an entry.

use crate::{transcribe::timing, utils::language_label::language_label};
use purr_common::history::entry::{HistorySummary, LanguageOrigin};

/// The text shown for the transcript of an entry that has no speech.
pub const NO_SPEECH: &str = "No speech was recognised";

/// The spoken language of an entry, naming how it was settled when it was detected.
#[must_use]
pub fn language_text(language: Option<&str>, origin: LanguageOrigin) -> String {
    match (language, origin) {
        (Some(code), LanguageOrigin::Detected) => format!("{} (detected)", language_label(code)),
        (Some(code), LanguageOrigin::Chosen) => language_label(code),
        (None, LanguageOrigin::Detected | LanguageOrigin::Chosen) => {
            "Language not reported".to_owned()
        }
    }
}

/// The preview of an entry, or a note when nothing was said.
#[must_use]
pub fn preview_text(summary: &HistorySummary) -> &str {
    if summary.preview.is_empty() {
        NO_SPEECH
    } else {
        &summary.preview
    }
}

/// When, how long, in which language and with which model: the line under a row.
#[must_use]
pub fn row_meta(when: &str, summary: &HistorySummary) -> String {
    format!(
        "{when} · {} · {} · {}",
        timing::clock(summary.audio_seconds),
        short_language(summary),
        summary.model_label
    )
}

/// When, how long and in which language: the line under a card.
#[must_use]
pub fn card_meta(when: &str, summary: &HistorySummary) -> String {
    format!(
        "{when} · {} · {}",
        timing::clock(summary.audio_seconds),
        short_language(summary)
    )
}

fn short_language(summary: &HistorySummary) -> String {
    summary
        .language
        .as_deref()
        .map_or_else(|| "Language unknown".to_owned(), language_label)
}

/// "1 transcription" or "12 transcriptions".
#[must_use]
pub fn transcriptions(count: usize) -> String {
    if count == 1 {
        "1 transcription".to_owned()
    } else {
        format!("{count} transcriptions")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use purr_common::history::entry::{AudioAvailability, HistoryId};

    fn summary(language: Option<&str>, preview: &str) -> HistorySummary {
        HistorySummary {
            id: HistoryId::parse("a").unwrap(),
            title: "memo.ogg".to_owned(),
            file_name: "memo.ogg".to_owned(),
            created_at_ms: 0,
            audio_seconds: 65.0,
            language: language.map(str::to_owned),
            language_origin: LanguageOrigin::Detected,
            model_id: "tiny".to_owned(),
            model_label: "Tiny".to_owned(),
            processing_seconds: 1.0,
            preview: preview.to_owned(),
            segment_count: 1,
            search_text: String::new(),
            audio: AudioAvailability::NotKept,
        }
    }

    #[test]
    fn a_row_says_when_how_long_which_language_and_model() {
        assert_eq!(
            row_meta("Today, 09:15", &summary(Some("fr"), "Bonjour")),
            "Today, 09:15 · 1:05 · French · Tiny"
        );
    }

    #[test]
    fn a_card_leaves_the_model_out() {
        assert_eq!(
            card_meta("Nov 7, 17:48", &summary(None, "")),
            "Nov 7, 17:48 · 1:05 · Language unknown"
        );
    }

    #[test]
    fn detection_is_named_and_a_choice_is_not() {
        assert_eq!(
            language_text(Some("fr"), LanguageOrigin::Detected),
            "French (detected)"
        );
        assert_eq!(language_text(Some("fr"), LanguageOrigin::Chosen), "French");
        assert_eq!(
            language_text(None, LanguageOrigin::Detected),
            "Language not reported"
        );
    }

    #[test]
    fn an_empty_preview_gets_a_note() {
        assert_eq!(preview_text(&summary(None, "")), NO_SPEECH);
        assert_eq!(preview_text(&summary(None, "Salut")), "Salut");
    }

    #[test]
    fn counts_agree_with_their_noun() {
        assert_eq!(transcriptions(1), "1 transcription");
        assert_eq!(transcriptions(0), "0 transcriptions");
        assert_eq!(transcriptions(12), "12 transcriptions");
    }
}
