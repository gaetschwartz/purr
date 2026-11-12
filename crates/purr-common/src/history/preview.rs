//! The short texts derived from a transcript: its preview and its search text.

use crate::platform::TranscriptSegment;

/// The longest preview, in characters, ellipsis included.
pub const PREVIEW_CHARS: usize = 160;

fn words(segments: &[TranscriptSegment]) -> impl Iterator<Item = &str> {
    segments
        .iter()
        .flat_map(|segment| segment.text.split_whitespace())
}

/// The first words of the transcript on one line, cut at [`PREVIEW_CHARS`].
#[must_use]
pub fn preview(segments: &[TranscriptSegment]) -> String {
    let mut text = String::new();
    for word in words(segments) {
        if !text.is_empty() {
            text.push(' ');
        }
        text.push_str(word);
        if text.chars().count() > PREVIEW_CHARS {
            break;
        }
    }
    if text.chars().count() <= PREVIEW_CHARS {
        return text;
    }
    let mut cut: String = text.chars().take(PREVIEW_CHARS - 1).collect();
    cut.truncate(cut.trim_end().len());
    cut.push('…');
    cut
}

/// The whole transcript in lower case on one line, which a search compares against.
#[must_use]
pub fn search_text(segments: &[TranscriptSegment]) -> String {
    words(segments).collect::<Vec<_>>().join(" ").to_lowercase()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn segment(text: &str) -> TranscriptSegment {
        TranscriptSegment {
            text: text.to_owned(),
            start_time: 0.0,
            end_time: 1.0,
            words: Vec::new(),
        }
    }

    #[test]
    fn a_short_transcript_is_its_own_preview() {
        let segments = [segment(" Bonjour  tout"), segment("le monde\n")];
        assert_eq!(preview(&segments), "Bonjour tout le monde");
    }

    #[test]
    fn no_speech_gives_an_empty_preview() {
        assert_eq!(preview(&[]), "");
        assert_eq!(preview(&[segment("   ")]), "");
    }

    #[test]
    fn a_long_transcript_is_cut_with_an_ellipsis() {
        let segments = [segment(&"mot ".repeat(100))];
        let cut = preview(&segments);
        assert_eq!(cut.chars().count(), PREVIEW_CHARS);
        assert_eq!(cut.chars().last(), Some('…'));
    }

    #[test]
    fn a_cut_never_splits_a_character() {
        let segments = [segment(&"é".repeat(PREVIEW_CHARS + 20))];
        let cut = preview(&segments);
        assert_eq!(cut.chars().count(), PREVIEW_CHARS);
        assert_eq!(
            cut.chars().filter(|letter| *letter == 'é').count(),
            PREVIEW_CHARS - 1
        );
    }

    #[test]
    fn search_text_is_lower_case_on_one_line() {
        let segments = [segment("Été  Chaud"), segment("Ça va")];
        assert_eq!(search_text(&segments), "été chaud ça va");
    }
}
