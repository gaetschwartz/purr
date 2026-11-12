//! How the transcript is shown, remembered between transcriptions.

use serde::{Deserialize, Serialize};

/// How the transcript is laid out.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum TranscriptLayout {
    /// One row per segment, with its time.
    #[default]
    Segments,
    /// Segments gathered into paragraphs.
    Reading,
}

/// The share of the run's words, in percent, that "mark unsure words" underlines.
///
/// A multiple of 5 from 5 to 30; any other number read from a file snaps to the nearest one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(from = "u8", into = "u8")]
pub struct UnsureShare(u8);

impl UnsureShare {
    pub const MIN: u8 = 5;
    pub const MAX: u8 = 30;
    pub const STEP: u8 = 5;

    /// The share closest to `percent` that the setting allows.
    #[must_use]
    pub fn new(percent: u8) -> Self {
        let steps = (u16::from(percent) + u16::from(Self::STEP) / 2) / u16::from(Self::STEP);
        let snapped = steps.saturating_mul(u16::from(Self::STEP));
        let allowed = snapped.clamp(u16::from(Self::MIN), u16::from(Self::MAX));
        Self(u8::try_from(allowed).unwrap_or(Self::MAX))
    }

    #[must_use]
    pub const fn percent(self) -> u8 {
        self.0
    }
}

impl Default for UnsureShare {
    fn default() -> Self {
        Self(10)
    }
}

impl From<u8> for UnsureShare {
    fn from(percent: u8) -> Self {
        Self::new(percent)
    }
}

impl From<UnsureShare> for u8 {
    fn from(share: UnsureShare) -> Self {
        share.0
    }
}

const fn shown() -> bool {
    true
}

/// How the transcript card shows the transcript.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct TranscriptView {
    pub layout: TranscriptLayout,
    /// Show the time of each segment
    pub show_times: bool,
    /// Underline the words the recogniser was least sure of
    pub mark_unsure: bool,
    pub unsure_share: UnsureShare,
    /// Let a click on a word play the audio from there
    pub word_seek: bool,
    /// Show the decoder's competing hypotheses while a transcription runs
    pub show_beams: bool,
}

impl Default for TranscriptView {
    fn default() -> Self {
        Self {
            layout: TranscriptLayout::default(),
            show_times: shown(),
            mark_unsure: false,
            unsure_share: UnsureShare::default(),
            word_seek: false,
            show_beams: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_share_snaps_to_the_nearest_allowed_step() {
        let snapped = |percent| UnsureShare::new(percent).percent();
        assert_eq!(snapped(0), 5);
        assert_eq!(snapped(2), 5);
        assert_eq!(snapped(7), 5);
        assert_eq!(snapped(8), 10);
        assert_eq!(snapped(10), 10);
        assert_eq!(snapped(27), 25);
        assert_eq!(snapped(28), 30);
        assert_eq!(snapped(31), 30);
        assert_eq!(snapped(255), 30);
    }

    #[test]
    fn the_defaults_show_times_and_mark_nothing() {
        let view = TranscriptView::default();
        assert_eq!(view.layout, TranscriptLayout::Segments);
        assert!(view.show_times);
        assert!(!view.mark_unsure);
        assert_eq!(view.unsure_share.percent(), 10);
        assert!(!view.word_seek);
        assert!(!view.show_beams);
    }

    #[test]
    fn an_out_of_range_share_in_a_file_loads_snapped() {
        let view: TranscriptView =
            toml::from_str("unsure_share = 99\nlayout = \"Reading\"").unwrap();
        assert_eq!(view.unsure_share.percent(), 30);
        assert_eq!(view.layout, TranscriptLayout::Reading);
        assert!(view.show_times);
    }

    #[test]
    fn a_view_survives_a_save_and_a_load() {
        let view = TranscriptView {
            layout: TranscriptLayout::Reading,
            show_times: false,
            mark_unsure: true,
            unsure_share: UnsureShare::new(20),
            word_seek: true,
            show_beams: true,
        };
        let saved = toml::to_string_pretty(&view).unwrap();
        assert_eq!(toml::from_str::<TranscriptView>(&saved).unwrap(), view);
    }
}
