/// A group of settings the jump links lead to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SettingsSection {
    Appearance,
    Transcription,
    Audio,
    Performance,
    Logging,
    About,
}

impl SettingsSection {
    const ALL: [Self; 6] = [
        Self::Appearance,
        Self::Transcription,
        Self::Audio,
        Self::Performance,
        Self::Logging,
        Self::About,
    ];

    /// The sections shown in this build, in page order.
    #[must_use]
    pub fn visible(is_web: bool) -> Vec<Self> {
        Self::ALL
            .into_iter()
            .filter(|section| !(is_web && section.desktop_only()))
            .collect()
    }

    const fn desktop_only(self) -> bool {
        match self {
            Self::Audio | Self::Performance => true,
            Self::Appearance | Self::Transcription | Self::Logging | Self::About => false,
        }
    }

    /// The title of the section and of its jump link.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Appearance => "Appearance",
            Self::Transcription => "Transcription",
            Self::Audio => "Audio",
            Self::Performance => "Performance",
            Self::Logging => "Logging",
            Self::About => "About",
        }
    }

    /// The DOM id of the section.
    #[must_use]
    pub const fn anchor(self) -> &'static str {
        match self {
            Self::Appearance => "s-appearance",
            Self::Transcription => "s-transcription",
            Self::Audio => "s-audio",
            Self::Performance => "s-performance",
            Self::Logging => "s-logging",
            Self::About => "s-about",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::SettingsSection::{About, Appearance, Audio, Logging, Performance, Transcription};
    use super::*;

    #[test]
    fn the_desktop_shows_every_section_in_page_order() {
        assert_eq!(
            SettingsSection::visible(false),
            [
                Appearance,
                Transcription,
                Audio,
                Performance,
                Logging,
                About
            ]
        );
    }

    #[test]
    fn the_browser_leaves_out_audio_and_performance() {
        assert_eq!(
            SettingsSection::visible(true),
            [Appearance, Transcription, Logging, About]
        );
    }

    #[test]
    fn every_section_has_its_own_anchor() {
        let mut anchors: Vec<_> = SettingsSection::ALL.iter().map(|s| s.anchor()).collect();
        anchors.sort_unstable();
        anchors.dedup();
        assert_eq!(anchors.len(), SettingsSection::ALL.len());
    }
}
