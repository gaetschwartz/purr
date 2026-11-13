/// A page of the Settings dialog.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SettingsPage {
    #[default]
    General,
    Transcription,
    Hardware,
    Advanced,
}

impl SettingsPage {
    const ALL: [Self; 4] = [
        Self::General,
        Self::Transcription,
        Self::Hardware,
        Self::Advanced,
    ];

    /// The pages shown in this build, in list order.
    #[must_use]
    pub fn visible(is_web: bool) -> Vec<Self> {
        Self::ALL
            .into_iter()
            .filter(|page| !(is_web && page.desktop_only()))
            .collect()
    }

    /// `self` if this build shows it, the first page otherwise.
    #[must_use]
    pub fn or_first(self, is_web: bool) -> Self {
        let visible = Self::visible(is_web);
        if visible.contains(&self) {
            self
        } else {
            visible.first().copied().unwrap_or_default()
        }
    }

    const fn desktop_only(self) -> bool {
        match self {
            Self::Hardware => true,
            Self::General | Self::Transcription | Self::Advanced => false,
        }
    }

    /// The name in the page list.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::General => "General",
            Self::Transcription => "Transcription",
            Self::Hardware => "Hardware",
            Self::Advanced => "Advanced",
        }
    }

    /// The DOM id of the page's tab.
    #[must_use]
    pub const fn tab_id(self) -> &'static str {
        match self {
            Self::General => "settings-tab-general",
            Self::Transcription => "settings-tab-transcription",
            Self::Hardware => "settings-tab-hardware",
            Self::Advanced => "settings-tab-advanced",
        }
    }

    /// The page a navigation key leads to from `self` among `pages`.
    #[must_use]
    pub fn step(self, pages: &[Self], step: TabStep) -> Self {
        let Some(index) = pages.iter().position(|page| *page == self) else {
            return self;
        };
        let count = pages.len();
        let target = match step {
            TabStep::Next => (index + 1) % count,
            TabStep::Previous => (index + count - 1) % count,
            TabStep::First => 0,
            TabStep::Last => count - 1,
        };
        pages.get(target).copied().unwrap_or(self)
    }
}

/// Where a navigation key moves the selected page.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TabStep {
    Next,
    Previous,
    First,
    Last,
}

#[cfg(test)]
mod tests {
    use super::SettingsPage::{Advanced, General, Hardware, Transcription};
    use super::*;

    #[test]
    fn the_desktop_lists_every_page_in_order() {
        assert_eq!(
            SettingsPage::visible(false),
            [General, Transcription, Hardware, Advanced]
        );
    }

    #[test]
    fn the_browser_leaves_out_hardware() {
        assert_eq!(
            SettingsPage::visible(true),
            [General, Transcription, Advanced]
        );
    }

    #[test]
    fn a_page_the_build_lacks_falls_back_to_the_first() {
        assert_eq!(Hardware.or_first(true), General);
        assert_eq!(Hardware.or_first(false), Hardware);
        assert_eq!(Advanced.or_first(true), Advanced);
    }

    #[test]
    fn every_page_has_its_own_tab_id() {
        let mut ids: Vec<_> = SettingsPage::ALL.iter().map(|page| page.tab_id()).collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), SettingsPage::ALL.len());
    }

    #[test]
    fn navigation_keys_wrap_around_the_visible_pages() {
        let web = SettingsPage::visible(true);
        assert_eq!(Advanced.step(&web, TabStep::Next), General);
        assert_eq!(General.step(&web, TabStep::Previous), Advanced);
        assert_eq!(General.step(&web, TabStep::Next), Transcription);
        assert_eq!(Advanced.step(&web, TabStep::Previous), Transcription);
        assert_eq!(Transcription.step(&web, TabStep::First), General);
        assert_eq!(Transcription.step(&web, TabStep::Last), Advanced);
        assert_eq!(Hardware.step(&web, TabStep::Next), Hardware);
    }
}
