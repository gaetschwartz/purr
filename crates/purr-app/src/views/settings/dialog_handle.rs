use super::settings_page::SettingsPage;
use crate::platform::IS_WEB;
use dioxus::prelude::*;

/// Opens and closes the Settings dialog from anywhere in the app; provided at the root.
///
/// The dialog stays on the page it was last on while the app runs.
#[derive(Clone, Copy)]
pub struct SettingsDialog {
    open: Signal<bool>,
    page: Signal<SettingsPage>,
}

impl SettingsDialog {
    /// Opens the dialog on `page`, or on the first page where this build has none such.
    pub fn open(mut self, page: SettingsPage) {
        self.page.set(page.or_first(IS_WEB));
        self.open.set(true);
    }

    /// Opens the dialog on the page it was last on.
    pub fn reopen(mut self) {
        self.open.set(true);
    }

    pub fn close(mut self) {
        self.open.set(false);
    }

    pub fn select(mut self, page: SettingsPage) {
        self.page.set(page);
    }

    #[must_use]
    pub fn is_open(self) -> bool {
        (self.open)()
    }

    /// The page shown, always one this build has.
    #[must_use]
    pub fn page(self) -> SettingsPage {
        (self.page)().or_first(IS_WEB)
    }
}

/// Provides the [`SettingsDialog`]; call once at the root.
pub fn use_settings_dialog_provider() -> SettingsDialog {
    use_context_provider(|| SettingsDialog {
        open: Signal::new(false),
        page: Signal::new(SettingsPage::default()),
    })
}

/// The dialog handle provided at the root.
pub fn use_settings_dialog() -> SettingsDialog {
    use_context::<SettingsDialog>()
}
