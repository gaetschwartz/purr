use dioxus::prelude::*;

/// Opens and closes the developer panel from anywhere in the app; provided at the root.
#[derive(Clone, Copy)]
pub struct DeveloperPanelHandle {
    open: Signal<bool>,
}

impl DeveloperPanelHandle {
    pub fn open(mut self) {
        self.open.set(true);
    }

    pub fn close(mut self) {
        self.open.set(false);
    }

    pub fn toggle(mut self) {
        let next = !(self.open)();
        self.open.set(next);
    }

    #[must_use]
    pub fn is_open(self) -> bool {
        (self.open)()
    }
}

/// Provides the [`DeveloperPanelHandle`]; call once at the root.
pub fn use_developer_panel_provider() -> DeveloperPanelHandle {
    use_context_provider(|| DeveloperPanelHandle {
        open: Signal::new(false),
    })
}

pub fn use_developer_panel() -> DeveloperPanelHandle {
    use_context::<DeveloperPanelHandle>()
}
