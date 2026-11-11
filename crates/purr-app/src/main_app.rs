//! Routing: every view renders inside the [`Shell`] layout.

use crate::platform::logging::apply_saved_log_level;
use crate::settings_store::use_settings;
use crate::views::{
    history::History, home::Home, logs::Logs, models::Models, not_found::NotFound,
    settings::Settings, shell::Shell, transcription::Transcription,
};
use dioxus::prelude::*;
use purr_common::platform::FileId;
use std::collections::HashMap;

/// File names of the uploads made in this session, by upload id.
#[derive(Clone, Copy)]
pub struct UploadNames(pub Signal<HashMap<FileId, String>>);

/// The app's routes.
#[derive(Debug, Clone, Routable, PartialEq, Eq)]
#[rustfmt::skip]
pub enum Route {
    #[layout(Shell)]
        #[route("/")]
        Home {},
        #[route("/transcription/:file")]
        Transcription { file: FileId },
        #[route("/history")]
        History {},
        #[route("/models")]
        Models {},
        #[route("/settings")]
        Settings {},
        #[route("/settings/logs")]
        Logs {},
        #[route("/:..segments")]
        NotFound { segments: Vec<String> },
}

/// The navigation entry a route belongs to, which the sidebar and tab bar mark as current.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NavSection {
    Transcribe,
    History,
    Models,
    Settings,
    Logs,
}

impl NavSection {
    /// The section `route` lives in; `None` for pages outside the navigation.
    #[must_use]
    pub const fn of(route: &Route) -> Option<Self> {
        match route {
            Route::Home {} | Route::Transcription { .. } => Some(Self::Transcribe),
            Route::History {} => Some(Self::History),
            Route::Models {} => Some(Self::Models),
            Route::Settings {} => Some(Self::Settings),
            Route::Logs {} => Some(Self::Logs),
            Route::NotFound { .. } => None,
        }
    }
}

/// Session state shared by the views, and the router.
#[component]
pub fn MainApp() -> Element {
    use_context_provider(|| UploadNames(Signal::new(HashMap::new())));
    let settings = use_settings();
    let log_level = use_memo(move || settings.with(|settings| settings.logging.level.clone()));
    use_effect(move || apply_saved_log_level(&log_level.read()));

    rsx! {
        Router::<Route> {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detail_and_log_routes_light_up_their_section() {
        assert_eq!(
            NavSection::of(&Route::Transcription {
                file: FileId::new("upload".to_owned())
            }),
            Some(NavSection::Transcribe)
        );
        assert_eq!(NavSection::of(&Route::Logs {}), Some(NavSection::Logs));
        assert_eq!(
            NavSection::of(&Route::Settings {}),
            Some(NavSection::Settings)
        );
        assert_eq!(NavSection::of(&Route::NotFound { segments: vec![] }), None);
    }

    #[test]
    fn new_routes_parse_from_deep_links() {
        assert_eq!("/history".parse::<Route>().ok(), Some(Route::History {}));
        assert_eq!("/models".parse::<Route>().ok(), Some(Route::Models {}));
        assert_eq!("/settings/logs".parse::<Route>().ok(), Some(Route::Logs {}));
    }
}
