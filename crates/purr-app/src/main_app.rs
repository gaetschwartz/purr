//! Routing: every view renders inside the [`Shell`] layout.

use crate::platform::logging::apply_saved_log_level;
use crate::settings_store::use_settings;
use crate::views::{
    history::History, history_item::HistoryItem, home::Home, licenses::Licenses, logs::Logs,
    models::Models, not_found::NotFound, shell::Shell,
};
use dioxus::prelude::*;
use purr_common::history::entry::HistoryId;

/// The app's routes.
#[derive(Debug, Clone, Routable, PartialEq, Eq)]
#[rustfmt::skip]
pub enum Route {
    #[layout(Shell)]
        #[route("/")]
        Home {},
        #[route("/history")]
        History {},
        #[route("/history/:id")]
        HistoryItem { id: HistoryId },
        #[route("/models")]
        Models {},
        #[route("/settings/licenses")]
        Licenses {},
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
}

impl NavSection {
    /// The section `route` lives in; `None` for pages outside the navigation.
    #[must_use]
    pub const fn of(route: &Route) -> Option<Self> {
        match route {
            Route::Home {} => Some(Self::Transcribe),
            Route::History {} | Route::HistoryItem { .. } => Some(Self::History),
            Route::Models {} => Some(Self::Models),
            Route::Licenses {} | Route::Logs {} | Route::NotFound { .. } => None,
        }
    }
}

/// Session state shared by the views, and the router.
#[component]
pub fn MainApp() -> Element {
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
    fn detail_routes_light_up_their_section_and_standalone_pages_do_not() {
        assert_eq!(
            NavSection::of(&Route::Home {}),
            Some(NavSection::Transcribe)
        );
        assert_eq!(NavSection::of(&Route::Models {}), Some(NavSection::Models));
        assert_eq!(NavSection::of(&Route::Logs {}), None);
        let saved = HistoryId::parse("0d6c-41af").unwrap();
        assert_eq!(
            NavSection::of(&Route::HistoryItem { id: saved }),
            Some(NavSection::History)
        );
        assert_eq!(NavSection::of(&Route::Licenses {}), None);
        assert_eq!(NavSection::of(&Route::NotFound { segments: vec![] }), None);
    }

    #[test]
    fn new_routes_parse_from_deep_links() {
        assert_eq!("/history".parse::<Route>().ok(), Some(Route::History {}));
        assert_eq!("/models".parse::<Route>().ok(), Some(Route::Models {}));
        assert_eq!("/settings/logs".parse::<Route>().ok(), Some(Route::Logs {}));
        assert_eq!(
            "/settings/licenses".parse::<Route>().ok(),
            Some(Route::Licenses {})
        );
        assert_eq!(
            "/history/0d6c-41af".parse::<Route>().ok(),
            HistoryId::parse("0d6c-41af")
                .ok()
                .map(|id| Route::HistoryItem { id })
        );
    }

    #[test]
    fn a_history_link_that_is_not_an_id_is_not_found() {
        assert_eq!(
            "/history/../x".parse::<Route>().ok(),
            Some(Route::NotFound {
                segments: vec!["history".to_owned(), "..".to_owned(), "x".to_owned()]
            })
        );
    }
}
