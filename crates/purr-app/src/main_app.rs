//! Routing: every view renders inside the [`Shell`] layout.

use crate::platform::logging::apply_saved_log_level;
use crate::settings_store::use_settings;
use crate::views::{
    history::History, home::Home, licenses::Licenses, logs::Logs, models::Models,
    not_found::NotFound, shell::Shell, transcription::Transcription,
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
        #[route("/transcription/:id")]
        Transcription { id: HistoryId },
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

impl Route {
    /// Whether this is the address of `live_run`, the run the session holds.
    #[must_use]
    pub fn is_live_run(&self, live_run: Option<&HistoryId>) -> bool {
        match self {
            Self::Transcription { id } => live_run == Some(id),
            Self::Home {}
            | Self::History {}
            | Self::Models {}
            | Self::Licenses {}
            | Self::Logs {}
            | Self::NotFound { .. } => false,
        }
    }
}

impl NavSection {
    /// The section `route` lives in; `None` for pages outside the navigation. A transcription is
    /// part of Transcribe while the session holds it as its run, and of History otherwise.
    #[must_use]
    pub fn of(route: &Route, live_run: Option<&HistoryId>) -> Option<Self> {
        match route {
            Route::Home {} => Some(Self::Transcribe),
            Route::Transcription { id } if live_run == Some(id) => Some(Self::Transcribe),
            Route::History {} | Route::Transcription { .. } => Some(Self::History),
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

    fn id(raw: &str) -> HistoryId {
        HistoryId::parse(raw).unwrap()
    }

    #[test]
    fn detail_routes_light_up_their_section_and_standalone_pages_do_not() {
        assert_eq!(
            NavSection::of(&Route::Home {}, None),
            Some(NavSection::Transcribe)
        );
        assert_eq!(
            NavSection::of(&Route::Models {}, None),
            Some(NavSection::Models)
        );
        assert_eq!(NavSection::of(&Route::Logs {}, None), None);
        assert_eq!(NavSection::of(&Route::Licenses {}, None), None);
        assert_eq!(
            NavSection::of(&Route::NotFound { segments: vec![] }, None),
            None
        );
    }

    #[test]
    fn a_transcription_belongs_to_transcribe_while_it_is_the_live_run_and_to_history_after() {
        let run = id("0d6c-41af");
        let route = Route::Transcription { id: run.clone() };
        assert_eq!(
            NavSection::of(&route, Some(&run)),
            Some(NavSection::Transcribe)
        );
        assert_eq!(NavSection::of(&route, None), Some(NavSection::History));
        assert_eq!(
            NavSection::of(&route, Some(&id("another"))),
            Some(NavSection::History)
        );
    }

    #[test]
    fn only_the_live_runs_own_address_is_the_live_run() {
        let run = id("0d6c-41af");
        assert!(Route::Transcription { id: run.clone() }.is_live_run(Some(&run)));
        assert!(!Route::Transcription { id: run.clone() }.is_live_run(None));
        assert!(!Route::Home {}.is_live_run(Some(&run)));
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
            "/transcription/0d6c-41af".parse::<Route>().ok(),
            Some(Route::Transcription {
                id: id("0d6c-41af")
            })
        );
    }

    #[test]
    fn a_transcription_link_that_is_not_an_id_is_not_found() {
        assert_eq!(
            "/transcription/../x".parse::<Route>().ok(),
            Some(Route::NotFound {
                segments: vec!["transcription".to_owned(), "..".to_owned(), "x".to_owned()]
            })
        );
    }

    #[test]
    fn the_old_history_item_address_is_gone() {
        assert_eq!(
            "/history/0d6c-41af".parse::<Route>().ok(),
            Some(Route::NotFound {
                segments: vec!["history".to_owned(), "0d6c-41af".to_owned()]
            })
        );
    }
}
