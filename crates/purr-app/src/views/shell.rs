use crate::{
    components::{
        brand_mark::BrandMark,
        icon::{Icon, IconName, IconSize},
    },
    history::store::use_history,
    hooks::use_models::use_models,
    main_app::{NavSection, Route},
    model_catalog::listing::summarize,
    platform::IS_WEB,
    shortcuts::{Key, KeyChord, Shortcut, use_shortcut},
    views::{
        about::AboutSheet, settings_load_banner::SettingsLoadBanner,
        shortcuts_sheet::ShortcutsSheet,
    },
};
use dioxus::prelude::*;
use std::rc::Rc;

struct NavEntry {
    section: NavSection,
    to: Route,
    icon: IconName,
    label: &'static str,
}

const fn entry(section: NavSection) -> NavEntry {
    match section {
        NavSection::Transcribe => NavEntry {
            section,
            to: Route::Home {},
            icon: IconName::Wave,
            label: "Transcribe",
        },
        NavSection::History => NavEntry {
            section,
            to: Route::History {},
            icon: IconName::Clock,
            label: "History",
        },
        NavSection::Models => NavEntry {
            section,
            to: Route::Models {},
            icon: IconName::Cube,
            label: "Models",
        },
        NavSection::Settings => NavEntry {
            section,
            to: Route::Settings {},
            icon: IconName::Sliders,
            label: "Settings",
        },
        NavSection::Logs => NavEntry {
            section,
            to: Route::Logs {},
            icon: IconName::Logs,
            label: "Logs",
        },
    }
}

/// The chord that opens the screen numbered `digit`; browsers keep Ctrl+digit for their own tabs.
const fn screen_chord(web: bool, digit: char) -> KeyChord {
    if web {
        KeyChord::key(Key::Char(digit)).with_alt()
    } else {
        KeyChord::primary(Key::Char(digit))
    }
}

/// The tab bar has no Logs tab; Logs is reached from Settings, so Settings stays lit there.
const fn tab_is_current(tab: NavSection, current: Option<NavSection>) -> bool {
    match (tab, current) {
        (NavSection::Settings, Some(NavSection::Settings | NavSection::Logs))
        | (NavSection::Transcribe, Some(NavSection::Transcribe))
        | (NavSection::History, Some(NavSection::History))
        | (NavSection::Models, Some(NavSection::Models)) => true,
        (
            NavSection::Transcribe
            | NavSection::History
            | NavSection::Models
            | NavSection::Settings
            | NavSection::Logs,
            _,
        ) => false,
    }
}

/// The app frame: glass sidebar (icon rail at middle widths, tab bar on phones) around the views.
#[component]
pub fn Shell() -> Element {
    let route = use_route::<Route>();
    let current = NavSection::of(&route);
    let navigator = use_navigator();
    let mut about_open = use_signal(|| false);
    let mut keys_open = use_signal(|| false);
    let mut main_area = use_signal(|| None::<Rc<MountedData>>);

    use_shortcut(
        Shortcut::new(screen_chord(IS_WEB, '1'), "Go to Transcribe").in_inputs(),
        move || {
            navigator.push(Route::Home {});
        },
    );
    use_shortcut(
        Shortcut::new(screen_chord(IS_WEB, '2'), "Go to History").in_inputs(),
        move || {
            navigator.push(Route::History {});
        },
    );
    use_shortcut(
        Shortcut::new(screen_chord(IS_WEB, '3'), "Go to Models").in_inputs(),
        move || {
            navigator.push(Route::Models {});
        },
    );
    use_shortcut(
        Shortcut::new(KeyChord::primary(Key::Char(',')), "Open Settings").in_inputs(),
        move || {
            navigator.push(Route::Settings {});
        },
    );
    use_shortcut(
        Shortcut::new(KeyChord::key(Key::Char('?')), "Show the keyboard shortcuts"),
        move || keys_open.set(true),
    );

    let skip = move |_| {
        if let Some(main) = main_area() {
            spawn(async move {
                if let Err(error) = main.set_focus(true).await {
                    tracing::debug!("Could not move focus to the content: {error}");
                }
            });
        }
    };

    rsx! {
        div {
            class: "shell",
            ondragover: move |event| event.prevent_default(),
            ondrop: move |event| event.prevent_default(),
            button { class: "skip-link", r#type: "button", onclick: skip, "Skip to content" }
            nav { class: "sidebar glass g1", aria_label: "Purr",
                Link { class: "side-brand", to: Route::Home {}, aria_label: "Purr, Transcribe",
                    BrandMark {}
                    b { "Purr" }
                }
                for section in [NavSection::Transcribe, NavSection::History, NavSection::Models] {
                    NavLink { key: "{section:?}", section, current }
                }
                div { class: "side-foot",
                    NavLink { section: NavSection::Settings, current }
                    NavLink { section: NavSection::Logs, current }
                    button {
                        class: "privacy-pill",
                        r#type: "button",
                        title: "Stays on this device",
                        aria_haspopup: "dialog",
                        onclick: move |_| about_open.set(true),
                        Icon { name: IconName::Lock, size: IconSize::Small }
                        span { class: "nav-text",
                            b { "Stays on this device" }
                            "Audio and transcripts are never uploaded."
                        }
                    }
                }
            }
            div { class: "main",
                main {
                    id: "main-content",
                    class: "scroller",
                    tabindex: "-1",
                    onmounted: move |event: MountedEvent| main_area.set(Some(event.data())),
                    SettingsLoadBanner {}
                    Outlet::<Route> {}
                }
                nav { class: "tabbar glass g1", aria_label: "Purr",
                    for section in [
                        NavSection::Transcribe,
                        NavSection::History,
                        NavSection::Models,
                        NavSection::Settings,
                    ]
                    {
                        TabLink { key: "{section:?}", section, current }
                    }
                }
            }
        }
        if about_open() {
            AboutSheet { on_close: move |()| about_open.set(false) }
        }
        if keys_open() {
            ShortcutsSheet { on_close: move |()| keys_open.set(false) }
        }
    }
}

#[component]
fn NavLink(section: NavSection, current: Option<NavSection>) -> Element {
    let history = use_history();
    let models = use_models();
    let count = match section {
        NavSection::History => history
            .listing()
            .map_or(0, |listing| listing.summaries.len()),
        NavSection::Models => models
            .rows()
            .and_then(Result::ok)
            .map_or(0, |rows| summarize(&rows).installed),
        NavSection::Transcribe | NavSection::Settings | NavSection::Logs => 0,
    };
    let NavEntry {
        section,
        to,
        icon,
        label,
    } = entry(section);
    let here = current == Some(section);
    rsx! {
        Link {
            class: "nav-item",
            to,
            title: label,
            aria_current: here.then_some("page"),
            Icon { name: icon }
            span { class: "nav-text", "{label}" }
            if count > 0 {
                span { class: "count tnum", "{count}" }
            }
        }
    }
}

#[component]
fn TabLink(section: NavSection, current: Option<NavSection>) -> Element {
    let NavEntry {
        to, icon, label, ..
    } = entry(section);
    rsx! {
        Link {
            to,
            aria_current: tab_is_current(section, current).then_some("page"),
            Icon { name: icon }
            "{label}"
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn browsers_get_alt_for_the_screen_shortcuts() {
        assert_eq!(
            screen_chord(true, '2'),
            KeyChord::key(Key::Char('2')).with_alt()
        );
        assert_eq!(screen_chord(false, '2'), KeyChord::primary(Key::Char('2')));
    }

    #[test]
    fn the_settings_tab_stays_lit_on_the_logs_page() {
        assert!(tab_is_current(NavSection::Settings, Some(NavSection::Logs)));
        assert!(!tab_is_current(
            NavSection::Transcribe,
            Some(NavSection::Logs)
        ));
        assert!(!tab_is_current(NavSection::History, None));
    }
}
