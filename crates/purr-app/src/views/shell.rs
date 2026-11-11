use crate::{
    components::{
        brand_mark::BrandMark,
        icon::{Icon, IconName, IconSize},
    },
    main_app::{NavSection, Route},
    views::about::AboutSheet,
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
    let mut about_open = use_signal(|| false);
    let mut main_area = use_signal(|| None::<Rc<MountedData>>);

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
    }
}

#[component]
fn NavLink(section: NavSection, current: Option<NavSection>) -> Element {
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
    fn the_settings_tab_stays_lit_on_the_logs_page() {
        assert!(tab_is_current(NavSection::Settings, Some(NavSection::Logs)));
        assert!(!tab_is_current(
            NavSection::Transcribe,
            Some(NavSection::Logs)
        ));
        assert!(!tab_is_current(NavSection::History, None));
    }
}
