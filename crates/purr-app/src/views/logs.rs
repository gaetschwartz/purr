//! The Logs view: the entries captured in memory, filtered, searched, exported or cleared.

mod announce;
mod clear_button;
mod count;
mod export_menu;
mod filter;
mod follow;
mod list;
mod model;
mod row;
mod toolbar;
mod window;

use crate::{
    components::page::Page,
    main_app::Route,
    platform::{
        get_platform,
        logging::{applied_log_level, env_overrides_log_level, get_logs_storage},
    },
    utils::delay::sleep,
};
use announce::{Announcer, POLL_MS};
use clear_button::ClearButton;
use dioxus::prelude::*;
use export_menu::ExportMenu;
use filter::LevelFilter;
use follow::{DISPOSE_WATCHER, FollowMode, ScrollEvent, WATCHER};
use list::LogList;
use model::LogsModel;
use purr_common::{platform::Platform as _, settings::LogLevel};
use std::{cell::Cell, rc::Rc, time::Duration};
use toolbar::LogToolbar;

const fn level_name(level: &LogLevel) -> &'static str {
    match level {
        LogLevel::Error => "Error",
        LogLevel::Warn => "Warning",
        LogLevel::Info => "Info",
        LogLevel::Debug => "Debug",
        LogLevel::Trace => "Trace",
    }
}

const FOCUS_LIST: &str =
    "requestAnimationFrame(() => document.getElementById('loglist')?.focus());";

fn recorded_at(level: &LogLevel) -> String {
    format!(
        "Recorded at {} level. Logs stay on this device.",
        level_name(level)
    )
}

struct Tracker {
    seen: Cell<u64>,
    announcer: Cell<Announcer>,
}

/// The page that lists, filters, exports and clears the captured log.
#[component]
pub fn Logs() -> Element {
    let storage = get_logs_storage();
    let tracker = use_hook(|| {
        Rc::new(Tracker {
            seen: Cell::new(storage.revision()),
            announcer: Cell::new(Announcer::default()),
        })
    });
    let mut model = use_signal(|| LogsModel::new(Rc::new(storage.snapshot())));
    let mut follow = use_signal(FollowMode::default);
    let mut announcement = use_signal(String::new);
    let settings = use_resource(|| async { get_platform().load_settings().await });
    let watcher = use_hook(|| document::eval(WATCHER));

    let sync = use_callback({
        let tracker = Rc::clone(&tracker);
        move |()| {
            let storage = get_logs_storage();
            let revision = storage.revision();
            if revision == tracker.seen.get() {
                return 0;
            }
            tracker.seen.set(revision);
            model.write().refresh(Rc::new(storage.snapshot()))
        }
    });

    let set_filter = use_callback({
        let tracker = Rc::clone(&tracker);
        move |(level, query): (LevelFilter, String)| {
            model.write().set_filter(level, query);
            tracker.announcer.set(Announcer::default());
            let model = model.peek();
            announcement.set(format!(
                "Showing {} of {}",
                model.matching().len(),
                count::entries(model.total())
            ));
        }
    });

    use_future({
        let tracker = Rc::clone(&tracker);
        move || {
            let tracker = Rc::clone(&tracker);
            async move {
                loop {
                    sleep(Duration::from_millis(POLL_MS)).await;
                    let fresh = sync.call(());
                    let mut announcer = tracker.announcer.get();
                    let message = announcer.tick(fresh);
                    tracker.announcer.set(announcer);
                    if let Some(message) = message {
                        announcement.set(message);
                    }
                }
            }
        }
    });

    use_effect(move || {
        let mode = follow();
        model.write().set_following(mode.is_following());
        if let Err(error) = watcher.send(mode) {
            tracing::warn!("Could not tell the page whether to follow the log: {error}");
        }
    });

    use_future(move || async move {
        let mut watcher = watcher;
        loop {
            match watcher.recv::<ScrollEvent>().await {
                Ok(event) => {
                    let mode = follow.peek().after(event);
                    follow.set(mode);
                }
                Err(error) => {
                    tracing::warn!("The log scroll watcher stopped: {error}");
                    break;
                }
            }
        }
    });

    use_drop(|| {
        document::eval(DISPOSE_WATCHER);
    });

    let saved_level = match &*settings.read() {
        Some(Ok(settings)) => Some(settings.logging.level.clone()),
        Some(Err(_)) | None => None,
    };
    let subtitle = match (applied_log_level(), env_overrides_log_level(), saved_level) {
        (Some(level), _, _) | (None, false, Some(level)) => recorded_at(&level),
        (None, true, _) => {
            "Recorded at the level RUST_LOG sets. Logs stay on this device.".to_owned()
        }
        (None, false, None) => "Logs stay on this device.".to_owned(),
    };
    let settings_error = match &*settings.read() {
        Some(Err(error)) => Some(error.to_string()),
        Some(Ok(_)) | None => None,
    };

    let model_now = model.read();
    let matching = Rc::clone(model_now.matching());
    let total = model_now.total();
    let summary = format!("Showing {} of {}.", matching.len(), count::entries(total));

    rsx! {
        Page {
            title: "Logs",
            subtitle,
            wide: true,
            actions: rsx! {
                ExportMenu { entries: Rc::clone(&matching) }
                ClearButton {
                    total,
                    on_cleared: move |()| {
                        sync.call(());
                        tracker.announcer.set(Announcer::default());
                        document::eval(FOCUS_LIST);
                    },
                }
            },
            section { class: "card glass tight", aria_label: "Log entries",
                LogToolbar {
                    level: model_now.level(),
                    query: model_now.query().to_owned(),
                    follow: follow(),
                    on_level: move |level| {
                        let query = model.peek().query().to_owned();
                        set_filter.call((level, query));
                    },
                    on_query: move |query: String| {
                        let level = model.peek().level();
                        set_filter.call((level, query));
                    },
                    on_follow: move |on| follow.set(FollowMode::switched(on)),
                }
                LogList {
                    entries: matching,
                    rows: model_now.rows(),
                    recorded: total > 0,
                    on_earlier: move |()| model.write().show_earlier(),
                    on_reset_filter: move |()| set_filter.call((LevelFilter::All, String::new())),
                }
            }
            p { class: "foot", style: "padding-left:16px",
                "{summary} File names appear in logs; transcript text never does. "
                Link { to: Route::Settings {}, "Change the log level in Settings" }
                "."
            }
            if let Some(error) = settings_error {
                p { class: "foot", style: "padding-left:16px",
                    "Couldn't read the log level setting: {error}"
                }
            }
            div {
                class: "sr",
                role: "status",
                aria_live: "polite",
                aria_atomic: "true",
                "{announcement}"
            }
        }
    }
}
