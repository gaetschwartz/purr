//! Writing the last change before the window or the page goes away.

use super::SettingsStore;
#[cfg(any(purr_web, feature = "desktop"))]
use super::writer::FlushOutcome;

#[cfg(any(purr_web, feature = "desktop"))]
fn log_outcome(outcome: &FlushOutcome) {
    match outcome {
        FlushOutcome::NothingToSave => tracing::debug!("No unsaved settings when closing"),
        FlushOutcome::Saved => tracing::info!("Saved the last settings change before closing"),
        FlushOutcome::Failed(error) => {
            tracing::error!("Could not save the last settings change before closing: {error}");
        }
        FlushOutcome::Locked => tracing::warn!(
            "Closing without saving: the stored settings were unreadable and were not reset"
        ),
        FlushOutcome::Busy => {
            tracing::warn!("Closing while a settings write was still running");
        }
    }
}

/// Hides the window on close, writes what is unsaved, then closes it for real.
#[cfg(all(feature = "desktop", not(purr_web)))]
pub fn use_flush_on_exit(store: SettingsStore) {
    use dioxus::desktop::{
        WindowCloseBehaviour, WindowEvent, tao::event::Event, use_window, use_wry_event_handler,
    };
    use dioxus::prelude::*;
    use std::{rc::Rc, time::Duration};

    const FLUSH_LIMIT: Duration = Duration::from_secs(5);

    let window = use_window();
    use_hook({
        let window = Rc::clone(&window);
        move || window.set_close_behavior(WindowCloseBehaviour::WindowHides)
    });
    use_wry_event_handler(move |event, _| {
        if let Event::WindowEvent {
            event: WindowEvent::CloseRequested,
            ..
        } = event
        {
            let window = Rc::clone(&window);
            spawn(async move {
                match tokio::time::timeout(FLUSH_LIMIT, store.flush()).await {
                    Ok(outcome) => log_outcome(&outcome),
                    Err(elapsed) => {
                        tracing::error!("Gave up saving the settings before closing: {elapsed}");
                    }
                }
                window.set_close_behavior(WindowCloseBehaviour::WindowCloses);
                window.close();
            });
        }
    });
}

/// Writes what is unsaved when the page is hidden, which is the last moment a tab is sure to run.
#[cfg(purr_web)]
pub fn use_flush_on_exit(store: SettingsStore) {
    use super::{clock, save_to_platform, writer};
    use dioxus::core::use_hook_with_cleanup;
    use std::rc::Rc;
    use wasm_bindgen::{JsCast as _, closure::Closure};

    use_hook_with_cleanup(
        move || {
            let queue = store.shared_queue();
            let on_pagehide = Closure::<dyn FnMut()>::new(move || {
                log_outcome(&writer::flush_now(
                    &queue,
                    &save_to_platform,
                    clock::now_millis,
                ));
            });
            if let Some(window) = web_sys::window() {
                window.set_onpagehide(Some(on_pagehide.as_ref().unchecked_ref()));
            } else {
                tracing::warn!("No window: settings are not flushed when the page closes");
            }
            Rc::new(on_pagehide)
        },
        |on_pagehide| {
            if let Some(window) = web_sys::window() {
                window.set_onpagehide(None);
            }
            drop(on_pagehide);
        },
    );
}

#[cfg(not(any(purr_web, all(feature = "desktop", not(purr_web)))))]
pub const fn use_flush_on_exit(_store: SettingsStore) {}
