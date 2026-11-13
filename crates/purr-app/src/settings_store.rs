//! The single owner of the user's settings.
//!
//! The store loads the settings once at startup, keeps them in memory, validates every change and
//! is the only code that writes them to storage. Writes go through one queue that collapses a
//! burst of changes into the newest value and never runs two writes at once.
//!
//! ```ignore
//! let settings = use_settings();
//! let model = settings.with(|s| s.model.model_name.clone());
//! settings.set_model("small")?;
//! settings.update(|s| s.model.temperature = 0.2)?;
//! ```

mod clock;
mod flush_on_exit;
mod loaded;
mod save_queue;
pub mod status;
pub mod writer;

use crate::{
    components::toast::{Toast, ToastKind, Toasts},
    platform,
};
use dioxus::prelude::*;
use futures::{
    StreamExt as _,
    channel::mpsc::{UnboundedReceiver, UnboundedSender, unbounded},
};
use loaded::{accept_loaded, apply_change, validated};
use purr_common::{
    platform::Platform as _,
    settings::{LogLevel, Settings, Theme},
};
use save_queue::{Finished, SaveQueue, WriteAccess};
use status::{LoadState, SaveStatus, UpdateError};
use std::{cell::RefCell, rc::Rc};
use writer::FlushOutcome;

/// Handle to the settings store, from [`use_settings`]. Cheap to copy into closures.
#[derive(Clone, Copy, PartialEq)]
pub struct SettingsStore {
    settings: Signal<Settings>,
    status: Signal<SaveStatus>,
    load: Signal<LoadState>,
    queue: CopyValue<Rc<RefCell<SaveQueue>>>,
    wake: CopyValue<UnboundedSender<()>>,
    toasts: Toasts,
}

impl SettingsStore {
    /// A copy of the current settings (subscribes the caller).
    #[must_use]
    pub fn get(&self) -> Settings {
        self.settings.read().clone()
    }

    /// A copy of the current settings, without subscribing the caller.
    #[must_use]
    pub fn peek(&self) -> Settings {
        self.settings.peek().clone()
    }

    /// Reads part of the settings without copying them all (subscribes the caller).
    pub fn with<R>(&self, read: impl FnOnce(&Settings) -> R) -> R {
        read(&self.settings.read())
    }

    /// Applies `change` and saves the result, unless it fails validation; then nothing changes.
    pub fn update(&self, change: impl FnOnce(&mut Settings)) -> Result<(), UpdateError> {
        let next = apply_change(&self.settings.peek(), change)?;
        self.commit(next);
        Ok(())
    }

    /// Replaces every setting at once, for example to undo a reset.
    pub fn replace(&self, settings: Settings) -> Result<(), UpdateError> {
        self.commit(validated(settings)?);
        Ok(())
    }

    /// Light, dark or following the system.
    pub fn set_theme(&self, theme: Theme) -> Result<(), UpdateError> {
        self.update(|settings| settings.ui.theme = theme)
    }

    /// Solid surfaces instead of glass.
    pub fn set_reduce_transparency(&self, reduce: bool) -> Result<(), UpdateError> {
        self.update(|settings| settings.ui.reduce_transparency = reduce)
    }

    /// The model used for new transcriptions, by its id.
    pub fn set_model(&self, model_name: impl Into<String>) -> Result<(), UpdateError> {
        let model_name = model_name.into();
        self.update(|settings| settings.model.model_name = model_name)
    }

    /// The model each launch starts with, or `None` to keep the last one used.
    pub fn set_default_model(&self, model_name: Option<String>) -> Result<(), UpdateError> {
        self.update(|settings| settings.model.default_model = model_name)
    }

    /// The language each launch starts with, or `None` to keep the last one used.
    pub fn set_default_language(&self, code: Option<String>) -> Result<(), UpdateError> {
        self.update(|settings| settings.ui.default_language = code)
    }

    /// The spoken language as a Whisper language code, or `auto` to detect it.
    pub fn set_language(&self, code: impl Into<String>) -> Result<(), UpdateError> {
        let code = code.into();
        self.update(|settings| settings.ui.language = code)
    }

    /// The minimum level the app logs; applied at once unless `RUST_LOG` is set.
    pub fn set_log_level(&self, level: LogLevel) -> Result<(), UpdateError> {
        self.update(|settings| settings.logging.level = level)
    }

    /// Restores the defaults, saves them and returns what they replaced, so a view can offer Undo.
    ///
    /// This is also how the user gets out of a failed load: it is the only change written over
    /// settings that could not be read.
    pub fn reset_to_defaults(&self) -> Settings {
        let mut load = self.load;
        let previous = self.peek();
        if matches!(*load.peek(), LoadState::Failed(_)) {
            load.set(LoadState::Loaded);
        }
        self.queue.read().borrow_mut().set_access(WriteAccess::Open);
        self.commit_always(Settings::default());
        previous
    }

    /// Where the latest change stands on its way to storage (subscribes the caller).
    #[must_use]
    pub fn status(&self) -> SaveStatus {
        self.status.read().clone()
    }

    /// Whether the stored settings were loaded (subscribes the caller).
    #[must_use]
    pub fn load_state(&self) -> LoadState {
        self.load.read().clone()
    }

    /// Tries a failed write again; changes made after a failure are retried on their own.
    pub fn retry_save(&self) {
        let released = self.queue.read().borrow_mut().release_held();
        if released {
            self.wake_writer();
        }
    }

    /// Writes any unsaved change now, after the write in flight; used when the app closes.
    pub async fn flush(&self) -> FlushOutcome {
        let queue = self.shared_queue();
        let outcome = writer::flush(&queue, &save_to_platform, clock::now_millis).await;
        self.refresh_status();
        outcome
    }

    fn commit(&self, next: Settings) {
        if *self.settings.peek() == next {
            return;
        }
        self.commit_always(next);
    }

    fn commit_always(&self, next: Settings) {
        let mut settings = self.settings;
        settings.set(next.clone());
        self.queue.read().borrow_mut().submit(next);
        self.refresh_status();
        self.wake_writer();
    }

    fn wake_writer(&self) {
        if let Err(error) = self.wake.read().unbounded_send(()) {
            tracing::error!("The settings writer has stopped: {error}");
        }
    }

    fn refresh_status(&self) {
        let mut status = self.status;
        let current = self.queue.read().borrow().status();
        if *status.peek() != current {
            status.set(current);
        }
    }

    fn shared_queue(&self) -> Rc<RefCell<SaveQueue>> {
        Rc::clone(&self.queue.read())
    }

    async fn load(self) {
        let mut load = self.load;
        let mut settings = self.settings;
        match accept_loaded(platform::get_platform().load_settings().await) {
            Ok(loaded) => {
                settings.set(loaded);
                self.queue.read().borrow_mut().set_access(WriteAccess::Open);
                load.set(LoadState::Loaded);
            }
            Err(failure) => {
                tracing::error!("Could not load the settings, running on the defaults: {failure}");
                load.set(LoadState::Failed(failure));
            }
        }
        self.refresh_status();
    }

    async fn run_writer(self, mut wakes: UnboundedReceiver<()>) {
        let queue = self.shared_queue();
        while wakes.next().await.is_some() {
            writer::drain(&queue, &save_to_platform, clock::now_millis, |finished| {
                self.refresh_status();
                if let Some(Finished::Failed { error, first: true }) = finished {
                    self.report_failure(error);
                }
            })
            .await;
        }
    }

    fn report_failure(self, error: String) {
        self.toasts.show(
            Toast::new(ToastKind::Error, "Couldn't save the settings")
                .with_detail(error)
                .with_action("Retry", move || self.retry_save()),
        );
    }
}

async fn save_to_platform(settings: Settings) -> Result<(), String> {
    platform::get_platform()
        .save_settings(&settings)
        .await
        .map_err(|error| {
            tracing::error!("Could not save the settings: {error}");
            error.to_string()
        })
}

/// Creates the store at the app root and starts loading the stored settings.
///
/// Views must not render before [`SettingsStore::load_state`] leaves [`LoadState::Loading`].
pub fn use_settings_store_provider(toasts: Toasts) -> SettingsStore {
    let store = use_hook(|| {
        let (store, wakes) = SettingsStore::new(toasts, Settings::default(), LoadState::Loading);
        spawn(store.run_writer(wakes));
        spawn(store.load());
        store
    });
    flush_on_exit::use_flush_on_exit(store);
    use_context_provider(|| store)
}

/// The settings store provided at the root.
pub fn use_settings() -> SettingsStore {
    use_context::<SettingsStore>()
}

impl SettingsStore {
    fn new(toasts: Toasts, settings: Settings, load: LoadState) -> (Self, UnboundedReceiver<()>) {
        let access = match load {
            LoadState::Loaded => WriteAccess::Open,
            LoadState::Loading | LoadState::Failed(_) => WriteAccess::Locked,
        };
        let (wake, wakes) = unbounded();
        let store = Self {
            settings: Signal::new(settings),
            status: Signal::new(SaveStatus::Idle),
            load: Signal::new(load),
            queue: CopyValue::new(Rc::new(RefCell::new(SaveQueue::new(access)))),
            wake: CopyValue::new(wake),
            toasts,
        };
        (store, wakes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::toast::use_toast_provider;
    use purr_common::settings::SettingsError;
    use status::LoadFailure;

    /// Runs `scenario` inside a component, where signals work, with a store that never writes:
    /// the writer is not started, so the wake-ups only pile up in `wakes`.
    fn run_with_store(load: LoadState, scenario: fn(SettingsStore) -> Vec<String>) -> Vec<String> {
        #[derive(Props, Clone)]
        struct ScenarioProps {
            load: LoadState,
            scenario: fn(SettingsStore) -> Vec<String>,
            out: Rc<RefCell<Vec<String>>>,
        }
        impl PartialEq for ScenarioProps {
            fn eq(&self, _: &Self) -> bool {
                false
            }
        }
        #[allow(non_snake_case, reason = "a component")]
        fn Scenario(props: ScenarioProps) -> Element {
            let toasts = use_toast_provider();
            use_hook(|| {
                let (store, wakes) = SettingsStore::new(toasts, Settings::default(), props.load);
                let notes = (props.scenario)(store);
                props.out.borrow_mut().extend(notes);
                drop(wakes);
            });
            rsx! {}
        }

        let out = Rc::new(RefCell::new(Vec::new()));
        let mut dom = VirtualDom::new_with_props(
            Scenario,
            ScenarioProps {
                load,
                scenario,
                out: Rc::clone(&out),
            },
        );
        dom.rebuild_in_place();
        out.take()
    }

    fn describe(store: SettingsStore) -> String {
        format!(
            "{} {:?} {:?}",
            store.peek().model.model_name,
            store.status.peek(),
            store.queue.read().borrow().access()
        )
    }

    #[test]
    fn reset_returns_the_settings_it_replaced() {
        let notes = run_with_store(LoadState::Loaded, |store| {
            let mut notes = vec![format!("{:?}", store.set_model("small"))];
            let previous = store.reset_to_defaults();
            notes.push(previous.model.model_name.clone());
            notes.push(describe(store));
            notes.push(format!("{:?}", store.replace(previous)));
            notes.push(describe(store));
            notes
        });
        assert_eq!(
            notes,
            [
                "Ok(())",
                "small",
                "base Saving Open",
                "Ok(())",
                "small Saving Open"
            ]
        );
    }

    #[test]
    fn an_invalid_change_is_refused_and_nothing_is_queued() {
        let notes = run_with_store(LoadState::Loaded, |store| {
            let refused = store.update(|settings| settings.model.temperature = 3.0);
            vec![
                format!(
                    "{:?}",
                    refused
                        == Err(UpdateError::Invalid(vec![
                            SettingsError::InvalidTemperature { value: 3.0 }
                        ]))
                ),
                describe(store),
            ]
        });
        assert_eq!(notes, ["true", "base Idle Open"]);
    }

    #[test]
    fn after_a_failed_load_changes_stay_in_memory_until_a_reset() {
        let failed = LoadState::Failed(LoadFailure::Unreadable("bad json".to_owned()));
        let notes = run_with_store(failed, |store| {
            let mut notes = vec![format!("{:?}", store.set_model("small"))];
            notes.push(describe(store));
            store.reset_to_defaults();
            notes.push(describe(store));
            notes.push(format!("{:?}", store.load.peek()));
            notes
        });
        assert_eq!(
            notes,
            [
                "Ok(())",
                "small Paused Locked",
                "base Saving Open",
                "Loaded"
            ]
        );
    }
}
