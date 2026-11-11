//! The saved transcriptions, kept at the root so every screen and toast action can reach them.

use crate::{
    components::{
        icon::IconName,
        toast::{Toast, ToastKind, Toasts, use_toasts},
    },
    platform::{self, Platform as _, PlatformError},
};
use dioxus::{
    core::{Runtime, ScopeId, spawn_forever},
    prelude::*,
};
use purr_common::history::{
    entry::{HistoryEntry, HistoryId},
    listing::{HistoryListing, Saved},
};
use std::rc::Rc;

/// What the history list shows.
#[derive(Debug, Clone, PartialEq)]
pub enum HistoryLoad {
    Loading,
    Ready(Rc<HistoryListing>),
    /// The platform could not list the history.
    Failed(String),
}

/// Handle to the saved transcriptions, from [`use_history`]. Cheap to copy; every method may be
/// called from any scope, including overlays and toast actions.
#[derive(Clone, Copy, PartialEq)]
pub struct HistoryStore {
    load: Signal<HistoryLoad>,
    toasts: Toasts,
}

impl HistoryStore {
    /// The list as last loaded (subscribes the caller).
    #[must_use]
    pub fn load(&self) -> HistoryLoad {
        self.load.read().clone()
    }

    /// The readable entries, newest first, once loaded (subscribes the caller).
    #[must_use]
    pub fn listing(&self) -> Option<Rc<HistoryListing>> {
        match &*self.load.read() {
            HistoryLoad::Ready(listing) => Some(Rc::clone(listing)),
            HistoryLoad::Loading | HistoryLoad::Failed(_) => None,
        }
    }

    /// Loads the list again from storage.
    pub fn reload(mut self) {
        spawn_forever(async move {
            let next = match platform::get_platform().list_history().await {
                Ok(listing) => HistoryLoad::Ready(Rc::new(listing)),
                Err(error) => {
                    tracing::error!("Could not list the history: {error}");
                    HistoryLoad::Failed(error.to_string())
                }
            };
            self.load.set(next);
        });
    }

    fn patch(mut self, change: impl FnOnce(&HistoryListing) -> HistoryListing) {
        let current = self.listing();
        if let Some(listing) = current {
            self.load.set(HistoryLoad::Ready(Rc::new(change(&listing))));
        }
    }

    /// Stores `entry`, then refreshes the list; the entries dropped to stay within the retention
    /// limits are reported in the result.
    pub async fn save(self, entry: &HistoryEntry) -> Result<Saved, PlatformError> {
        let saved = platform::get_platform().save_history_entry(entry).await?;
        self.reload();
        Ok(saved)
    }

    /// Gives the entry `id` the title `title`, which must already be a valid one.
    pub fn rename(self, id: HistoryId, title: String) {
        spawn_forever(async move {
            match platform::get_platform()
                .rename_history_entry(&id, &title)
                .await
            {
                Ok(()) => {
                    self.patch(|listing| listing.retitled(&id, &title));
                    self.toasts.success("Renamed", title);
                }
                Err(error) => {
                    tracing::error!("Could not rename {id}: {error}");
                    self.toasts
                        .error("Couldn't rename the transcription", error.to_string());
                    self.reload();
                }
            }
        });
    }

    /// Deletes the entry `id` and offers Undo, which puts it back as it was.
    pub fn delete(self, id: HistoryId, title: String) {
        spawn_forever(async move {
            let platform = platform::get_platform();
            let record = match platform.get_history_entry(&id).await {
                Ok(record) => record,
                Err(error) => {
                    tracing::error!("Could not read {id} before deleting it: {error}");
                    self.toasts
                        .error("Couldn't delete the transcription", error.to_string());
                    self.reload();
                    return;
                }
            };
            if let Err(error) = platform.delete_history_entry(&id).await {
                tracing::error!("Could not delete {id}: {error}");
                self.toasts
                    .error("Couldn't delete the transcription", error.to_string());
                self.reload();
                return;
            }
            self.patch(|listing| listing.removed(&id));
            let entry = Rc::new(record.entry);
            self.toasts.show(
                Toast::new(ToastKind::Info, "Transcription deleted")
                    .with_detail(title)
                    .with_icon(IconName::Trash)
                    .with_action("Undo", move || self.restore(Rc::clone(&entry))),
            );
        });
    }

    fn restore(self, entry: Rc<HistoryEntry>) {
        spawn_forever(async move {
            match self.save(&entry).await {
                Ok(saved) if saved.evicted.contains(&entry.id) => {
                    self.toasts.warning(
                        "Couldn't restore the transcription",
                        "History is full of newer transcriptions, so it was dropped again.",
                    );
                }
                Ok(_) => {
                    self.toasts
                        .success("Transcription restored", entry.title.clone());
                }
                Err(error) => {
                    tracing::error!("Could not restore {}: {error}", entry.id);
                    self.toasts
                        .error("Couldn't restore the transcription", error.to_string());
                }
            }
        });
    }

    /// Deletes every saved transcription, including unreadable ones.
    pub fn clear(self) {
        spawn_forever(async move {
            match platform::get_platform().clear_history().await {
                Ok(removed) => {
                    self.patch(|_| HistoryListing::default());
                    self.toasts.success(
                        "History cleared",
                        format!("{removed} stored {} removed.", noun(removed)),
                    );
                }
                Err(error) => {
                    tracing::error!("Could not clear the history: {error}");
                    self.toasts
                        .error("Couldn't clear the history", error.to_string());
                    self.reload();
                }
            }
        });
    }
}

fn noun(count: usize) -> &'static str {
    if count == 1 { "item" } else { "items" }
}

/// Creates the store at the app root and loads the list; call once, after the toast provider.
pub fn use_history_provider() -> HistoryStore {
    let toasts = use_toasts();
    let store = use_hook(|| {
        let store = Runtime::current().in_scope(ScopeId::ROOT, || HistoryStore {
            load: Signal::new(HistoryLoad::Loading),
            toasts,
        });
        store.reload();
        store
    });
    use_context_provider(|| store)
}

/// The store provided at the root.
pub fn use_history() -> HistoryStore {
    use_context::<HistoryStore>()
}
