//! The single `IndexedDB` database of the web platform: opening it, its object stores and the
//! promise-like waiting on requests and transactions.

use crate::error::{StorageError, WebError, WebResult};
use std::{cell::Cell, cell::RefCell, rc::Rc};
use tokio::sync::oneshot;
use wasm_bindgen::{JsCast as _, JsValue, closure::Closure};
use web_sys::{
    DomException, Event, IdbDatabase, IdbOpenDbRequest, IdbRequest, IdbTransaction,
    IdbVersionChangeEvent,
};

/// The name of the database.
pub const DB_NAME: &str = "PurrWebStorage";

/// The schema version: 1 held `files` and `metadata`, 2 added `history`, 3 adds `history_audio`.
pub const DB_VERSION: u32 = 3;

/// Uploaded audio, by file id.
pub const FILES: &str = "files";

/// Upload details, by file id.
pub const METADATA: &str = "metadata";

/// Saved transcriptions as JSON text, by history id.
pub const HISTORY: &str = "history";

/// The audio of saved transcriptions, by history id; apart from `files`, which holds uploads that
/// are purged after a day.
pub const HISTORY_AUDIO: &str = "history_audio";

const STORES: [&str; 4] = [FILES, METADATA, HISTORY, HISTORY_AUDIO];

thread_local! {
    static CONNECTION: RefCell<Option<IdbDatabase>> = const { RefCell::new(None) };
}

fn create_missing_stores(request: &IdbOpenDbRequest) {
    let database = match request
        .result()
        .map(wasm_bindgen::JsCast::unchecked_into::<IdbDatabase>)
    {
        Ok(database) => database,
        Err(error) => {
            tracing::error!("Could not read the database being upgraded: {error:?}");
            return;
        }
    };
    let existing = database.object_store_names();
    for store in STORES {
        if !existing.contains(store)
            && let Err(error) = database.create_object_store(store)
        {
            tracing::error!("Could not create the {store} store: {error:?}");
        }
    }
}

async fn open() -> WebResult<IdbDatabase> {
    let window = web_sys::window().ok_or_else(|| StorageError::IndexedDbUnavailable {
        js_error: "Window not available".to_owned(),
    })?;
    let factory = window
        .indexed_db()
        .map_err(|e| StorageError::IndexedDbUnavailable {
            js_error: crate::error::format_js_error(e),
        })?
        .ok_or_else(|| StorageError::IndexedDbUnavailable {
            js_error: "IndexedDB not available".to_owned(),
        })?;
    let request = factory
        .open_with_u32(DB_NAME, DB_VERSION)
        .map_err(StorageError::database_open_failed)?;

    let (tx, rx) = oneshot::channel::<WebResult<()>>();
    let sender = Rc::new(Cell::new(Some(tx)));

    let upgrading = request.clone();
    let on_upgrade = Closure::<dyn FnMut(IdbVersionChangeEvent)>::new(move |_| {
        create_missing_stores(&upgrading);
    });
    let on_success = {
        let sender = Rc::clone(&sender);
        Closure::once(move |_: Event| {
            if let Some(tx) = sender.take()
                && tx.send(Ok(())).is_err()
            {
                tracing::debug!("The database opened after its caller left");
            }
        })
    };
    let on_failure = {
        let sender = Rc::clone(&sender);
        let failed = request.clone();
        Closure::once(move |_: Event| {
            if let Some(tx) = sender.take() {
                let detail = failed
                    .error()
                    .ok()
                    .flatten()
                    .map_or_else(|| "unknown error".to_owned(), |error| error.message());
                let outcome = Err(StorageError::DatabaseOpenFailed { js_error: detail }.into());
                if tx.send(outcome).is_err() {
                    tracing::debug!("The database failed to open after its caller left");
                }
            }
        })
    };
    let on_blocked = {
        let sender = Rc::clone(&sender);
        Closure::once(move |_: Event| {
            if let Some(tx) = sender.take() {
                let outcome = Err(StorageError::DatabaseOpenFailed {
                    js_error: "another Purr tab still uses an older version of the storage; close it and reload".to_owned(),
                }
                .into());
                if tx.send(outcome).is_err() {
                    tracing::debug!("The database was blocked after its caller left");
                }
            }
        })
    };
    request.set_onupgradeneeded(Some(on_upgrade.as_ref().unchecked_ref()));
    request.set_onsuccess(Some(on_success.as_ref().unchecked_ref()));
    request.set_onerror(Some(on_failure.as_ref().unchecked_ref()));
    request.set_onblocked(Some(on_blocked.as_ref().unchecked_ref()));

    rx.await.map_err(|error| {
        WebError::web_api("IndexedDB open", JsValue::from_str(&error.to_string()))
    })??;
    let database: IdbDatabase = request
        .result()
        .map_err(StorageError::database_connection_failed)?
        .dyn_into()
        .map_err(|value| StorageError::DatabaseCastFailed {
            js_error: crate::error::format_js_error(value),
        })?;

    let closing = database.clone();
    let on_version_change = Closure::<dyn FnMut(Event)>::new(move |_| {
        closing.close();
        CONNECTION.with(|slot| slot.borrow_mut().take());
    });
    database.set_onversionchange(Some(on_version_change.as_ref().unchecked_ref()));
    on_version_change.forget();
    Ok(database)
}

/// The shared connection, opened on first use.
pub async fn connect() -> WebResult<IdbDatabase> {
    if let Some(database) = CONNECTION.with(|slot| slot.borrow().clone()) {
        return Ok(database);
    }
    let opened = open().await?;
    let kept = CONNECTION.with(|slot| {
        let mut slot = slot.borrow_mut();
        slot.get_or_insert_with(|| opened.clone()).clone()
    });
    if kept != opened {
        opened.close();
    }
    Ok(kept)
}

/// Detaches the handlers it guards when dropped, so a late event never reaches a dropped closure.
struct Detach<F: Fn()>(F);

impl<F: Fn()> Drop for Detach<F> {
    fn drop(&mut self) {
        (self.0)();
    }
}

const QUOTA_EXCEEDED: &str = "QuotaExceededError";

/// The error of a failed request or transaction: running out of storage is told apart.
fn failure(action: &'static str, exception: Option<DomException>, fallback: &str) -> WebError {
    match exception {
        Some(exception) if exception.name() == QUOTA_EXCEEDED => WebError::StorageFull {
            operation: action.to_owned(),
        },
        Some(exception) => WebError::IndexedDb {
            operation: action.to_owned(),
            js_error: exception.message(),
        },
        None => WebError::IndexedDb {
            operation: action.to_owned(),
            js_error: fallback.to_owned(),
        },
    }
}

/// Waits for `request` and returns its result.
pub async fn settle(request: &IdbRequest, action: &'static str) -> WebResult<JsValue> {
    let (tx, rx) = oneshot::channel::<WebResult<JsValue>>();
    let sender = Rc::new(Cell::new(Some(tx)));
    let on_success = {
        let sender = Rc::clone(&sender);
        let done = request.clone();
        Closure::once(move |_: Event| {
            if let Some(tx) = sender.take() {
                let outcome = done.result().map_err(|e| WebError::indexeddb(action, e));
                if tx.send(outcome).is_err() {
                    tracing::debug!("{action} finished after its caller left");
                }
            }
        })
    };
    let on_failure = {
        let failed = request.clone();
        Closure::once(move |_: Event| {
            if let Some(tx) = sender.take() {
                let outcome = Err(failure(
                    action,
                    failed.error().ok().flatten(),
                    "unknown error",
                ));
                if tx.send(outcome).is_err() {
                    tracing::debug!("{action} failed after its caller left");
                }
            }
        })
    };
    request.set_onsuccess(Some(on_success.as_ref().unchecked_ref()));
    request.set_onerror(Some(on_failure.as_ref().unchecked_ref()));
    let _detach = Detach(|| {
        request.set_onsuccess(None);
        request.set_onerror(None);
    });
    rx.await
        .map_err(|e| WebError::indexeddb(action, JsValue::from_str(&e.to_string())))?
}

/// Waits until `transaction` is committed.
pub async fn committed(transaction: &IdbTransaction, action: &'static str) -> WebResult<()> {
    let (tx, rx) = oneshot::channel::<WebResult<()>>();
    let sender = Rc::new(Cell::new(Some(tx)));
    let on_complete = {
        let sender = Rc::clone(&sender);
        Closure::once(move |_: Event| {
            if let Some(tx) = sender.take()
                && tx.send(Ok(())).is_err()
            {
                tracing::debug!("{action} committed after its caller left");
            }
        })
    };
    let on_abort = {
        let aborted = transaction.clone();
        Closure::<dyn FnMut(Event)>::new(move |_| {
            if let Some(tx) = sender.take() {
                let outcome = Err(failure(
                    action,
                    aborted.error(),
                    "the transaction was aborted",
                ));
                if tx.send(outcome).is_err() {
                    tracing::debug!("{action} aborted after its caller left");
                }
            }
        })
    };
    transaction.set_oncomplete(Some(on_complete.as_ref().unchecked_ref()));
    transaction.set_onabort(Some(on_abort.as_ref().unchecked_ref()));
    transaction.set_onerror(Some(on_abort.as_ref().unchecked_ref()));
    let _detach = Detach(|| {
        transaction.set_oncomplete(None);
        transaction.set_onabort(None);
        transaction.set_onerror(None);
    });
    rx.await
        .map_err(|e| WebError::indexeddb(action, JsValue::from_str(&e.to_string())))?
}
