//! The browser's history: one JSON text per entry in the `history` object store.

use crate::{
    error::WebError,
    idb::{self, HISTORY},
    local::run_local,
};
use js_sys::Array;
use purr_common::{
    history::{
        entry::{AudioAvailability, HistoryEntry, HistoryId, HistoryRecord},
        envelope::{self, UnreadableReason},
        listing::{HistoryListing, Saved, UnreadableEntry},
        retention::{Limits, Stored, evictions},
        title::{TitleProblem, clean_title},
    },
    platform::PlatformError,
};
use wasm_bindgen::JsValue;
use web_sys::{IdbObjectStore, IdbTransactionMode};

/// Why a history operation in the browser failed.
#[derive(Debug, thiserror::Error)]
pub enum HistoryStoreError {
    #[error(transparent)]
    Web(#[from] WebError),
    #[error("There is no saved transcription {0}")]
    NotFound(HistoryId),
    #[error("The saved transcription {id} {reason}")]
    Unreadable {
        id: HistoryId,
        reason: UnreadableReason,
    },
    #[error("The transcription could not be written: {0}")]
    Encode(#[from] serde_json::Error),
    #[error("{0}")]
    Title(#[from] TitleProblem),
}

impl From<HistoryStoreError> for PlatformError {
    fn from(error: HistoryStoreError) -> Self {
        match error {
            HistoryStoreError::NotFound(id) => Self::HistoryEntryNotFound { id },
            HistoryStoreError::Web(_)
            | HistoryStoreError::Unreadable { .. }
            | HistoryStoreError::Encode(_)
            | HistoryStoreError::Title(_) => Self::history(error),
        }
    }
}

type Outcome<T> = Result<T, HistoryStoreError>;

fn web(operation: &'static str) -> impl FnOnce(JsValue) -> HistoryStoreError {
    move |error| WebError::indexeddb(operation, error).into()
}

fn key_of(id: &HistoryId) -> JsValue {
    JsValue::from_str(id.as_str())
}

fn count_of(value: &JsValue) -> Outcome<usize> {
    value.as_f64().map(|count| count as usize).ok_or_else(|| {
        WebError::indexeddb(
            "history count",
            JsValue::from_str("the count is not a number"),
        )
        .into()
    })
}

struct Row {
    key: String,
    text: Result<String, UnreadableReason>,
}

async fn rows(store: &IdbObjectStore) -> Outcome<Vec<Row>> {
    let keys_request = store.get_all_keys().map_err(web("history keys"))?;
    let values_request = store.get_all().map_err(web("history values"))?;
    let keys: Array = idb::settle(&keys_request, "history keys").await?.into();
    let values: Array = idb::settle(&values_request, "history values").await?.into();
    Ok((0..keys.length())
        .map(|index| {
            let key = keys
                .get(index)
                .as_string()
                .unwrap_or_else(|| "(a key that is not text)".to_owned());
            let text = values
                .get(index)
                .as_string()
                .ok_or_else(|| UnreadableReason::Corrupt {
                    detail: "the stored value is not text".to_owned(),
                });
            Row { key, text }
        })
        .collect())
}

fn decode_row(row: &Row) -> Result<HistoryEntry, UnreadableReason> {
    let text = row.text.as_ref().map_err(Clone::clone)?;
    let entry = envelope::decode(text)?;
    if entry.id.as_str() == row.key {
        Ok(entry)
    } else {
        Err(UnreadableReason::Corrupt {
            detail: "its id differs from its key".to_owned(),
        })
    }
}

async fn fetch(store: &IdbObjectStore, id: &HistoryId) -> Outcome<HistoryEntry> {
    let request = store.get(&key_of(id)).map_err(web("history get"))?;
    let value = idb::settle(&request, "history get").await?;
    let Some(text) = value.as_string() else {
        return Err(if value.is_undefined() {
            HistoryStoreError::NotFound(id.clone())
        } else {
            HistoryStoreError::Unreadable {
                id: id.clone(),
                reason: UnreadableReason::Corrupt {
                    detail: "the stored value is not text".to_owned(),
                },
            }
        });
    };
    envelope::decode(&text).map_err(|reason| HistoryStoreError::Unreadable {
        id: id.clone(),
        reason,
    })
}

async fn put(store: &IdbObjectStore, entry: &HistoryEntry) -> Outcome<()> {
    let text = envelope::encode(entry)?;
    let request = store
        .put_with_key(&JsValue::from_str(&text), &key_of(&entry.id))
        .map_err(web("history put"))?;
    idb::settle(&request, "history put").await?;
    Ok(())
}

async fn writable() -> Outcome<(web_sys::IdbTransaction, IdbObjectStore)> {
    let database = idb::connect().await?;
    let transaction = database
        .transaction_with_str_and_mode(HISTORY, IdbTransactionMode::Readwrite)
        .map_err(web("history transaction"))?;
    let store = transaction
        .object_store(HISTORY)
        .map_err(web("history store"))?;
    Ok((transaction, store))
}

/// Every saved transcription as a summary, newest first, and the records that could not be read.
pub async fn list() -> Outcome<HistoryListing> {
    run_local(|| async {
        let database = idb::connect().await?;
        let transaction = database
            .transaction_with_str(HISTORY)
            .map_err(web("history transaction"))?;
        let store = transaction
            .object_store(HISTORY)
            .map_err(web("history store"))?;
        let mut summaries = Vec::new();
        let mut unreadable = Vec::new();
        for row in rows(&store).await? {
            match decode_row(&row) {
                Ok(entry) => summaries.push(entry.summary(AudioAvailability::NotKept)),
                Err(reason) => unreadable.push(UnreadableEntry {
                    name: row.key,
                    reason,
                }),
            }
        }
        Ok(HistoryListing::new(summaries, unreadable))
    })
    .await
}

/// The entry `id`.
pub async fn get(id: &HistoryId) -> Outcome<HistoryRecord> {
    let id = id.clone();
    run_local(move || async move {
        let database = idb::connect().await?;
        let transaction = database
            .transaction_with_str(HISTORY)
            .map_err(web("history transaction"))?;
        let store = transaction
            .object_store(HISTORY)
            .map_err(web("history store"))?;
        let entry = fetch(&store, &id).await?;
        Ok(HistoryRecord {
            entry,
            audio: AudioAvailability::NotKept,
        })
    })
    .await
}

/// Stores `entry` and drops the oldest entries beyond `limits`, in one transaction.
pub async fn save(entry: &HistoryEntry, limits: Limits) -> Outcome<Saved> {
    let entry = entry.clone();
    run_local(move || async move {
        let (transaction, store) = writable().await?;
        put(&store, &entry).await?;
        let mut stored = Vec::new();
        for row in rows(&store).await? {
            if let (Ok(read), Ok(text)) = (decode_row(&row), &row.text) {
                stored.push(Stored {
                    id: read.id,
                    created_at_ms: read.created_at_ms,
                    bytes: text.len() as u64,
                });
            }
        }
        let evicted = evictions(&stored, limits);
        for id in &evicted {
            let request = store.delete(&key_of(id)).map_err(web("history delete"))?;
            idb::settle(&request, "history delete").await?;
        }
        idb::committed(&transaction, "history save").await?;
        Ok(Saved { evicted })
    })
    .await
}

/// Gives the entry `id` the title `title`.
pub async fn rename(id: &HistoryId, title: &str) -> Outcome<()> {
    let (id, title) = (id.clone(), title.to_owned());
    run_local(move || async move {
        let title = clean_title(&title)?;
        let (transaction, store) = writable().await?;
        let mut entry = fetch(&store, &id).await?;
        entry.title = title;
        put(&store, &entry).await?;
        idb::committed(&transaction, "history rename").await?;
        Ok(())
    })
    .await
}

/// Deletes the entry `id`.
pub async fn delete(id: &HistoryId) -> Outcome<()> {
    let id = id.clone();
    run_local(move || async move {
        let (transaction, store) = writable().await?;
        let found = idb::settle(
            &store
                .count_with_key(&key_of(&id))
                .map_err(web("history count"))?,
            "history count",
        )
        .await?;
        if count_of(&found)? < 1 {
            return Err(HistoryStoreError::NotFound(id));
        }
        let request = store.delete(&key_of(&id)).map_err(web("history delete"))?;
        idb::settle(&request, "history delete").await?;
        idb::committed(&transaction, "history delete").await?;
        Ok(())
    })
    .await
}

/// Deletes every stored entry, and returns how many there were.
pub async fn clear() -> Outcome<usize> {
    run_local(|| async {
        let (transaction, store) = writable().await?;
        let counted = idb::settle(
            &store.count().map_err(web("history count"))?,
            "history count",
        )
        .await?;
        let request = store.clear().map_err(web("history clear"))?;
        idb::settle(&request, "history clear").await?;
        idb::committed(&transaction, "history clear").await?;
        count_of(&counted)
    })
    .await
}
