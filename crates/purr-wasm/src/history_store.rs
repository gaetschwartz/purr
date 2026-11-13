//! The browser's history: one JSON text per entry in the `history` object store, and the audio
//! kept with an entry in `history_audio` under the same id.

use crate::{
    error::WebError,
    history_audio,
    idb::{self, HISTORY, HISTORY_AUDIO},
    local::run_local,
};
use js_sys::Array;
use purr_common::{
    history::{
        audio::{AudioOutcome, HistoryAudio},
        entry::{AudioAvailability, HistoryEntry, HistoryId, HistoryRecord, HistorySource},
        envelope::{self, UnreadableReason},
        listing::{HistoryListing, UnreadableEntry},
        title::{TitleProblem, clean_title},
    },
    platform::PlatformError,
};
use wasm_bindgen::JsValue;
use web_sys::{IdbObjectStore, IdbTransaction, IdbTransactionMode};

/// Why a history operation in the browser failed.
#[derive(Debug, thiserror::Error)]
pub enum HistoryStoreError {
    #[error(transparent)]
    Web(#[from] WebError),
    #[error("There is no saved transcription {0}")]
    NotFound(HistoryId),
    #[error("The saved transcription {id} can't be read: {reason}")]
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

pub type Outcome<T> = Result<T, HistoryStoreError>;

pub fn web(operation: &'static str) -> impl FnOnce(JsValue) -> HistoryStoreError {
    move |error| WebError::indexeddb(operation, error).into()
}

pub fn key_of(id: &HistoryId) -> JsValue {
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

/// One transaction over the entries and their audio, which are always changed together.
pub struct Stores {
    pub transaction: IdbTransaction,
    pub entries: IdbObjectStore,
    pub audio: IdbObjectStore,
}

pub async fn open(mode: IdbTransactionMode) -> Outcome<Stores> {
    let database = idb::connect().await?;
    let names = Array::of2(
        &JsValue::from_str(HISTORY),
        &JsValue::from_str(HISTORY_AUDIO),
    );
    let transaction = database
        .transaction_with_str_sequence_and_mode(&names, mode)
        .map_err(web("history transaction"))?;
    let entries = transaction
        .object_store(HISTORY)
        .map_err(web("history store"))?;
    let audio = transaction
        .object_store(HISTORY_AUDIO)
        .map_err(web("history audio store"))?;
    Ok(Stores {
        transaction,
        entries,
        audio,
    })
}

pub struct Row {
    pub key: String,
    pub text: Result<String, UnreadableReason>,
}

pub async fn rows(store: &IdbObjectStore) -> Outcome<Vec<Row>> {
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

pub fn decode_row(row: &Row) -> Result<HistoryEntry, UnreadableReason> {
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

pub async fn put(store: &IdbObjectStore, entry: &HistoryEntry) -> Outcome<()> {
    let text = envelope::encode(entry)?;
    let request = store
        .put_with_key(&JsValue::from_str(&text), &key_of(&entry.id))
        .map_err(web("history put"))?;
    idb::settle(&request, "history put").await?;
    Ok(())
}

fn availability(entry: &HistoryEntry, stored: bool) -> AudioAvailability {
    match &entry.source {
        HistorySource::NotKept => AudioAvailability::NotKept,
        HistorySource::StoredInBrowser if stored => AudioAvailability::Available,
        HistorySource::StoredInBrowser | HistorySource::Path { .. } => AudioAvailability::Missing,
    }
}

/// Every saved transcription as a summary, newest first, and the records that could not be read.
pub async fn list() -> Outcome<HistoryListing> {
    run_local(|| async {
        let stores = open(IdbTransactionMode::Readonly).await?;
        let entry_rows = rows(&stores.entries).await?;
        let stored = history_audio::keys(&stores.audio).await?;
        let mut summaries = Vec::new();
        let mut unreadable = Vec::new();
        for row in entry_rows {
            match decode_row(&row) {
                Ok(entry) => {
                    let audio = availability(&entry, stored.contains(entry.id.as_str()));
                    summaries.push(entry.summary(audio));
                }
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
        let stores = open(IdbTransactionMode::Readonly).await?;
        let entry = fetch(&stores.entries, &id).await?;
        let stored = history_audio::has(&stores.audio, &id).await?;
        let audio = availability(&entry, stored);
        Ok(HistoryRecord { entry, audio })
    })
    .await
}

/// Stores `entry`, with `audio` when there is some to keep with it.
pub async fn save(entry: &HistoryEntry, audio: Option<HistoryAudio>) -> Outcome<AudioOutcome> {
    match audio {
        Some(audio) => history_audio::save(entry, audio).await,
        None => {
            let entry = entry.clone();
            run_local(move || async move {
                let stores = open(IdbTransactionMode::Readwrite).await?;
                let mut entry = entry;
                if entry.source == HistorySource::StoredInBrowser {
                    let stored = history_audio::has(&stores.audio, &entry.id).await?;
                    if !stored {
                        entry.source = HistorySource::NotKept;
                    }
                }
                put(&stores.entries, &entry).await?;
                idb::committed(&stores.transaction, "history save").await?;
                Ok(AudioOutcome::NotOffered)
            })
            .await
        }
    }
}

/// Gives the entry `id` the title `title`.
pub async fn rename(id: &HistoryId, title: &str) -> Outcome<()> {
    let (id, title) = (id.clone(), title.to_owned());
    run_local(move || async move {
        let title = clean_title(&title)?;
        let stores = open(IdbTransactionMode::Readwrite).await?;
        let mut entry = fetch(&stores.entries, &id).await?;
        entry.title = title;
        put(&stores.entries, &entry).await?;
        idb::committed(&stores.transaction, "history rename").await?;
        Ok(())
    })
    .await
}

/// Deletes the entry `id` and its audio.
pub async fn delete(id: &HistoryId) -> Outcome<()> {
    let id = id.clone();
    run_local(move || async move {
        let stores = open(IdbTransactionMode::Readwrite).await?;
        let found = idb::settle(
            &stores
                .entries
                .count_with_key(&key_of(&id))
                .map_err(web("history count"))?,
            "history count",
        )
        .await?;
        if count_of(&found)? < 1 {
            return Err(HistoryStoreError::NotFound(id));
        }
        let request = stores
            .entries
            .delete(&key_of(&id))
            .map_err(web("history delete"))?;
        idb::settle(&request, "history delete").await?;
        history_audio::remove(&stores.audio, &id).await?;
        idb::committed(&stores.transaction, "history delete").await?;
        Ok(())
    })
    .await
}

/// Deletes every stored entry and all the audio, and returns how many entries there were.
pub async fn clear() -> Outcome<usize> {
    run_local(|| async {
        let stores = open(IdbTransactionMode::Readwrite).await?;
        let counted = idb::settle(
            &stores.entries.count().map_err(web("history count"))?,
            "history count",
        )
        .await?;
        let request = stores.entries.clear().map_err(web("history clear"))?;
        idb::settle(&request, "history clear").await?;
        let request = stores.audio.clear().map_err(web("history audio clear"))?;
        idb::settle(&request, "history audio clear").await?;
        idb::committed(&stores.transaction, "history clear").await?;
        count_of(&counted)
    })
    .await
}
