//! The audio kept with saved transcriptions in the browser: one blob per history id.

use crate::{
    error::WebError,
    history_store::{HistoryStoreError, Outcome, Row, decode_row, key_of, open, put, rows, web},
    idb,
    local::run_local,
    storage_space,
};
use js_sys::{Array, Uint8Array};
use purr_common::history::{
    audio::{AudioDropReason, AudioOutcome, HistoryAudio},
    audio_budget::{self, AudioPlan, StoredAudio},
    entry::{HistoryEntry, HistoryId, HistorySource},
};
use std::{
    cell::Cell,
    collections::{HashMap, HashSet},
};
use wasm_bindgen::{JsCast as _, JsValue};
use wasm_bindgen_futures::{JsFuture, spawn_local};
use web_sys::{Blob, BlobPropertyBag, IdbObjectStore, IdbTransactionMode};

thread_local! {
    static PERSISTENCE_ASKED: Cell<bool> = const { Cell::new(false) };
}

fn api_error(api: &'static str, error: JsValue) -> HistoryStoreError {
    WebError::web_api(api, error).into()
}

fn blob_of(audio: &HistoryAudio) -> Outcome<Blob> {
    let options = BlobPropertyBag::new();
    if let Some(content_type) = &audio.content_type {
        options.set_type(content_type);
    }
    let parts = Array::new();
    parts.push(&Uint8Array::from(audio.bytes.as_ref()));
    Blob::new_with_u8_array_sequence_and_options(&parts, &options)
        .map_err(|error| api_error("Blob", error))
}

fn stored_size(value: &JsValue) -> u64 {
    value.dyn_ref::<Blob>().map_or_else(
        || {
            tracing::warn!("A stored audio record is not a blob");
            0
        },
        |blob| blob.size() as u64,
    )
}

/// The ids that have audio stored.
pub async fn keys(store: &IdbObjectStore) -> Outcome<HashSet<String>> {
    let request = store.get_all_keys().map_err(web("history audio keys"))?;
    let keys: Array = idb::settle(&request, "history audio keys").await?.into();
    Ok(keys.iter().filter_map(|key| key.as_string()).collect())
}

/// Whether audio is stored for `id`.
pub async fn has(store: &IdbObjectStore, id: &HistoryId) -> Outcome<bool> {
    let request = store
        .count_with_key(&key_of(id))
        .map_err(web("history audio count"))?;
    let counted = idb::settle(&request, "history audio count").await?;
    Ok(counted.as_f64().is_some_and(|count| count >= 1.0))
}

/// The size in bytes of each stored audio file, by id.
async fn sizes(store: &IdbObjectStore) -> Outcome<HashMap<String, u64>> {
    let keys_request = store.get_all_keys().map_err(web("history audio keys"))?;
    let values_request = store.get_all().map_err(web("history audio values"))?;
    let ids: Array = idb::settle(&keys_request, "history audio keys")
        .await?
        .into();
    let blobs: Array = idb::settle(&values_request, "history audio values")
        .await?
        .into();
    Ok(ids
        .iter()
        .zip(blobs.iter())
        .filter_map(|(id, blob)| Some((id.as_string()?, stored_size(&blob))))
        .collect())
}

/// Deletes the audio stored for `id`, if any.
pub async fn remove(store: &IdbObjectStore, id: &HistoryId) -> Outcome<()> {
    let request = store
        .delete(&key_of(id))
        .map_err(web("history audio delete"))?;
    idb::settle(&request, "history audio delete").await?;
    Ok(())
}

fn stored_audio(entries: &[HistoryEntry], sizes: &HashMap<String, u64>) -> Vec<StoredAudio> {
    entries
        .iter()
        .filter(|entry| entry.source == HistorySource::StoredInBrowser)
        .filter_map(|entry| {
            sizes.get(entry.id.as_str()).map(|bytes| StoredAudio {
                id: entry.id.clone(),
                created_at_ms: entry.created_at_ms,
                bytes: *bytes,
            })
        })
        .collect()
}

fn readable(rows: &[Row]) -> Vec<HistoryEntry> {
    rows.iter().filter_map(|row| decode_row(row).ok()).collect()
}

async fn budget() -> u64 {
    match storage_space::estimate().await {
        Ok(estimate) => audio_budget::budget(estimate.quota),
        Err(error) => {
            tracing::warn!(
                "Could not read the browser's storage quota, so the audio is stored without a budget: {error}"
            );
            u64::MAX
        }
    }
}

async fn store_with_audio(
    entry: &HistoryEntry,
    audio: &HistoryAudio,
    budget: u64,
) -> Outcome<AudioOutcome> {
    let stores = open(IdbTransactionMode::Readwrite).await?;
    let entry_rows = rows(&stores.entries).await?;
    let mut entries = readable(&entry_rows);
    let sizes = sizes(&stores.audio).await?;
    let plan = audio_budget::plan(
        &stored_audio(&entries, &sizes),
        &entry.id,
        audio.bytes.len() as u64,
        budget,
    );
    let mut stored = entry.clone();
    let outcome = match plan {
        AudioPlan::TooLarge => {
            stored.source = HistorySource::NotKept;
            AudioOutcome::Dropped(AudioDropReason::OverBudget)
        }
        AudioPlan::Store { drop_first } => {
            for id in &drop_first {
                remove(&stores.audio, id).await?;
                if let Some(older) = entries.iter_mut().find(|older| &older.id == id) {
                    older.source = HistorySource::NotKept;
                    put(&stores.entries, older).await?;
                }
                tracing::info!("Dropped the stored audio of {id} to stay within the budget");
            }
            let request = stores
                .audio
                .put_with_key(&blob_of(audio)?.into(), &key_of(&entry.id))
                .map_err(web("history audio put"))?;
            idb::settle(&request, "history audio put").await?;
            stored.source = HistorySource::StoredInBrowser;
            AudioOutcome::Kept
        }
    };
    put(&stores.entries, &stored).await?;
    idb::committed(&stores.transaction, "history save").await?;
    Ok(outcome)
}

async fn store_without_audio(entry: &HistoryEntry) -> Outcome<()> {
    let stores = open(IdbTransactionMode::Readwrite).await?;
    let mut stored = entry.clone();
    stored.source = HistorySource::NotKept;
    put(&stores.entries, &stored).await?;
    idb::committed(&stores.transaction, "history save").await?;
    Ok(())
}

fn ask_for_persistence() {
    if PERSISTENCE_ASKED.with(|asked| asked.replace(true)) {
        return;
    }
    spawn_local(async {
        match storage_space::request_persistence().await {
            Ok(true) => tracing::info!("The browser will keep this site's storage"),
            Ok(false) => tracing::info!("The browser did not promise to keep this site's storage"),
            Err(error) => tracing::debug!("Could not ask for persistent storage: {error}"),
        }
    });
}

/// Stores `entry` with `audio`. When the audio cannot be written the transcript is saved without
/// it.
pub async fn save(entry: &HistoryEntry, audio: HistoryAudio) -> Outcome<AudioOutcome> {
    let (entry, audio) = (entry.clone(), audio);
    run_local(move || async move {
        let budget = budget().await;
        match store_with_audio(&entry, &audio, budget).await {
            Ok(outcome) => {
                if outcome == AudioOutcome::Kept {
                    ask_for_persistence();
                }
                Ok(outcome)
            }
            Err(HistoryStoreError::Web(WebError::StorageFull { operation })) => {
                tracing::warn!("The browser has no room for the audio ({operation})");
                store_without_audio(&entry).await?;
                Ok(AudioOutcome::Dropped(AudioDropReason::StorageFull))
            }
            Err(error @ (HistoryStoreError::Web(_) | HistoryStoreError::Encode(_))) => {
                tracing::warn!("Could not store the audio, saving the transcript alone: {error}");
                store_without_audio(&entry).await?;
                Ok(AudioOutcome::Dropped(AudioDropReason::Failed {
                    detail: error.to_string(),
                }))
            }
            Err(
                error @ (HistoryStoreError::NotFound(_)
                | HistoryStoreError::Unreadable { .. }
                | HistoryStoreError::Title(_)),
            ) => Err(error),
        }
    })
    .await
}

/// The audio stored for `id`, if there is some.
pub async fn read(id: &HistoryId) -> Outcome<Option<HistoryAudio>> {
    let id = id.clone();
    run_local(move || async move {
        let stores = open(IdbTransactionMode::Readonly).await?;
        let request = stores
            .audio
            .get(&key_of(&id))
            .map_err(web("history audio get"))?;
        let value = idb::settle(&request, "history audio get").await?;
        if value.is_undefined() {
            return Ok(None);
        }
        let blob: Blob = value
            .dyn_into()
            .map_err(|_| api_error("Blob", JsValue::from_str("the stored audio is not a blob")))?;
        let buffer = JsFuture::from(blob.array_buffer())
            .await
            .map_err(|error| api_error("Blob.arrayBuffer", error))?;
        let content_type = Some(blob.type_()).filter(|kind| !kind.is_empty());
        Ok(Some(HistoryAudio {
            bytes: Uint8Array::new(&buffer).to_vec().into(),
            content_type,
        }))
    })
    .await
}

/// How many bytes of audio the entries have stored.
pub async fn used_bytes() -> Outcome<u64> {
    run_local(|| async {
        let stores = open(IdbTransactionMode::Readonly).await?;
        let entries = readable(&rows(&stores.entries).await?);
        let sizes = sizes(&stores.audio).await?;
        Ok(stored_audio(&entries, &sizes)
            .iter()
            .map(|audio| audio.bytes)
            .sum())
    })
    .await
}

/// Deletes all the stored audio and marks the entries that had some as not kept; returns the bytes
/// freed.
pub async fn clear() -> Outcome<u64> {
    run_local(|| async {
        let stores = open(IdbTransactionMode::Readwrite).await?;
        let mut entries = readable(&rows(&stores.entries).await?);
        let sizes = sizes(&stores.audio).await?;
        for entry in &mut entries {
            if entry.source == HistorySource::StoredInBrowser {
                entry.source = HistorySource::NotKept;
                put(&stores.entries, entry).await?;
            }
        }
        let request = stores.audio.clear().map_err(web("history audio clear"))?;
        idb::settle(&request, "history audio clear").await?;
        idb::committed(&stores.transaction, "history audio clear").await?;
        Ok(sizes.values().sum())
    })
    .await
}

/// Deletes the stored audio whose entry no longer exists; returns how many files went.
pub async fn sweep() -> Outcome<usize> {
    run_local(|| async {
        let stores = open(IdbTransactionMode::Readwrite).await?;
        let request = stores.entries.get_all_keys().map_err(web("history keys"))?;
        let entry_keys: HashSet<String> =
            Array::from(&idb::settle(&request, "history keys").await?)
                .iter()
                .filter_map(|key| key.as_string())
                .collect();
        let orphans: Vec<String> = keys(&stores.audio)
            .await?
            .into_iter()
            .filter(|key| !entry_keys.contains(key))
            .collect();
        for key in &orphans {
            let request = stores
                .audio
                .delete(&JsValue::from_str(key))
                .map_err(web("history audio delete"))?;
            idb::settle(&request, "history audio delete").await?;
        }
        idb::committed(&stores.transaction, "history audio sweep").await?;
        Ok(orphans.len())
    })
    .await
}
