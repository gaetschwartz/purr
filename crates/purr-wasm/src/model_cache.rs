//! Presence of a model's files in the Cache Storage bucket transformers.js downloads into.

use crate::error::{WebError, WebResult};
use purr_common::web_model::registry::WebModelSpec;
use std::future::Future;
use tokio::sync::oneshot;
use wasm_bindgen::{JsCast as _, JsValue};
use wasm_bindgen_futures::{JsFuture, spawn_local};
use web_sys::{Cache, CacheStorage};

const CACHE_NAME: &str = "transformers-cache";

fn cache_storage() -> WebResult<CacheStorage> {
    let window = web_sys::window().ok_or_else(|| {
        WebError::web_api("Window", JsValue::from_str("no window object available"))
    })?;
    window
        .caches()
        .map_err(|e| WebError::web_api("CacheStorage", e))
}

async fn open_existing() -> WebResult<Option<Cache>> {
    let storage = cache_storage()?;
    let exists = JsFuture::from(storage.has(CACHE_NAME))
        .await
        .map_err(|e| WebError::web_api("CacheStorage.has", e))?
        .as_bool()
        .ok_or_else(|| {
            WebError::web_api(
                "CacheStorage.has",
                JsValue::from_str("did not resolve to a boolean"),
            )
        })?;
    if !exists {
        return Ok(None);
    }
    let cache = JsFuture::from(storage.open(CACHE_NAME))
        .await
        .map_err(|e| WebError::web_api("CacheStorage.open", e))?
        .dyn_into::<Cache>()
        .map_err(|e| WebError::web_api("Cache cast", e))?;
    Ok(Some(cache))
}

async fn all_cached(urls: Vec<String>) -> WebResult<bool> {
    let Some(cache) = open_existing().await? else {
        return Ok(false);
    };
    for url in &urls {
        let hit = JsFuture::from(cache.match_with_str(url))
            .await
            .map_err(|e| WebError::web_api("Cache.match", e))?;
        if hit.is_undefined() {
            return Ok(false);
        }
    }
    Ok(true)
}

async fn delete_all(urls: Vec<String>) -> WebResult<()> {
    let Some(cache) = open_existing().await? else {
        return Ok(());
    };
    for url in &urls {
        JsFuture::from(cache.delete_with_str(url))
            .await
            .map_err(|e| WebError::web_api("Cache.delete", e))?;
    }
    Ok(())
}

/// Runs a browser-API task on the local executor so the caller's future stays `Send`.
async fn run_local<T, Fut>(start: impl FnOnce() -> Fut + Send + 'static) -> WebResult<T>
where
    Fut: Future<Output = WebResult<T>> + 'static,
    T: Send + 'static,
{
    let (tx, rx) = oneshot::channel();
    spawn_local(async move {
        if tx.send(start().await).is_err() {
            tracing::warn!("Model cache caller went away before the task finished");
        }
    });
    rx.await
        .map_err(|e| WebError::web_api("Model cache task", JsValue::from_str(&e.to_string())))?
}

fn file_urls(spec: &WebModelSpec) -> Vec<String> {
    spec.files.iter().map(|file| spec.file_url(file)).collect()
}

/// Whether every file of the model is cached.
pub async fn is_installed(spec: &WebModelSpec) -> WebResult<bool> {
    let urls = file_urls(spec);
    run_local(move || all_cached(urls)).await
}

/// Drops every cached file of the model.
pub async fn remove(spec: &WebModelSpec) -> WebResult<()> {
    let urls = file_urls(spec);
    run_local(move || delete_all(urls)).await
}

#[must_use]
pub fn location(spec: &WebModelSpec) -> String {
    format!("cache://{CACHE_NAME}/{}", spec.id)
}
