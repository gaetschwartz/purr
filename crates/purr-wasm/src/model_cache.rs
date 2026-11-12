//! The Cache Storage bucket transformers.js downloads model files into.

use crate::error::{IntegrityError, WebError, WebResult};
use crate::integrity::FileChecker;
use crate::local::run_local;
use gloo_timers::future::TimeoutFuture;
use js_sys::Uint8Array;
use purr_common::web_model::registry::{WebModelFile, WebModelSpec};
use wasm_bindgen::{JsCast as _, JsValue};
use wasm_bindgen_futures::JsFuture;
use web_sys::{Cache, CacheStorage, ReadableStreamDefaultReader, Response};

const CACHE_NAME: &str = "transformers-cache";
const YIELD_EVERY_BYTES: usize = 4 * 1024 * 1024;

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

async fn cached_response(cache: &Cache, url: &str) -> WebResult<Option<Response>> {
    let hit = JsFuture::from(cache.match_with_str(url))
        .await
        .map_err(|e| WebError::web_api("Cache.match", e))?;
    if hit.is_undefined() {
        return Ok(None);
    }
    hit.dyn_into::<Response>()
        .map(Some)
        .map_err(|e| WebError::web_api("Response cast", e))
}

async fn all_cached(urls: Vec<String>) -> WebResult<bool> {
    let Some(cache) = open_existing().await? else {
        return Ok(false);
    };
    for url in &urls {
        if cached_response(&cache, url).await?.is_none() {
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

async fn hash_cached_file(cache: &Cache, url: &str, file: &WebModelFile) -> WebResult<()> {
    let response = cached_response(cache, url)
        .await?
        .ok_or_else(|| IntegrityError::Missing {
            path: file.path.clone(),
        })?;
    let body = response.body().ok_or_else(|| {
        WebError::web_api(
            "Response.body",
            JsValue::from_str("the cached file has no body"),
        )
    })?;
    let reader = body
        .get_reader()
        .dyn_into::<ReadableStreamDefaultReader>()
        .map_err(|e| WebError::web_api("ReadableStream.getReader", e.into()))?;
    let mut checker = FileChecker::new(file);
    let mut unyielded = 0_usize;
    loop {
        let step = JsFuture::from(reader.read())
            .await
            .map_err(|e| WebError::web_api("ReadableStreamDefaultReader.read", e))?;
        let done = js_sys::Reflect::get(&step, &JsValue::from_str("done"))
            .map_err(|e| WebError::web_api("ReadableStreamDefaultReader.read", e))?
            .as_bool()
            .ok_or_else(|| {
                WebError::web_api(
                    "ReadableStreamDefaultReader.read",
                    JsValue::from_str("the read result has no `done` flag"),
                )
            })?;
        if done {
            break;
        }
        let chunk = js_sys::Reflect::get(&step, &JsValue::from_str("value"))
            .map_err(|e| WebError::web_api("ReadableStreamDefaultReader.read", e))?
            .dyn_into::<Uint8Array>()
            .map_err(|e| WebError::web_api("ReadableStream chunk", e))?;
        let bytes = chunk.to_vec();
        checker.update(&bytes);
        unyielded += bytes.len();
        if unyielded >= YIELD_EVERY_BYTES {
            TimeoutFuture::new(0).await;
            unyielded = 0;
        }
    }
    checker.finish()?;
    Ok(())
}

async fn verify_all(files: Vec<(String, WebModelFile)>) -> WebResult<()> {
    let cache = open_existing()
        .await?
        .ok_or_else(|| IntegrityError::Missing {
            path: CACHE_NAME.to_owned(),
        })?;
    for (url, file) in &files {
        hash_cached_file(&cache, url, file).await?;
    }
    Ok(())
}

fn file_urls(spec: &WebModelSpec) -> Vec<String> {
    spec.files.iter().map(|file| spec.file_url(file)).collect()
}

/// Whether every file of the model is cached.
pub async fn is_installed(spec: &WebModelSpec) -> WebResult<bool> {
    let urls = file_urls(spec);
    run_local(move || all_cached(urls)).await
}

/// Drops the given cached files of the model.
pub async fn remove(spec: &WebModelSpec, files: &[&WebModelFile]) -> WebResult<()> {
    let urls = files.iter().map(|file| spec.file_url(file)).collect();
    run_local(move || delete_all(urls)).await
}

/// Hashes every cached file against the manifest. Files that fail the check are evicted so the
/// next use downloads them again.
pub async fn verify(spec: &WebModelSpec) -> WebResult<()> {
    let files: Vec<(String, WebModelFile)> = spec
        .files
        .iter()
        .map(|file| (spec.file_url(file), file.clone()))
        .collect();
    let outcome = run_local(move || verify_all(files)).await;
    if matches!(outcome, Err(WebError::Integrity(_))) {
        let everything: Vec<&WebModelFile> = spec.files.iter().collect();
        remove(spec, &everything).await?;
    }
    outcome
}

#[must_use]
pub fn location(spec: &WebModelSpec) -> String {
    format!("cache://{CACHE_NAME}/{}", spec.id)
}
