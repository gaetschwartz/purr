//! The Cache Storage bucket transformers.js downloads model files into.

use crate::byte_stream::{body_reader, next_chunk};
use crate::error::{IntegrityError, WebError, WebResult};
use crate::integrity::FileChecker;
use crate::local::run_local;
use gloo_timers::future::TimeoutFuture;
use purr_common::platform::{ModelCheckProgress, ModelCheckSender};
use purr_common::web_model::registry::{CachedFile, WebModelFile, WebModelSpec};
use wasm_bindgen::{JsCast as _, JsValue};
use wasm_bindgen_futures::JsFuture;
use web_sys::{Cache, CacheStorage, Response};

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

/// The cache transformers.js reads, created if the browser has none yet.
pub async fn open_for_writing() -> WebResult<Cache> {
    JsFuture::from(cache_storage()?.open(CACHE_NAME))
        .await
        .map_err(|e| WebError::web_api("CacheStorage.open", e))?
        .dyn_into::<Cache>()
        .map_err(|e| WebError::web_api("Cache cast", e))
}

pub async fn open_existing() -> WebResult<Option<Cache>> {
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

/// Whether the cache holds everything `file` is stored as.
pub async fn has_file(cache: &Cache, spec: &WebModelSpec, file: &WebModelFile) -> WebResult<bool> {
    for url in spec.cached_urls(file) {
        if cached_response(cache, &url).await?.is_none() {
            return Ok(false);
        }
    }
    Ok(true)
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

struct CheckReporter {
    sender: Option<ModelCheckSender>,
    total_bytes: u64,
}

impl CheckReporter {
    fn report(&self, checked_bytes: u64) {
        let Some(sender) = &self.sender else {
            return;
        };
        let progress = ModelCheckProgress {
            checked_bytes,
            total_bytes: self.total_bytes,
        };
        if sender.unbounded_send(progress).is_err() {
            tracing::debug!("Nobody follows the model check any more");
        }
    }
}

async fn hash_cached_file(
    cache: &Cache,
    url: &str,
    file: &CachedFile,
    checked_before: u64,
    reporter: &CheckReporter,
) -> WebResult<()> {
    let response = cached_response(cache, url)
        .await?
        .ok_or_else(|| IntegrityError::Missing {
            path: file.path.clone(),
        })?;
    let reader = body_reader(&response)?;
    let mut checker = FileChecker::new(file);
    let mut unyielded = 0_usize;
    while let Some(chunk) = next_chunk(&reader).await? {
        let bytes = chunk.to_vec();
        checker.update(&bytes);
        unyielded += bytes.len();
        if unyielded >= YIELD_EVERY_BYTES {
            reporter.report(checked_before.saturating_add(checker.received_bytes()));
            TimeoutFuture::new(0).await;
            unyielded = 0;
        }
    }
    checker.finish()?;
    reporter.report(checked_before.saturating_add(file.size_bytes));
    Ok(())
}

async fn verify_all(
    files: Vec<(String, CachedFile)>,
    progress: Option<ModelCheckSender>,
) -> WebResult<()> {
    let cache = open_existing()
        .await?
        .ok_or_else(|| IntegrityError::Missing {
            path: CACHE_NAME.to_owned(),
        })?;
    let reporter = CheckReporter {
        sender: progress,
        total_bytes: files.iter().map(|(_, file)| file.size_bytes).sum(),
    };
    reporter.report(0);
    let mut checked_before = 0_u64;
    for (url, file) in &files {
        hash_cached_file(&cache, url, file, checked_before, &reporter).await?;
        checked_before = checked_before.saturating_add(file.size_bytes);
    }
    Ok(())
}

fn file_urls(spec: &WebModelSpec) -> Vec<String> {
    spec.files
        .iter()
        .flat_map(|file| spec.cached_urls(file))
        .collect()
}

/// Whether every file of the model is cached, converted files as both their graph and weights.
pub async fn is_installed(spec: &WebModelSpec) -> WebResult<bool> {
    let urls = file_urls(spec);
    run_local(move || all_cached(urls)).await
}

/// Drops the given cached files of the model.
pub async fn remove(spec: &WebModelSpec, files: &[&WebModelFile]) -> WebResult<()> {
    let urls = files
        .iter()
        .flat_map(|file| spec.cached_urls(file))
        .collect();
    run_local(move || delete_all(urls)).await
}

/// Hashes every cached file against the manifest. Files that fail the check are evicted so the
/// next use downloads them again.
pub async fn verify(spec: &WebModelSpec) -> WebResult<()> {
    verify_reporting(spec, None).await
}

/// Like [`verify`], telling `progress` how many bytes are hashed.
pub async fn verify_with_progress(
    spec: &WebModelSpec,
    progress: ModelCheckSender,
) -> WebResult<()> {
    verify_reporting(spec, Some(progress)).await
}

async fn verify_reporting(
    spec: &WebModelSpec,
    progress: Option<ModelCheckSender>,
) -> WebResult<()> {
    let files: Vec<(String, CachedFile)> = spec
        .files
        .iter()
        .flat_map(WebModelFile::cached_files)
        .map(|file| (spec.cached_url(&file), file))
        .collect();
    let outcome = run_local(move || verify_all(files, progress)).await;
    if matches!(outcome, Err(WebError::Integrity { .. })) {
        let everything: Vec<&WebModelFile> = spec.files.iter().collect();
        remove(spec, &everything).await?;
    }
    outcome
}

#[must_use]
pub fn location(spec: &WebModelSpec) -> String {
    format!("cache://{CACHE_NAME}/{}", spec.id)
}
