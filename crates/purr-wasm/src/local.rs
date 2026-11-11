//! Bridges browser (non-`Send`) work into the `Send` futures the platform trait demands.

use crate::error::{WebError, WebResult};
use std::future::Future;
use tokio::sync::oneshot;
use wasm_bindgen::JsValue;
use wasm_bindgen_futures::spawn_local;

/// Runs a browser-API task on the local executor so the caller's future stays `Send`.
pub async fn run_local<T, Fut>(start: impl FnOnce() -> Fut + Send + 'static) -> WebResult<T>
where
    Fut: Future<Output = WebResult<T>> + 'static,
    T: Send + 'static,
{
    let (tx, rx) = oneshot::channel();
    spawn_local(async move {
        if tx.send(start().await).is_err() {
            tracing::warn!("The caller went away before a browser task finished");
        }
    });
    rx.await
        .map_err(|e| WebError::web_api("Local task", JsValue::from_str(&e.to_string())))?
}
