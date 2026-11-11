//! How much the browser still lets this site store, from `navigator.storage.estimate()`.

use crate::error::{WebError, WebResult};
use js_sys::{Function, Promise, Reflect};
use wasm_bindgen::{JsCast as _, JsValue};
use wasm_bindgen_futures::JsFuture;

/// What the browser reports about this site's storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StorageEstimate {
    pub usage: u64,
    pub quota: u64,
}

impl StorageEstimate {
    /// Bytes still free under the quota.
    #[must_use]
    pub const fn free(self) -> u64 {
        self.quota.saturating_sub(self.usage)
    }
}

fn number(value: &JsValue, field: &str) -> WebResult<u64> {
    let raw = Reflect::get(value, &JsValue::from_str(field))
        .map_err(|e| WebError::web_api("StorageManager.estimate", e))?;
    raw.as_f64()
        .filter(|bytes| bytes.is_finite() && *bytes >= 0.0)
        .map(|bytes| bytes as u64)
        .ok_or_else(|| {
            WebError::web_api(
                "StorageManager.estimate",
                JsValue::from_str(&format!("{field} is not a byte count")),
            )
        })
}

/// Asks the browser for this site's storage usage and quota.
pub async fn estimate() -> WebResult<StorageEstimate> {
    let navigator = web_sys::window()
        .ok_or_else(|| WebError::web_api("StorageManager", JsValue::from_str("no window")))?
        .navigator();
    let storage = Reflect::get(&navigator, &JsValue::from_str("storage"))
        .map_err(|e| WebError::web_api("StorageManager", e))?;
    let estimate: Function = Reflect::get(&storage, &JsValue::from_str("estimate"))
        .map_err(|e| WebError::web_api("StorageManager.estimate", e))?
        .dyn_into()
        .map_err(|value| WebError::web_api("StorageManager.estimate", value))?;
    let promise: Promise = estimate
        .call0(&storage)
        .map_err(|e| WebError::web_api("StorageManager.estimate", e))?
        .dyn_into()
        .map_err(|value| WebError::web_api("StorageManager.estimate", value))?;
    let result = JsFuture::from(promise)
        .await
        .map_err(|e| WebError::web_api("StorageManager.estimate", e))?;
    Ok(StorageEstimate {
        usage: number(&result, "usage")?,
        quota: number(&result, "quota")?,
    })
}
