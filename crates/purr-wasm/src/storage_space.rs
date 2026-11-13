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

/// Asks the browser not to evict this site's storage under pressure, and tells whether it agreed.
pub async fn request_persistence() -> WebResult<bool> {
    let navigator = web_sys::window()
        .ok_or_else(|| WebError::web_api("StorageManager", JsValue::from_str("no window")))?
        .navigator();
    let storage = Reflect::get(&navigator, &JsValue::from_str("storage"))
        .map_err(|e| WebError::web_api("StorageManager", e))?;
    let persist: Function = Reflect::get(&storage, &JsValue::from_str("persist"))
        .map_err(|e| WebError::web_api("StorageManager.persist", e))?
        .dyn_into()
        .map_err(|value| WebError::web_api("StorageManager.persist", value))?;
    let promise: Promise = persist
        .call0(&storage)
        .map_err(|e| WebError::web_api("StorageManager.persist", e))?
        .dyn_into()
        .map_err(|value| WebError::web_api("StorageManager.persist", value))?;
    let granted = JsFuture::from(promise)
        .await
        .map_err(|e| WebError::web_api("StorageManager.persist", e))?;
    Ok(granted.as_bool() == Some(true))
}
