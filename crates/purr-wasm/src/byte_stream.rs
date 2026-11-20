//! Reading the body of a fetched or cached `Response` chunk by chunk.

use crate::error::{WebError, WebResult};
use js_sys::{Reflect, Uint8Array};
use wasm_bindgen::{JsCast as _, JsValue};
use wasm_bindgen_futures::JsFuture;
use web_sys::{ReadableStreamDefaultReader, Response};

pub fn body_reader(response: &Response) -> WebResult<ReadableStreamDefaultReader> {
    let body = response.body().ok_or_else(|| {
        WebError::web_api(
            "Response.body",
            JsValue::from_str("the response has no body"),
        )
    })?;
    body.get_reader()
        .dyn_into::<ReadableStreamDefaultReader>()
        .map_err(|e| WebError::web_api("ReadableStream.getReader", e.into()))
}

/// The next chunk of the stream, or `None` at its end.
pub async fn next_chunk(reader: &ReadableStreamDefaultReader) -> WebResult<Option<Uint8Array>> {
    let step = JsFuture::from(reader.read())
        .await
        .map_err(|e| WebError::web_api("ReadableStreamDefaultReader.read", e))?;
    let done = Reflect::get(&step, &JsValue::from_str("done"))
        .map_err(|e| WebError::web_api("ReadableStreamDefaultReader.read", e))?
        .as_bool()
        .ok_or_else(|| {
            WebError::web_api(
                "ReadableStreamDefaultReader.read",
                JsValue::from_str("the read result has no `done` flag"),
            )
        })?;
    if done {
        return Ok(None);
    }
    Reflect::get(&step, &JsValue::from_str("value"))
        .map_err(|e| WebError::web_api("ReadableStreamDefaultReader.read", e))?
        .dyn_into::<Uint8Array>()
        .map(Some)
        .map_err(|e| WebError::web_api("ReadableStream chunk", e))
}
