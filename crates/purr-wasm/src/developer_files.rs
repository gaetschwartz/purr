//! The files a user picked in a file input of the page.

use crate::developer_install::LocalFile;
use crate::error::{WebError, WebResult};
use wasm_bindgen::{JsCast as _, JsValue};
use web_sys::HtmlInputElement;

/// The files chosen in the `<input type="file">` with DOM id `input_id`.
pub fn picked(input_id: &str) -> WebResult<Vec<LocalFile>> {
    let document = web_sys::window()
        .and_then(|window| window.document())
        .ok_or_else(|| WebError::web_api("Document", JsValue::from_str("no document available")))?;
    let input = document
        .get_element_by_id(input_id)
        .ok_or_else(|| WebError::developer_model("the file picker is not on the page"))?
        .dyn_into::<HtmlInputElement>()
        .map_err(|e| WebError::web_api("HtmlInputElement cast", e.into()))?;
    let Some(list) = input.files() else {
        return Ok(Vec::new());
    };
    Ok((0..list.length())
        .filter_map(|index| list.get(index))
        .map(|file| LocalFile {
            name: file.name(),
            file,
        })
        .collect())
}
