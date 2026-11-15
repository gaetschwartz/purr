//! Hands text to the user as a file or on the clipboard.

use dioxus::document;

/// Why a file or the clipboard could not be handed to the user.
#[oopsie::oopsie]
pub enum ExportError {
    #[oopsie(
        "The browser window is not available",
        code = "app::export::window_unavailable"
    )]
    WindowUnavailable,
    #[oopsie(
        "Could not create the download: {reason}",
        code = "app::export::create_download"
    )]
    CreateDownload { reason: String },
    #[oopsie(
        "Could not create the download link: {reason}",
        code = "app::export::create_download_url"
    )]
    CreateDownloadUrl { reason: String },
    #[oopsie(
        "Could not create the download link element: {reason}",
        code = "app::export::create_link_element"
    )]
    CreateLinkElement { reason: String },
    #[oopsie(
        "The download link is not an HTML element: {reason}",
        code = "app::export::link_not_html"
    )]
    LinkNotHtml { reason: String },
    #[oopsie(
        "Could not set the download link: {reason}",
        code = "app::export::set_link_target"
    )]
    SetLinkTarget { reason: String },
    #[oopsie(
        "Could not name the download: {reason}",
        code = "app::export::name_download"
    )]
    NameDownload { reason: String },
    #[oopsie(
        "Could not write the file {}",
        path.display(),
        code = "app::export::write_file"
    )]
    WriteFile {
        path: std::path::PathBuf,
        source: std::io::Error,
    },
    #[oopsie(
        "Saving files is not supported on this platform",
        code = "app::export::save_unsupported"
    )]
    SaveUnsupported,
    #[oopsie(
        "The text could not be prepared for the clipboard",
        code = "app::export::encode_clipboard_text"
    )]
    EncodeClipboardText { source: serde_json::Error },
    #[oopsie(
        "The clipboard is not available",
        code = "app::export::clipboard_unavailable"
    )]
    ClipboardUnavailable { source: document::EvalError },
    #[oopsie(
        "The clipboard refused the text: {reason}",
        code = "app::export::clipboard_refused"
    )]
    ClipboardRefused { reason: String },
}

/// A text file the user can save.
pub struct TextFile<'a> {
    pub file_name: &'a str,
    pub content_type: &'a str,
    pub content: &'a str,
}

/// What became of a save request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SaveOutcome {
    Saved,
    Cancelled,
}

/// Saves the file through the browser's download mechanism.
#[cfg(purr_web)]
pub async fn save_text_file(file: TextFile<'_>) -> Result<SaveOutcome, ExportError> {
    use wasm_bindgen::JsCast as _;
    use web_sys::{Blob, BlobPropertyBag, HtmlElement, Url};

    let document = web_sys::window()
        .and_then(|window| window.document())
        .ok_or(ExportError::WindowUnavailable)?;

    let options = BlobPropertyBag::new();
    options.set_type(file.content_type);
    let parts = js_sys::Array::new();
    parts.push(&file.content.into());
    let blob = Blob::new_with_str_sequence_and_options(&parts, &options).map_err(|e| {
        ExportError::CreateDownload {
            reason: format!("{e:?}"),
        }
    })?;
    let url =
        Url::create_object_url_with_blob(&blob).map_err(|e| ExportError::CreateDownloadUrl {
            reason: format!("{e:?}"),
        })?;

    let link = document
        .create_element("a")
        .map_err(|e| ExportError::CreateLinkElement {
            reason: format!("{e:?}"),
        })?
        .dyn_into::<HtmlElement>()
        .map_err(|e| ExportError::LinkNotHtml {
            reason: format!("{e:?}"),
        })?;
    link.set_attribute("href", &url)
        .map_err(|e| ExportError::SetLinkTarget {
            reason: format!("{e:?}"),
        })?;
    link.set_attribute("download", file.file_name)
        .map_err(|e| ExportError::NameDownload {
            reason: format!("{e:?}"),
        })?;
    link.click();

    wasm_bindgen_futures::spawn_local(async move {
        gloo_timers::future::TimeoutFuture::new(10_000).await;
        if let Err(error) = Url::revoke_object_url(&url) {
            tracing::warn!("Could not release the download: {error:?}");
        }
    });
    Ok(SaveOutcome::Saved)
}

/// Asks where to save the file and writes it there; closing the dialog cancels.
#[cfg(all(feature = "desktop", not(purr_web)))]
pub async fn save_text_file(file: TextFile<'_>) -> Result<SaveOutcome, ExportError> {
    let Some(handle) = rfd::AsyncFileDialog::new()
        .set_file_name(file.file_name)
        .save_file()
        .await
    else {
        return Ok(SaveOutcome::Cancelled);
    };
    tokio::fs::write(handle.path(), file.content)
        .await
        .map_err(|source| ExportError::WriteFile {
            path: handle.path().to_path_buf(),
            source,
        })?;
    tracing::info!("File saved to: {}", handle.path().display());
    Ok(SaveOutcome::Saved)
}

/// Saving files is not available without a desktop or web platform.
#[cfg(not(any(purr_web, feature = "desktop")))]
pub async fn save_text_file(_file: TextFile<'_>) -> Result<SaveOutcome, ExportError> {
    Err(ExportError::SaveUnsupported)
}

/// Result of the clipboard script, as sent back by the page.
#[derive(serde::Deserialize)]
struct ClipboardOutcome {
    error: Option<String>,
}

/// Puts `text` on the system clipboard; the returned future reports how it went.
///
/// The page script starts right away, so call this from an event handler rather than from inside
/// a spawned task: the desktop webview ends scripts started from a task without a live scope.
pub fn copy_to_clipboard(text: &str) -> impl Future<Output = Result<(), ExportError>> + use<> {
    let started = serde_json::to_string(text)
        .map_err(|source| ExportError::EncodeClipboardText { source })
        .map(|literal| {
            document::eval(&format!(
                "const text = {literal};
             const legacyCopy = () => {{
                 const area = document.createElement('textarea');
                 area.value = text;
                 area.setAttribute('readonly', '');
                 area.style.position = 'fixed';
                 area.style.opacity = '0';
                 document.body.appendChild(area);
                 area.select();
                 const copied = document.execCommand('copy');
                 area.remove();
                 if (!copied) throw new Error('the page may not write to the clipboard');
             }};
             try {{
                 if (navigator.clipboard && navigator.clipboard.writeText) {{
                     await navigator.clipboard.writeText(text);
                 }} else {{
                     legacyCopy();
                 }}
                 dioxus.send({{ error: null }});
             }} catch (error) {{
                 dioxus.send({{ error: String(error) }});
             }}"
            ))
        });
    async move {
        let mut eval = started?;
        let outcome: ClipboardOutcome = eval
            .recv()
            .await
            .map_err(|source| ExportError::ClipboardUnavailable { source })?;
        outcome.error.map_or(Ok(()), |reason| {
            Err(ExportError::ClipboardRefused { reason })
        })
    }
}
