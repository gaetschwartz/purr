//! Hands text to the user as a file or on the clipboard.

use dioxus::document;

/// A text file the user can save.
pub struct TextFile<'a> {
    pub file_name: &'a str,
    pub content_type: &'a str,
    pub content: &'a str,
}

/// Saves the file through the browser's download mechanism.
#[cfg(purr_web)]
pub async fn save_text_file(file: TextFile<'_>) -> miette::Result<()> {
    use miette::miette;
    use wasm_bindgen::JsCast as _;
    use web_sys::{Blob, BlobPropertyBag, HtmlElement, Url};

    let document = web_sys::window()
        .and_then(|window| window.document())
        .ok_or_else(|| miette!("The browser window is not available"))?;

    let options = BlobPropertyBag::new();
    options.set_type(file.content_type);
    let parts = js_sys::Array::new();
    parts.push(&file.content.into());
    let blob = Blob::new_with_str_sequence_and_options(&parts, &options)
        .map_err(|e| miette!("Could not create the download: {e:?}"))?;
    let url = Url::create_object_url_with_blob(&blob)
        .map_err(|e| miette!("Could not create the download link: {e:?}"))?;

    let link = document
        .create_element("a")
        .map_err(|e| miette!("Could not create the download link: {e:?}"))?
        .dyn_into::<HtmlElement>()
        .map_err(|e| miette!("Could not create the download link: {e:?}"))?;
    link.set_attribute("href", &url)
        .map_err(|e| miette!("Could not set the download link: {e:?}"))?;
    link.set_attribute("download", file.file_name)
        .map_err(|e| miette!("Could not name the download: {e:?}"))?;
    link.click();

    wasm_bindgen_futures::spawn_local(async move {
        gloo_timers::future::TimeoutFuture::new(10_000).await;
        if let Err(error) = Url::revoke_object_url(&url) {
            tracing::warn!("Could not release the download: {error:?}");
        }
    });
    Ok(())
}

/// Asks where to save the file and writes it there.
#[cfg(all(feature = "desktop", not(purr_web)))]
pub async fn save_text_file(file: TextFile<'_>) -> miette::Result<()> {
    use miette::IntoDiagnostic as _;

    let Some(handle) = rfd::AsyncFileDialog::new()
        .set_file_name(file.file_name)
        .save_file()
        .await
    else {
        return Ok(());
    };
    tokio::fs::write(handle.path(), file.content)
        .await
        .into_diagnostic()?;
    tracing::info!("File saved to: {}", handle.path().display());
    Ok(())
}

/// Saving files is not available without a desktop or web platform.
#[cfg(not(any(purr_web, feature = "desktop")))]
pub async fn save_text_file(_file: TextFile<'_>) -> miette::Result<()> {
    Err(miette::miette!(
        "Saving files is not supported on this platform"
    ))
}

/// Result of the clipboard script, as sent back by the page.
#[derive(serde::Deserialize)]
struct ClipboardOutcome {
    error: Option<String>,
}

/// Puts `text` on the system clipboard.
pub async fn copy_to_clipboard(text: &str) -> miette::Result<()> {
    use miette::{IntoDiagnostic as _, miette};

    let mut eval = document::eval(
        "const text = await dioxus.recv();
         try {
             await navigator.clipboard.writeText(text);
             dioxus.send({ error: null });
         } catch (error) {
             dioxus.send({ error: String(error) });
         }",
    );
    eval.send(text).into_diagnostic()?;
    let outcome: ClipboardOutcome = eval
        .recv()
        .await
        .map_err(|e| miette!("The clipboard is not available: {e}"))?;
    outcome.error.map_or(Ok(()), |error| {
        Err(miette!("The clipboard refused the text: {error}"))
    })
}
