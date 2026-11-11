//! The URL the player loads audio from.

use crate::transcribe::chosen_file::ChosenFile;
use purr_common::history::entry::HistoryId;
use std::path::PathBuf;

/// Audio the player can load.
#[derive(Debug, Clone, PartialEq)]
pub enum AudioSource {
    /// The copy of a picked file the session holds.
    Upload(ChosenFile),
    /// The original file of a saved transcription, still on the computer.
    Original { id: HistoryId, path: PathBuf },
}

/// A URL the page can `fetch` the audio of `source` from; release it with [`release`].
#[cfg(purr_web)]
pub fn url(source: &AudioSource) -> miette::Result<String> {
    use miette::miette;
    use web_sys::{Blob, BlobPropertyBag, Url};

    let AudioSource::Upload(file) = source else {
        return Err(miette!(
            "the browser does not keep the original audio of a saved transcription"
        ));
    };
    let options = BlobPropertyBag::new();
    if let Some(content_type) = &file.content_type {
        options.set_type(content_type);
    }
    let parts = js_sys::Array::new();
    parts.push(&js_sys::Uint8Array::from(file.bytes.as_ref()));
    let blob = Blob::new_with_u8_array_sequence_and_options(&parts, &options)
        .map_err(|e| miette!("Could not hand the audio to the player: {e:?}"))?;
    Url::create_object_url_with_blob(&blob)
        .map_err(|e| miette!("Could not hand the audio to the player: {e:?}"))
}

/// Frees a URL made by [`url`].
#[cfg(purr_web)]
pub fn release(url: &str) {
    if let Err(error) = web_sys::Url::revoke_object_url(url) {
        tracing::warn!("Could not release the audio for playback: {error:?}");
    }
}

/// The first path segment the desktop webview requests audio under.
#[cfg(all(feature = "desktop", not(purr_web)))]
pub const AUDIO_ROUTE: &str = "purr-audio";

/// A URL the page can `fetch` the audio of `source` from, served by [`use_audio_route`].
#[cfg(all(feature = "desktop", not(purr_web)))]
pub fn url(source: &AudioSource) -> miette::Result<String> {
    Ok(match source {
        AudioSource::Upload(file) => format!("/{AUDIO_ROUTE}/{}", file.id),
        AudioSource::Original { id, .. } => format!("/{AUDIO_ROUTE}/{id}"),
    })
}

/// Nothing to free on the desktop: the route serves the files.
#[cfg(all(feature = "desktop", not(purr_web)))]
pub fn release(_url: &str) {}

/// Serves the audio of the session's files and of saved transcriptions to the webview; call once
/// at the root.
#[cfg(all(feature = "desktop", not(purr_web)))]
pub fn use_audio_route(session: crate::transcribe::session::TranscribeSession) {
    use crate::utils::file_kind::audio_mime;
    use dioxus::desktop::wry::http::{Response, StatusCode, header::CONTENT_TYPE};

    dioxus::desktop::use_asset_handler(AUDIO_ROUTE, move |request, responder| {
        let id = request
            .uri()
            .path()
            .rsplit('/')
            .next()
            .unwrap_or_default()
            .to_owned();
        let found: Option<(String, Vec<u8>)> = if let Some(file) = session.file_by_id(&id) {
            Some((
                file.content_type
                    .filter(|kind| !kind.is_empty())
                    .unwrap_or_else(|| "application/octet-stream".to_owned()),
                file.bytes.to_vec(),
            ))
        } else {
            session
                .original_path(&id)
                .and_then(|path| match std::fs::read(&path) {
                    Ok(bytes) => {
                        let name = path
                            .file_name()
                            .map_or_else(String::new, |name| name.to_string_lossy().into_owned());
                        Some((audio_mime(&name).to_owned(), bytes))
                    }
                    Err(error) => {
                        tracing::error!(
                            "Could not read the original audio {}: {error}",
                            path.display()
                        );
                        None
                    }
                })
        };
        let response = match found {
            Some((content_type, body)) => Response::builder()
                .status(StatusCode::OK)
                .header(CONTENT_TYPE, content_type)
                .body(body),
            None => Response::builder()
                .status(StatusCode::NOT_FOUND)
                .body(Vec::new()),
        };
        match response {
            Ok(response) => responder.respond(response),
            Err(error) => tracing::error!("Could not answer the audio request for {id}: {error}"),
        }
    });
}

/// The browser plays blob URLs directly; there is no route to serve.
#[cfg(not(all(feature = "desktop", not(purr_web))))]
pub fn use_audio_route(_session: crate::transcribe::session::TranscribeSession) {}

/// Playback needs a desktop or web platform.
#[cfg(not(any(purr_web, feature = "desktop")))]
pub fn url(_source: &AudioSource) -> miette::Result<String> {
    Err(miette::miette!(
        "Playback is not supported on this platform"
    ))
}

/// Nothing to free without a platform.
#[cfg(not(any(purr_web, feature = "desktop")))]
pub fn release(_url: &str) {}
