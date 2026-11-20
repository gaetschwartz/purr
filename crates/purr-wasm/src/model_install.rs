//! Installs a model into the cache transformers.js reads: every file is downloaded here, checked
//! against the manifest while it streams in, and ONNX files are converted on the way through a
//! worker, so the cache only ever holds files that are verified and ready to load.

use crate::byte_stream::{body_reader, next_chunk};
use crate::converter_worker::{ConverterEvent, ConverterOutput, Sink, start};
use crate::error::{WebError, WebResult};
use crate::integrity::FileChecker;
use crate::model_cache::{has_file, open_for_writing};
use futures::join;
use js_sys::Uint8Array;
use purr_common::hugging_face_token::HuggingFaceToken;
use purr_common::web_model::registry::{CachedFile, WebModelFile, WebModelSpec};
use wasm_bindgen::{JsCast as _, JsValue};
use wasm_bindgen_futures::JsFuture;
use web_sys::{
    Cache, Headers, ReadableStreamDefaultReader, RequestInit, Response, ResponseInit,
    TransformStream, WritableStreamDefaultWriter,
};

const PROGRESS_INTERVAL_MS: f64 = 100.0;

/// What an install reports while it runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InstallProgress<'a> {
    /// Bytes of the original file received so far.
    Receiving {
        path: &'a str,
        loaded_bytes: u64,
    },
    Installed {
        path: &'a str,
    },
}

struct Reporter<'a> {
    report: &'a mut dyn FnMut(InstallProgress<'_>),
    last_ms: f64,
}

impl Reporter<'_> {
    fn receiving(&mut self, path: &str, loaded_bytes: u64) {
        let now = js_sys::Date::now();
        if now - self.last_ms >= PROGRESS_INTERVAL_MS {
            self.last_ms = now;
            (self.report)(InstallProgress::Receiving { path, loaded_bytes });
        }
    }

    fn installed(&mut self, path: &str) {
        (self.report)(InstallProgress::Installed { path });
    }
}

/// Downloads, verifies, converts and caches every file of `spec` the cache does not hold yet.
/// A file that fails leaves nothing of itself in the cache.
pub async fn install(
    spec: &WebModelSpec,
    token: Option<&HuggingFaceToken>,
    report: &mut dyn FnMut(InstallProgress<'_>),
) -> WebResult<()> {
    let cache = open_for_writing().await?;
    let mut reporter = Reporter {
        report,
        last_ms: 0.0,
    };
    for file in &spec.files {
        if has_file(&cache, spec, file).await? {
            reporter.installed(&file.path);
            continue;
        }
        let outcome = install_file(&cache, spec, file, token, &mut reporter).await;
        if outcome.is_err() {
            for url in spec.cached_urls(file) {
                if let Err(error) = JsFuture::from(cache.delete_with_str(&url)).await {
                    tracing::warn!("Could not drop the partial {url}: {error:?}");
                }
            }
        }
        outcome?;
        reporter.installed(&file.path);
    }
    Ok(())
}

async fn install_file(
    cache: &Cache,
    spec: &WebModelSpec,
    file: &WebModelFile,
    token: Option<&HuggingFaceToken>,
    reporter: &mut Reporter<'_>,
) -> WebResult<()> {
    let url = spec.file_url(file);
    let reader = open_download(&url, token).await?;
    if file.converted.is_some() {
        install_converted(cache, spec, file, &reader, reporter).await
    } else {
        install_plain(cache, spec, file, &reader, reporter).await
    }
}

async fn open_download(
    url: &str,
    token: Option<&HuggingFaceToken>,
) -> WebResult<ReadableStreamDefaultReader> {
    let init = RequestInit::new();
    if let Some(token) = token {
        let headers = Headers::new().map_err(|e| WebError::web_api("Headers.new", e))?;
        headers
            .set("Authorization", &format!("Bearer {}", token.expose()))
            .map_err(|e| WebError::web_api("Headers.set", e))?;
        init.set_headers(&headers);
    }
    let window = web_sys::window().ok_or_else(|| {
        WebError::web_api("Window", JsValue::from_str("no window object available"))
    })?;
    let response = JsFuture::from(window.fetch_with_str_and_init(url, &init))
        .await
        .map_err(|e| WebError::web_api("fetch", e))?
        .dyn_into::<Response>()
        .map_err(|e| WebError::web_api("Response cast", e))?;
    if !response.ok() {
        return Err(WebError::network_error(
            response.status(),
            response.status_text(),
            url,
        ));
    }
    body_reader(&response)
}

fn entry_headers(size_bytes: u64) -> WebResult<ResponseInit> {
    let headers = Headers::new().map_err(|e| WebError::web_api("Headers.new", e))?;
    for (name, value) in [
        ("Content-Type", "application/octet-stream".to_owned()),
        ("Content-Length", size_bytes.to_string()),
    ] {
        headers
            .set(name, &value)
            .map_err(|e| WebError::web_api("Headers.set", e))?;
    }
    let init = ResponseInit::new();
    init.set_headers(&headers);
    Ok(init)
}

async fn put_bytes(cache: &Cache, url: &str, mut bytes: Vec<u8>) -> WebResult<()> {
    let init = entry_headers(bytes.len() as u64)?;
    let response = Response::new_with_opt_u8_array_and_init(Some(&mut bytes), &init)
        .map_err(|e| WebError::web_api("Response.new", e))?;
    JsFuture::from(cache.put_with_str(url, &response))
        .await
        .map_err(|e| WebError::web_api("Cache.put", e))?;
    Ok(())
}

async fn install_plain(
    cache: &Cache,
    spec: &WebModelSpec,
    file: &WebModelFile,
    reader: &ReadableStreamDefaultReader,
    reporter: &mut Reporter<'_>,
) -> WebResult<()> {
    let stored = file.cached_files();
    let [expected] = stored.as_slice() else {
        return Err(WebError::conversion(format!(
            "{} is not an ONNX file but lists converted files",
            file.path
        )));
    };
    let mut checker = FileChecker::new(expected);
    let mut bytes = Vec::with_capacity(usize::try_from(expected.size_bytes).unwrap_or_default());
    while let Some(chunk) = next_chunk(reader).await? {
        let piece = chunk.to_vec();
        checker.update(&piece);
        bytes.extend_from_slice(&piece);
        reporter.receiving(&file.path, checker.received_bytes());
    }
    checker.finish()?;
    put_bytes(cache, &spec.cached_url(expected), bytes).await
}

struct Drained {
    graph: Vec<u8>,
}

async fn drain(
    output: &mut ConverterOutput,
    writer: &WritableStreamDefaultWriter,
    data_checker: &mut FileChecker<'_>,
    graph_checker: &mut FileChecker<'_>,
) -> WebResult<Drained> {
    let mut graph = Vec::new();
    loop {
        match output.next().await {
            Some(ConverterEvent::Output { sink, bytes }) => {
                let piece = bytes.to_vec();
                match sink {
                    Sink::Data => {
                        data_checker.update(&piece);
                        JsFuture::from(writer.write_with_chunk(&bytes))
                            .await
                            .map_err(|e| {
                                WebError::web_api("WritableStreamDefaultWriter.write", e)
                            })?;
                    }
                    Sink::Graph => {
                        graph_checker.update(&piece);
                        graph.extend_from_slice(&piece);
                    }
                }
                output.acknowledge(piece.len() as u32)?;
            }
            Some(ConverterEvent::Finished) => return Ok(Drained { graph }),
            Some(ConverterEvent::Failed(message)) => return Err(WebError::conversion(message)),
            None => {
                return Err(WebError::conversion(
                    "The converter worker ended without finishing",
                ));
            }
        }
    }
}

async fn feed(
    input: &mut crate::converter_worker::ConverterInput,
    reader: &ReadableStreamDefaultReader,
    original: &CachedFile,
    path: &str,
    reporter: &mut Reporter<'_>,
) -> WebResult<()> {
    let mut checker = FileChecker::new(original);
    let fed = async {
        while let Some(chunk) = next_chunk(reader).await? {
            let piece: Uint8Array = chunk;
            let bytes = piece.to_vec();
            checker.update(&bytes);
            input.write(&bytes).await?;
            reporter.receiving(path, checker.received_bytes());
        }
        Ok::<(), WebError>(())
    }
    .await;
    let verified = fed.and_then(|()| checker.finish().map_err(WebError::from));
    match verified {
        Ok(()) => input.end(),
        Err(error) => {
            if let Err(abort_error) = input.abort() {
                tracing::warn!("Could not stop the converter: {abort_error}");
            }
            Err(error)
        }
    }
}

fn basename(path: &str) -> &str {
    path.rsplit_once('/').map_or(path, |(_, name)| name)
}

async fn install_converted(
    cache: &Cache,
    spec: &WebModelSpec,
    file: &WebModelFile,
    reader: &ReadableStreamDefaultReader,
    reporter: &mut Reporter<'_>,
) -> WebResult<()> {
    let stored = file.cached_files();
    let [graph_file, data_file] = stored.as_slice() else {
        return Err(WebError::conversion(format!(
            "{} lists no converted files",
            file.path
        )));
    };
    let original = CachedFile {
        path: file.path.clone(),
        size_bytes: file.size_bytes,
        sha256: file.sha256,
    };
    let (worker_url, wasm_url) = crate::converter_urls();
    let (mut input, mut output) = start(
        &worker_url,
        &wasm_url,
        &format!("{}{}", basename(&file.path), WebModelFile::DATA_SUFFIX),
        spec.table,
    )?;

    let stream = TransformStream::new().map_err(|e| WebError::web_api("TransformStream.new", e))?;
    let writer = stream
        .writable()
        .get_writer()
        .map_err(|e| WebError::web_api("WritableStream.getWriter", e))?;
    let data_url = spec.cached_url(data_file);
    let data_response = Response::new_with_opt_readable_stream_and_init(
        Some(&stream.readable()),
        &entry_headers(data_file.size_bytes)?,
    )
    .map_err(|e| WebError::web_api("Response.new", e))?;
    let data_put = JsFuture::from(cache.put_with_str(&data_url, &data_response));

    let mut data_checker = FileChecker::new(data_file);
    let mut graph_checker = FileChecker::new(graph_file);
    let (fed, drained) = join!(
        feed(&mut input, reader, &original, &file.path, reporter),
        async {
            let drained = drain(&mut output, &writer, &mut data_checker, &mut graph_checker).await;
            output.stop();
            drained
        }
    );

    let outcome = match (fed, drained) {
        (Err(error), _) | (Ok(()), Err(error)) => Err(error),
        (Ok(()), Ok(drained)) => data_checker
            .finish()
            .and_then(|()| graph_checker.finish())
            .map(|()| drained.graph)
            .map_err(WebError::from),
    };
    let graph = match outcome {
        Ok(graph) => graph,
        Err(error) => {
            if JsFuture::from(writer.abort_with_reason(&JsValue::from_str(&error.to_string())))
                .await
                .is_err()
            {
                tracing::debug!("The data stream was already closed");
            }
            if data_put.await.is_ok() {
                tracing::debug!("The cache kept the data of a conversion that failed");
            }
            return Err(error);
        }
    };
    JsFuture::from(writer.close())
        .await
        .map_err(|e| WebError::web_api("WritableStreamDefaultWriter.close", e))?;
    data_put
        .await
        .map_err(|e| WebError::web_api("Cache.put", e))?;
    put_bytes(cache, &spec.cached_url(graph_file), graph).await
}
