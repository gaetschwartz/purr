//! Puts a developer model's files into the cache the transcription worker reads, without a
//! manifest: nothing is hashed against anything, and ONNX files are converted only when asked.

use crate::byte_stream::next_chunk;
use crate::developer_model::{cached_url, data_path, path};
use crate::error::{WebError, WebResult};
use crate::integrity::FileChecker;
use crate::model_cache::open_for_writing;
use crate::model_install::{
    Conversion as ConversionJob, InstallProgress, Reporter, convert_into_cache, entry_headers,
    open_download, put_bytes,
};
use purr_common::developer::model::{
    BrowserModel, BrowserOrigin, Conversion, DeveloperModel, DeveloperModelId,
};
use purr_common::hugging_face_token::HuggingFaceToken;
use purr_common::web_model::file_role::ModelFileRole;
use purr_common::web_model::manifest::{HUGGING_FACE_ORIGIN, file_url};
use serde::Deserialize;
use wasm_bindgen::{JsCast as _, JsValue};
use wasm_bindgen_futures::JsFuture;
use web_sys::{
    Cache, File, Headers, ReadableStreamDefaultReader, RequestInit, Response, TransformStream,
};

/// A file the user picked on their own disk.
pub struct LocalFile {
    pub name: String,
    pub file: File,
}

#[derive(Deserialize)]
struct Sibling {
    rfilename: String,
}

#[derive(Deserialize)]
struct RepoListing {
    siblings: Vec<Sibling>,
}

async fn repo_files(
    repo: &str,
    revision: &str,
    token: Option<&HuggingFaceToken>,
) -> WebResult<Vec<String>> {
    let url = format!("{HUGGING_FACE_ORIGIN}/api/models/{repo}/revision/{revision}");
    let init = RequestInit::new();
    if let Some(token) = token {
        let headers = Headers::new().map_err(|e| WebError::web_api("Headers.new", e))?;
        headers
            .set("Authorization", &format!("Bearer {}", token.expose()))
            .map_err(|e| WebError::web_api("Headers.set", e))?;
        init.set_headers(&headers);
    }
    let window = web_sys::window()
        .ok_or_else(|| WebError::web_api("Window", JsValue::from_str("no window available")))?;
    let response = JsFuture::from(window.fetch_with_str_and_init(&url, &init))
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
    let body = JsFuture::from(
        response
            .json()
            .map_err(|e| WebError::web_api("Response.json", e))?,
    )
    .await
    .map_err(|e| WebError::web_api("Response.json", e))?;
    let listing: RepoListing = serde_wasm_bindgen::from_value(body).map_err(|e| {
        WebError::developer_model(format!("the repository listing is unreadable: {e}"))
    })?;
    Ok(listing.siblings.into_iter().map(|s| s.rfilename).collect())
}

async fn already_cached(cache: &Cache, url: &str) -> WebResult<bool> {
    let hit = JsFuture::from(cache.match_with_str(url))
        .await
        .map_err(|e| WebError::web_api("Cache.match", e))?;
    Ok(!hit.is_undefined())
}

async fn drop_entry(cache: &Cache, url: &str) {
    if let Err(error) = JsFuture::from(cache.delete_with_str(url)).await {
        tracing::warn!("Could not drop the partial {url}: {error:?}");
    }
}

async fn stream_into_cache(
    cache: &Cache,
    url: &str,
    reader: &ReadableStreamDefaultReader,
    size: Option<u64>,
    path: &str,
    reporter: &mut Reporter<'_>,
) -> WebResult<()> {
    let stream = TransformStream::new().map_err(|e| WebError::web_api("TransformStream.new", e))?;
    let writer = stream
        .writable()
        .get_writer()
        .map_err(|e| WebError::web_api("WritableStream.getWriter", e))?;
    let response = Response::new_with_opt_readable_stream_and_init(
        Some(&stream.readable()),
        &entry_headers(size)?,
    )
    .map_err(|e| WebError::web_api("Response.new", e))?;
    let put = JsFuture::from(cache.put_with_str(url, &response));
    let mut received = 0_u64;
    let copied = async {
        while let Some(chunk) = next_chunk(reader).await? {
            JsFuture::from(writer.write_with_chunk(&chunk))
                .await
                .map_err(|e| WebError::web_api("WritableStreamDefaultWriter.write", e))?;
            received = received.saturating_add(u64::from(chunk.length()));
            reporter.receiving(path, received);
        }
        JsFuture::from(writer.close())
            .await
            .map_err(|e| WebError::web_api("WritableStreamDefaultWriter.close", e))?;
        Ok::<(), WebError>(())
    }
    .await;
    if let Err(error) = copied {
        if JsFuture::from(writer.abort_with_reason(&JsValue::from_str(&error.to_string())))
            .await
            .is_err()
        {
            tracing::debug!("The data stream was already closed");
        }
        if put.await.is_ok() {
            tracing::debug!("The cache kept an entry of a copy that failed");
        }
        return Err(error);
    }
    put.await.map_err(|e| WebError::web_api("Cache.put", e))?;
    Ok(())
}

/// Where one file's bytes come from.
enum Source {
    Hugging {
        url: String,
        token: Option<HuggingFaceToken>,
    },
    Local(File),
}

impl Source {
    async fn open(&self) -> WebResult<(ReadableStreamDefaultReader, Option<u64>)> {
        match self {
            Self::Hugging { url, token } => Ok((open_download(url, token.as_ref()).await?, None)),
            Self::Local(file) => {
                let reader = file
                    .stream()
                    .get_reader()
                    .dyn_into::<ReadableStreamDefaultReader>()
                    .map_err(|e| WebError::web_api("ReadableStream.getReader", e.into()))?;
                Ok((reader, Some(file.size() as u64)))
            }
        }
    }
}

async fn install_file(
    cache: &Cache,
    id: &DeveloperModelId,
    model: &BrowserModel,
    role: ModelFileRole,
    source: &Source,
    reporter: &mut Reporter<'_>,
) -> WebResult<()> {
    let file = path(role, model);
    let (reader, size) = source.open().await?;
    if let (true, Some(table)) = (role.is_onnx(), model.conversion.table()) {
        return convert_into_cache(
            cache,
            &reader,
            ConversionJob {
                path: &file,
                table,
                graph_url: cached_url(id, &file),
                data_url: cached_url(id, &data_path(&file)),
                data_size: None,
                original: FileChecker::unchecked(),
                data: FileChecker::unchecked(),
                graph: FileChecker::unchecked(),
            },
            reporter,
        )
        .await;
    }
    if role.is_onnx() {
        return stream_into_cache(
            cache,
            &cached_url(id, &file),
            &reader,
            size,
            &file,
            reporter,
        )
        .await;
    }
    let mut bytes = Vec::new();
    while let Some(chunk) = next_chunk(&reader).await? {
        bytes.extend_from_slice(&chunk.to_vec());
    }
    put_bytes(cache, &cached_url(id, &file), bytes).await
}

async fn install_role(
    cache: &Cache,
    id: &DeveloperModelId,
    model: &BrowserModel,
    role: ModelFileRole,
    source: Source,
    external: Option<Source>,
    reporter: &mut Reporter<'_>,
) -> WebResult<()> {
    let file = path(role, model);
    let mut written = vec![cached_url(id, &file)];
    let outcome = async {
        install_file(cache, id, model, role, &source, reporter).await?;
        if let Some(external) = external {
            let data = data_path(&file);
            let (reader, size) = external.open().await?;
            written.push(cached_url(id, &data));
            stream_into_cache(
                cache,
                &cached_url(id, &data),
                &reader,
                size,
                &data,
                reporter,
            )
            .await?;
        }
        Ok::<(), WebError>(())
    }
    .await;
    if outcome.is_err() {
        for url in &written {
            drop_entry(cache, url).await;
        }
        if role.is_onnx() && model.conversion.table().is_some() {
            drop_entry(cache, &cached_url(id, &data_path(&file))).await;
        }
    }
    outcome?;
    reporter.installed(&file);
    Ok(())
}

/// Downloads a developer model from its Hugging Face repository.
pub async fn install(
    model: &DeveloperModel,
    browser: &BrowserModel,
    token: Option<&HuggingFaceToken>,
    report: &mut dyn FnMut(InstallProgress<'_>),
) -> WebResult<()> {
    let BrowserOrigin::HuggingFace { repo, revision } = &browser.origin else {
        return Err(WebError::developer_model(
            "this model's files were picked locally and are no longer in the browser; add them again",
        ));
    };
    let cache = open_for_writing().await?;
    let listing = repo_files(repo, revision, token).await?;
    let mut reporter = Reporter {
        report,
        last_ms: 0.0,
    };
    for role in ModelFileRole::ALL {
        let file = path(role, browser);
        let required = role != ModelFileRole::GenerationConfig;
        if !listing.contains(&file) {
            if required {
                return Err(WebError::developer_model(format!(
                    "{repo} at {revision} has no {file}"
                )));
            }
            continue;
        }
        let url = cached_url(&model.id, &file);
        let data_listed = listing.contains(&data_path(&file));
        let data_cached = already_cached(&cache, &cached_url(&model.id, &data_path(&file))).await?;
        let converted = role.is_onnx() && browser.conversion.table().is_some();
        if already_cached(&cache, &url).await? && (!data_listed || data_cached || converted) {
            reporter.installed(&file);
            continue;
        }
        if converted && data_listed {
            return Err(WebError::developer_model(format!(
                "{file} keeps its weights in a separate file, which the streaming converter does not read; load it as it is"
            )));
        }
        let external = (role.is_onnx() && !converted && data_listed).then(|| Source::Hugging {
            url: file_url(repo, revision, &data_path(&file)),
            token: token.cloned(),
        });
        install_role(
            &cache,
            &model.id,
            browser,
            role,
            Source::Hugging {
                url: file_url(repo, revision, &file),
                token: token.cloned(),
            },
            external,
            &mut reporter,
        )
        .await?;
    }
    Ok(())
}

/// Stores the files a user picked, which the browser then keeps for the model.
pub async fn install_local(
    model: &DeveloperModel,
    browser: &BrowserModel,
    files: Vec<LocalFile>,
    report: &mut dyn FnMut(InstallProgress<'_>),
) -> WebResult<()> {
    let cache = open_for_writing().await?;
    let mut reporter = Reporter {
        report,
        last_ms: 0.0,
    };
    let find = |wanted: &str| files.iter().find(|file| file.name == wanted);
    let mut missing = Vec::new();
    for role in ModelFileRole::ALL {
        let expected = path(role, browser);
        let name = expected
            .rsplit_once('/')
            .map_or(expected.as_str(), |(_, name)| name);
        let Some(picked) = find(name).or_else(|| matching_onnx(&files, role)) else {
            if role != ModelFileRole::GenerationConfig {
                missing.push(name.to_owned());
            }
            continue;
        };
        let external = if role.is_onnx() && browser.conversion == Conversion::AsIs {
            let data_name = data_path(&picked.name);
            find(&data_name).map(|data| Source::Local(data.file.clone()))
        } else {
            None
        };
        install_role(
            &cache,
            &model.id,
            browser,
            role,
            Source::Local(picked.file.clone()),
            external,
            &mut reporter,
        )
        .await?;
    }
    if missing.is_empty() {
        Ok(())
    } else {
        Err(WebError::developer_model(format!(
            "pick these files too: {}",
            missing.join(", ")
        )))
    }
}

/// The picked ONNX file of a role whatever dtype suffix it carries.
fn matching_onnx(files: &[LocalFile], role: ModelFileRole) -> Option<&LocalFile> {
    let stem = match role {
        ModelFileRole::Encoder => "encoder_model",
        ModelFileRole::DecoderMerged => "decoder_model_merged",
        ModelFileRole::Config
        | ModelFileRole::GenerationConfig
        | ModelFileRole::PreprocessorConfig
        | ModelFileRole::Tokenizer
        | ModelFileRole::TokenizerConfig => return None,
    };
    let mut matches = files.iter().filter(|file| {
        file.name.starts_with(stem)
            && std::path::Path::new(&file.name)
                .extension()
                .is_some_and(|extension| extension.eq_ignore_ascii_case("onnx"))
    });
    let first = matches.next()?;
    matches.next().is_none().then_some(first)
}
