//! Refreshes the file list, sizes and sha256 in `purr-common/manifests/web-models.json` from the
//! Hugging Face tree API.
//!
//! Usage: `cargo run -p purr-core --example update_web_manifest -- [--bump] [--reconvert] [MANIFEST_PATH]`
//!
//! Every other field of the manifest is preserved. Without `--bump` each model is queried at its
//! pinned revision; with `--bump` the revision is first advanced to the latest commit. LFS files
//! take their digest from the tree listing; plain files are downloaded and hashed.
//!
//! Each ONNX file is also downloaded, verified against its digest and run through the browser's
//! converter to record the size and digest of the graph and weights it stores. A file whose
//! digest did not change keeps its recorded conversion unless `--reconvert` is given.

use oopsie::{Welp, WelpResultExt as _};
use purr_common::web_model::digest::{FileDigest, Sha256Digest};
use purr_common::web_model::file_role::ModelFileRole;
use purr_common::web_model::manifest::{
    HUGGING_FACE_ORIGIN, SUPPORTED_SCHEMA, WebManifest, WebManifestConversion, WebManifestFile,
    file_url,
};
use purr_common::web_model::registry::WebModelRegistry;
use purr_core::model::digest::Sha256Hasher;
use purr_onnx_convert::tooling::{Measured, cache_dir, convert_file, fetch_verified};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::PathBuf;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "lowercase")]
enum EntryKind {
    File,
    Directory,
}

#[derive(Debug, Deserialize)]
struct TreeEntry {
    #[serde(rename = "type")]
    kind: EntryKind,
    path: String,
    size: u64,
    lfs: Option<LfsPointer>,
}

#[derive(Debug, Deserialize)]
struct LfsPointer {
    oid: String,
    size: u64,
}

#[derive(Debug, Deserialize)]
struct RepoInfo {
    sha: String,
}

struct Args {
    bump: bool,
    reconvert: bool,
    manifest_path: PathBuf,
}

impl Args {
    fn parse() -> Result<Self, Welp> {
        let mut bump = false;
        let mut reconvert = false;
        let mut manifest_path = None;
        for arg in std::env::args().skip(1) {
            if arg == "--bump" {
                bump = true;
            } else if arg == "--reconvert" {
                reconvert = true;
            } else if manifest_path.is_none() {
                manifest_path = Some(PathBuf::from(arg));
            } else {
                return Err(Welp::new(format!("unexpected argument: {arg}")));
            }
        }
        Ok(Self {
            bump,
            reconvert,
            manifest_path: manifest_path.unwrap_or_else(|| {
                PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("..")
                    .join("purr-common")
                    .join("manifests")
                    .join("web-models.json")
            }),
        })
    }
}

async fn get_response(client: &reqwest::Client, url: &str) -> Result<reqwest::Response, Welp> {
    client
        .get(url)
        .send()
        .await
        .welp()?
        .error_for_status()
        .with_welp_context(|_| format!("GET {url}"))
}

async fn get_json<T: for<'de> Deserialize<'de>>(
    client: &reqwest::Client,
    url: &str,
) -> Result<T, Welp> {
    let response = get_response(client, url).await?;
    if response.headers().contains_key(reqwest::header::LINK) {
        return Err(Welp::new(format!(
            "paginated response is not supported: {url}"
        )));
    }
    let body = response.bytes().await.welp()?;
    serde_json::from_slice(&body).with_welp_context(|_| format!("parsing response of {url}"))
}

async fn latest_revision(client: &reqwest::Client, repo: &str) -> Result<String, Welp> {
    let url = format!("{HUGGING_FACE_ORIGIN}/api/models/{repo}");
    Ok(get_json::<RepoInfo>(client, &url).await?.sha)
}

async fn fetch_tree(
    client: &reqwest::Client,
    repo: &str,
    revision: &str,
) -> Result<HashMap<String, TreeEntry>, Welp> {
    let url = format!("{HUGGING_FACE_ORIGIN}/api/models/{repo}/tree/{revision}?recursive=true");
    let entries: Vec<TreeEntry> = get_json(client, &url).await?;
    Ok(entries
        .into_iter()
        .filter(|entry| match entry.kind {
            EntryKind::File => true,
            EntryKind::Directory => false,
        })
        .map(|entry| (entry.path.clone(), entry))
        .collect())
}

async fn hash_plain_file(
    client: &reqwest::Client,
    repo: &str,
    revision: &str,
    entry: &TreeEntry,
) -> Result<Sha256Digest, Welp> {
    let url = file_url(repo, revision, &entry.path);
    let body = get_response(client, &url).await?.bytes().await.welp()?;
    if body.len() as u64 != entry.size {
        return Err(Welp::new(format!(
            "{}: downloaded {} bytes but the tree lists {}",
            entry.path,
            body.len(),
            entry.size
        )));
    }
    let mut hasher = Sha256Hasher::new();
    hasher.update(&body);
    hasher
        .finalize()
        .to_string()
        .parse()
        .with_welp_context(|_| format!("hashing {url}"))
}

async fn describe_file(
    client: &reqwest::Client,
    repo: &str,
    revision: &str,
    entry: &TreeEntry,
) -> Result<WebManifestFile, Welp> {
    let (size_bytes, sha256) = match &entry.lfs {
        Some(lfs) => {
            if lfs.size != entry.size {
                return Err(Welp::new(format!(
                    "{}: LFS size {} differs from file size {}",
                    entry.path, lfs.size, entry.size
                )));
            }
            let digest: Sha256Digest = lfs.oid.parse().welp()?;
            (lfs.size, digest)
        }
        None => (
            entry.size,
            hash_plain_file(client, repo, revision, entry).await?,
        ),
    };
    Ok(WebManifestFile {
        path: entry.path.clone(),
        size_bytes,
        sha256,
        converted: None,
    })
}

type Conversions = HashMap<(String, String, String), WebManifestConversion>;

fn parse_digest(measured: &Measured) -> Result<FileDigest, Welp> {
    Ok(FileDigest {
        size_bytes: measured.size_bytes,
        sha256: measured.sha256.parse().welp()?,
    })
}

async fn convert_onnx(
    repo: &str,
    revision: &str,
    file: &WebManifestFile,
) -> Result<WebManifestConversion, Welp> {
    let url = file_url(repo, revision, &file.path);
    let original = Measured {
        size_bytes: file.size_bytes,
        sha256: file.sha256.to_string(),
    };
    let destination = cache_dir()
        .join(repo.replace('/', "--"))
        .join(revision)
        .join(&file.path);
    tokio::task::spawn_blocking(move || {
        fetch_verified(&url, &original, &destination)?;
        let conversion = convert_file(&destination, None)?;
        Ok(WebManifestConversion {
            graph: parse_digest(&conversion.graph)?,
            data: parse_digest(&conversion.data)?,
        })
    })
    .await
    .welp()?
}

#[tokio::main]
async fn main() -> Result<(), Welp> {
    let args = Args::parse()?;
    let text = std::fs::read_to_string(&args.manifest_path)
        .with_welp_context(|_| format!("reading {}", args.manifest_path.display()))?;
    let mut manifest = WebManifest::parse(&text).welp()?;
    let client = purr_core::http::client_builder().welp()?.build().welp()?;

    let mut changed = 0usize;
    let mut conversions = Conversions::new();
    for model in &mut manifest.models {
        if args.bump {
            let latest = latest_revision(&client, &model.repo).await?;
            if latest != model.revision {
                println!("{}: revision {} -> {latest}", model.id, model.revision);
                model.revision = latest;
            }
        }

        let tree = fetch_tree(&client, &model.repo, &model.revision).await?;
        let mut files = Vec::with_capacity(ModelFileRole::ALL.len());
        for role in ModelFileRole::ALL {
            let path = role.path(model.encoder_dtype, model.decoder_dtype);
            let entry = tree.get(&path).ok_or_else(|| {
                Welp::new(format!(
                    "{path} is not in {}@{}",
                    model.repo, model.revision
                ))
            })?;
            let mut file = describe_file(&client, &model.repo, &model.revision, entry).await?;
            if role.is_onnx() {
                let key = (
                    model.repo.clone(),
                    model.revision.clone(),
                    file.path.clone(),
                );
                let recorded = model
                    .files
                    .iter()
                    .find(|previous| previous.path == file.path && previous.sha256 == file.sha256)
                    .and_then(|previous| previous.converted)
                    .filter(|_| !args.reconvert);
                let converted = match conversions.get(&key).copied().or(recorded) {
                    Some(converted) => converted,
                    None => {
                        println!("{}: converting {}", model.id, file.path);
                        convert_onnx(&model.repo, &model.revision, &file).await?
                    }
                };
                conversions.insert(key, converted);
                file.converted = Some(converted);
            }
            files.push(file);
        }

        if files != model.files {
            let total: u64 = files.iter().map(|file| file.size_bytes).sum();
            println!("{}: {} files, {total} bytes", model.id, files.len());
            changed += 1;
            model.files = files;
        }
    }

    manifest.schema = SUPPORTED_SCHEMA;
    let mut json = serde_json::to_string_pretty(&manifest).welp()?;
    json.push('\n');
    WebModelRegistry::from_json(&json).welp()?;
    std::fs::write(&args.manifest_path, json)
        .with_welp_context(|_| format!("writing {}", args.manifest_path.display()))?;
    println!(
        "{} models changed in {}",
        changed,
        args.manifest_path.display()
    );
    Ok(())
}
