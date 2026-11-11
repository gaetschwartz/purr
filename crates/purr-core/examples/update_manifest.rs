//! Refreshes `size_bytes` and `sha256` in `models/manifest.json` from the Hugging Face tree API.
//!
//! Usage: `cargo run -p purr-core --example update_manifest -- [--bump] [MANIFEST_PATH]`
//!
//! Every other field of the manifest is preserved. Without `--bump` each source is queried at its
//! pinned revision; with `--bump` the revision is first advanced to the latest commit.

use miette::{IntoDiagnostic as _, WrapErr as _, miette};
use purr_core::model::digest::Sha256Digest;
use purr_core::model::manifest::{Manifest, Source};
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

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
    manifest_path: PathBuf,
}

impl Args {
    fn parse() -> miette::Result<Self> {
        let mut bump = false;
        let mut manifest_path = None;
        for arg in std::env::args().skip(1) {
            if arg == "--bump" {
                bump = true;
            } else if manifest_path.is_none() {
                manifest_path = Some(PathBuf::from(arg));
            } else {
                return Err(miette!("unexpected argument: {arg}"));
            }
        }
        Ok(Self {
            bump,
            manifest_path: manifest_path.unwrap_or_else(|| {
                PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("models")
                    .join("manifest.json")
            }),
        })
    }
}

async fn get_json<T: for<'de> Deserialize<'de>>(
    client: &reqwest::Client,
    url: &str,
) -> miette::Result<T> {
    let response = client
        .get(url)
        .send()
        .await
        .into_diagnostic()?
        .error_for_status()
        .into_diagnostic()
        .wrap_err_with(|| format!("GET {url}"))?;
    if response.headers().contains_key(reqwest::header::LINK) {
        return Err(miette!("paginated response is not supported: {url}"));
    }
    let body = response.bytes().await.into_diagnostic()?;
    serde_json::from_slice(&body)
        .into_diagnostic()
        .wrap_err_with(|| format!("parsing response of {url}"))
}

async fn latest_revision(client: &reqwest::Client, repo: &str) -> miette::Result<String> {
    let url = format!("{}/api/models/{repo}", Source::ORIGIN);
    Ok(get_json::<RepoInfo>(client, &url).await?.sha)
}

async fn fetch_tree(
    client: &reqwest::Client,
    source: &Source,
) -> miette::Result<HashMap<String, TreeEntry>> {
    let url = format!(
        "{}/api/models/{}/tree/{}",
        Source::ORIGIN,
        source.repo,
        source.revision
    );
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

#[tokio::main]
async fn main() -> miette::Result<()> {
    let args = Args::parse()?;
    let text = std::fs::read_to_string(&args.manifest_path)
        .into_diagnostic()
        .wrap_err_with(|| format!("reading {}", args.manifest_path.display()))?;
    let mut manifest = Manifest::parse(&text)?;
    let client = purr_core::http::client_builder()?
        .build()
        .into_diagnostic()?;

    let mut changed = 0usize;
    for source in &mut manifest.sources {
        if args.bump {
            let latest = latest_revision(&client, &source.repo).await?;
            if latest != source.revision {
                println!("{}: revision {} -> {latest}", source.id, source.revision);
                source.revision = latest;
            }
        }

        let tree = fetch_tree(&client, source).await?;
        let mut listed = HashSet::new();

        for model in manifest
            .models
            .iter_mut()
            .filter(|model| model.source == source.id)
        {
            let entry = tree.get(&model.filename).ok_or_else(|| {
                miette!(
                    "{} is not in {}@{}",
                    model.filename,
                    source.repo,
                    source.revision
                )
            })?;
            let lfs = entry
                .lfs
                .as_ref()
                .ok_or_else(|| miette!("{} is not an LFS file", model.filename))?;
            if lfs.size != entry.size {
                return Err(miette!(
                    "{}: LFS size {} differs from file size {}",
                    model.filename,
                    lfs.size,
                    entry.size
                ));
            }
            let digest: Sha256Digest = lfs.oid.parse().into_diagnostic()?;
            if model.size_bytes != lfs.size || model.sha256 != digest {
                println!("{}: size {} sha256 {digest}", model.id, lfs.size);
                changed += 1;
            }
            model.size_bytes = lfs.size;
            model.sha256 = digest;
            listed.insert(model.filename.clone());
        }

        let mut unlisted: Vec<&str> = tree
            .keys()
            .filter(|path| {
                Path::new(path)
                    .extension()
                    .is_some_and(|ext| ext.eq_ignore_ascii_case("bin"))
                    && !listed.contains(*path)
            })
            .map(String::as_str)
            .collect();
        unlisted.sort_unstable();
        for path in unlisted {
            println!("{}: {path} is published but not in the manifest", source.id);
        }
    }

    let mut json = serde_json::to_string_pretty(&manifest).into_diagnostic()?;
    json.push('\n');
    std::fs::write(&args.manifest_path, json)
        .into_diagnostic()
        .wrap_err_with(|| format!("writing {}", args.manifest_path.display()))?;
    println!(
        "{} models changed in {}",
        changed,
        args.manifest_path.display()
    );
    Ok(())
}
