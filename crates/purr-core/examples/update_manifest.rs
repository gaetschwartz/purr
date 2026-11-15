//! Refreshes `size_bytes` and `sha256` of the models and the voice activity model in
//! `models/manifest.json` from the Hugging Face tree API.
//!
//! Usage: `cargo run -p purr-core --example update_manifest -- [--bump] [MANIFEST_PATH]`
//!
//! Every other field of the manifest is preserved. Without `--bump` each source is queried at its
//! pinned revision; with `--bump` the revision is first advanced to the latest commit.

use oopsie::{Welp, WelpResultExt as _};
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
    fn parse() -> Result<Self, Welp> {
        let mut bump = false;
        let mut manifest_path = None;
        for arg in std::env::args().skip(1) {
            if arg == "--bump" {
                bump = true;
            } else if manifest_path.is_none() {
                manifest_path = Some(PathBuf::from(arg));
            } else {
                return Err(Welp::new(format!("unexpected argument: {arg}")));
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
) -> Result<T, Welp> {
    let response = client
        .get(url)
        .send()
        .await
        .welp()?
        .error_for_status()
        .with_welp_context(|_| format!("GET {url}"))?;
    if response.headers().contains_key(reqwest::header::LINK) {
        return Err(Welp::new(format!(
            "paginated response is not supported: {url}"
        )));
    }
    let body = response.bytes().await.welp()?;
    serde_json::from_slice(&body).with_welp_context(|_| format!("parsing response of {url}"))
}

async fn latest_revision(client: &reqwest::Client, repo: &str) -> Result<String, Welp> {
    let url = format!("{}/api/models/{repo}", Source::ORIGIN);
    Ok(get_json::<RepoInfo>(client, &url).await?.sha)
}

async fn fetch_tree(
    client: &reqwest::Client,
    source: &Source,
) -> Result<HashMap<String, TreeEntry>, Welp> {
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

/// Sets `size_bytes` and `sha256` from the tree entry of `filename`; true when either changed.
fn refresh(
    tree: &HashMap<String, TreeEntry>,
    source: &Source,
    id: &str,
    filename: &str,
    size_bytes: &mut u64,
    sha256: &mut Sha256Digest,
) -> Result<bool, Welp> {
    let entry = tree.get(filename).ok_or_else(|| {
        Welp::new(format!(
            "{filename} is not in {}@{}",
            source.repo, source.revision
        ))
    })?;
    let lfs = entry
        .lfs
        .as_ref()
        .ok_or_else(|| Welp::new(format!("{filename} is not an LFS file")))?;
    if lfs.size != entry.size {
        return Err(Welp::new(format!(
            "{filename}: LFS size {} differs from file size {}",
            lfs.size, entry.size
        )));
    }
    let digest: Sha256Digest = lfs.oid.parse().welp()?;
    let changed = *size_bytes != lfs.size || *sha256 != digest;
    if changed {
        println!("{id}: size {} sha256 {digest}", lfs.size);
    }
    *size_bytes = lfs.size;
    *sha256 = digest;
    Ok(changed)
}

#[tokio::main]
async fn main() -> Result<(), Welp> {
    let args = Args::parse()?;
    let text = std::fs::read_to_string(&args.manifest_path)
        .with_welp_context(|_| format!("reading {}", args.manifest_path.display()))?;
    let mut manifest = Manifest::parse(&text).welp()?;
    let client = purr_core::http::client_builder().welp()?.build().welp()?;

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
            changed += usize::from(refresh(
                &tree,
                source,
                &model.id,
                &model.filename,
                &mut model.size_bytes,
                &mut model.sha256,
            )?);
            listed.insert(model.filename.clone());
        }

        let vad = &mut manifest.vad;
        if vad.source == source.id {
            changed += usize::from(refresh(
                &tree,
                source,
                &vad.id,
                &vad.filename,
                &mut vad.size_bytes,
                &mut vad.sha256,
            )?);
            listed.insert(vad.filename.clone());
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

    let mut json = serde_json::to_string_pretty(&manifest).welp()?;
    json.push('\n');
    std::fs::write(&args.manifest_path, json)
        .with_welp_context(|_| format!("writing {}", args.manifest_path.display()))?;
    println!(
        "{} pinned files changed in {}",
        changed,
        args.manifest_path.display()
    );
    Ok(())
}
