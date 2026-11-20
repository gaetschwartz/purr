mod common;

use common::TestResult;
use common::counting_allocator::{CountingAllocator, measure};
use purr_common::web_model::file_role::ModelFileRole;
use purr_common::web_model::registry::{WebModelFile, WebModelRegistry, WebModelSpec};
use purr_onnx_convert::convert::CHUNK_BYTES;
use purr_onnx_convert::tooling::{Measured, cache_dir, convert_file, fetch_verified};
use std::path::{Path, PathBuf};
use std::process::Command;

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

const MODEL_ID: &str = "tiny";
const GRAPH_BUDGET_BYTES: usize = 8 * 1024 * 1024;

fn onnx_files(spec: &WebModelSpec) -> impl Iterator<Item = &WebModelFile> {
    spec.files.iter().filter(|file| file.role.is_onnx())
}

fn original_path(spec: &WebModelSpec, file: &WebModelFile) -> PathBuf {
    cache_dir()
        .join(spec.repo.replace('/', "--"))
        .join(&spec.revision)
        .join(&file.path)
}

fn fetch_original(
    spec: &WebModelSpec,
    file: &WebModelFile,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let path = original_path(spec, file);
    fetch_verified(
        &spec.file_url(file),
        &Measured {
            size_bytes: file.size_bytes,
            sha256: file.sha256.to_string(),
        },
        &path,
    )?;
    Ok(path)
}

#[test]
fn real_model_conversion_gives_the_pinned_files_in_bounded_memory() -> TestResult {
    let registry = WebModelRegistry::embedded()?;
    let spec = registry.get(MODEL_ID)?;
    for file in onnx_files(spec) {
        let original = fetch_original(spec, file)?;
        let pinned = file.converted.ok_or("the manifest pins no conversion")?;
        let (conversion, usage) = measure(|| convert_file(&original, None));
        let conversion = conversion?;

        assert_eq!(
            conversion.graph.size_bytes, pinned.graph.size_bytes,
            "{}",
            file.path
        );
        assert_eq!(
            conversion.graph.sha256,
            pinned.graph.sha256.to_string(),
            "{}",
            file.path
        );
        assert_eq!(
            conversion.data.size_bytes, pinned.data.size_bytes,
            "{}",
            file.path
        );
        assert_eq!(
            conversion.data.sha256,
            pinned.data.sha256.to_string(),
            "{}",
            file.path
        );

        let expected_branches = usize::from(file.role == ModelFileRole::DecoderMerged) * 2;
        assert_eq!(
            conversion.summary.tied_embedding_branches, expected_branches,
            "{}",
            file.path
        );
        assert!(
            usage.largest_allocation <= CHUNK_BYTES,
            "{}: {usage:?}",
            file.path
        );
        assert!(
            usage.peak_above_start <= GRAPH_BUDGET_BYTES,
            "{}: peak {} for {} bytes of weights",
            file.path,
            usage.peak_above_start,
            conversion.data.size_bytes
        );
        println!(
            "{}: graph {} data {} peak {} largest {}",
            file.path,
            conversion.graph.size_bytes,
            conversion.data.size_bytes,
            usage.peak_above_start,
            usage.largest_allocation
        );
    }
    Ok(())
}

#[test]
fn real_model_converted_files_give_the_logits_of_the_originals_in_onnx_runtime() -> TestResult {
    let registry = WebModelRegistry::embedded()?;
    let spec = registry.get(MODEL_ID)?;
    let converted_dir = cache_dir().join("converted").join(&spec.id);
    let mut originals: Vec<PathBuf> = Vec::new();
    for file in onnx_files(spec) {
        let original = fetch_original(spec, file)?;
        convert_file(&original, Some(&converted_dir))?;
        originals.push(original);
    }
    let original_dir = originals
        .first()
        .and_then(|path| path.parent())
        .ok_or("no original files")?;
    let name = |role| -> Result<String, Box<dyn std::error::Error>> {
        spec.files
            .iter()
            .find(|file| file.role == role)
            .and_then(|file| Path::new(&file.path).file_name())
            .and_then(|name| name.to_str())
            .map(str::to_owned)
            .ok_or_else(|| "the model lacks a file".into())
    };
    let script = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("scripts")
        .join("validate_onnx_conversion.py");
    let clip = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("samples")
        .join("jfk.wav");
    let status = Command::new("uv")
        .arg("run")
        .arg(&script)
        .arg(&spec.repo)
        .arg(original_dir)
        .arg(&converted_dir)
        .arg(name(ModelFileRole::Encoder)?)
        .arg(name(ModelFileRole::DecoderMerged)?)
        .arg(&clip)
        .status()?;
    assert!(status.success(), "the ONNX Runtime comparison failed");
    Ok(())
}
