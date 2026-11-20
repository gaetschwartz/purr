mod common;

use common::TestResult;
use common::counting_allocator::{CountingAllocator, measure};
use purr_common::web_model::file_role::ModelFileRole;
use purr_common::web_model::registry::{WebModelFile, WebModelRegistry, WebModelSpec};
use purr_common::web_model::variant::{TableStorage, VariantKind};
use purr_onnx_convert::convert::{CHUNK_BYTES, TablePrecision};
use purr_onnx_convert::tooling::{Measured, cache_dir, convert_file, fetch_verified};
use std::path::{Path, PathBuf};
use std::process::Command;

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

const MODEL_ID: &str = "tiny";
const TINY_LAYERS: usize = 4;
const GRAPH_BUDGET_BYTES: usize = 8 * 1024 * 1024;

fn precision(table: TableStorage) -> TablePrecision {
    match table {
        TableStorage::Fp32 => TablePrecision::Full,
        TableStorage::Fp16 => TablePrecision::Half,
    }
}

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

fn variant(kind: VariantKind) -> Result<WebModelSpec, Box<dyn std::error::Error>> {
    let registry = WebModelRegistry::embedded()?;
    Ok(registry
        .get(MODEL_ID)?
        .variant(kind)
        .ok_or("the model lacks the variant")?
        .clone())
}

fn file_name(
    spec: &WebModelSpec,
    role: ModelFileRole,
) -> Result<String, Box<dyn std::error::Error>> {
    spec.files
        .iter()
        .find(|file| file.role == role)
        .and_then(|file| Path::new(&file.path).file_name())
        .and_then(|name| name.to_str())
        .map(str::to_owned)
        .ok_or_else(|| "the model lacks a file".into())
}

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

#[test]
fn real_model_conversion_gives_the_pinned_files_in_bounded_memory() -> TestResult {
    for kind in [VariantKind::Fp32, VariantKind::Fp16] {
        let spec = variant(kind)?;
        for file in onnx_files(&spec) {
            let original = fetch_original(&spec, file)?;
            let pinned = file.converted.ok_or("the manifest pins no conversion")?;
            let (conversion, usage) =
                measure(|| convert_file(&original, None, precision(spec.table)));
            let conversion = conversion?;
            let label = format!("{kind} {}", file.path);

            assert_eq!(
                conversion.graph.size_bytes, pinned.graph.size_bytes,
                "{label}"
            );
            assert_eq!(
                conversion.graph.sha256,
                pinned.graph.sha256.to_string(),
                "{label}"
            );
            assert_eq!(
                conversion.data.size_bytes, pinned.data.size_bytes,
                "{label}"
            );
            assert_eq!(
                conversion.data.sha256,
                pinned.data.sha256.to_string(),
                "{label}"
            );

            let is_decoder = file.role == ModelFileRole::DecoderMerged;
            assert_eq!(
                conversion.summary.tied_embedding_branches,
                usize::from(is_decoder) * 2,
                "{label}"
            );
            assert_eq!(
                conversion.summary.fused_attention_blocks,
                usize::from(file.role == ModelFileRole::Encoder) * TINY_LAYERS,
                "{label}"
            );
            let narrowed = is_decoder && spec.table == TableStorage::Fp16;
            assert_eq!(
                conversion.summary.half_precision_tables,
                usize::from(narrowed),
                "{label}"
            );
            assert!(
                usage.largest_allocation <= CHUNK_BYTES,
                "{label}: {usage:?}"
            );
            assert!(
                usage.peak_above_start <= GRAPH_BUDGET_BYTES,
                "{label}: peak {} for {} bytes of weights",
                usage.peak_above_start,
                conversion.data.size_bytes
            );
            println!(
                "{label}: graph {} data {} peak {} largest {}",
                conversion.graph.size_bytes,
                conversion.data.size_bytes,
                usage.peak_above_start,
                usage.largest_allocation
            );
        }
    }
    Ok(())
}

fn validate(
    spec: &WebModelSpec,
    original_dir: &Path,
    converted_dir: &Path,
    extra: &[&str],
) -> TestResult {
    let clip = root().join("samples").join("jfk.wav");
    let status = Command::new("uv")
        .arg("run")
        .arg(root().join("scripts").join("validate_onnx_conversion.py"))
        .arg(&spec.repo)
        .arg(original_dir)
        .arg(converted_dir)
        .arg(file_name(spec, ModelFileRole::Encoder)?)
        .arg(file_name(spec, ModelFileRole::DecoderMerged)?)
        .arg(&clip)
        .args(extra)
        .status()?;
    assert!(status.success(), "the ONNX Runtime comparison failed");
    Ok(())
}

#[test]
fn real_model_converted_files_give_the_logits_of_the_originals_in_onnx_runtime() -> TestResult {
    let spec = variant(VariantKind::Fp32)?;
    let converted_dir = cache_dir().join("converted").join(&spec.id);
    let mut original_dir = None;
    for file in onnx_files(&spec) {
        let original = fetch_original(&spec, file)?;
        convert_file(&original, Some(&converted_dir), precision(spec.table))?;
        original_dir = original.parent().map(Path::to_path_buf);
    }
    validate(
        &spec,
        &original_dir.ok_or("no original files")?,
        &converted_dir,
        &[],
    )
}

#[test]
fn real_model_half_table_decoder_agrees_with_the_original_in_onnx_runtime() -> TestResult {
    let spec = variant(VariantKind::Fp16)?;
    let converted_dir = cache_dir().join("converted-half").join(&spec.id);
    let decoder = onnx_files(&spec)
        .find(|file| file.role == ModelFileRole::DecoderMerged)
        .ok_or("no decoder")?;
    let original = fetch_original(&spec, decoder)?;
    let full_spec = variant(VariantKind::Fp32)?;
    let encoder = onnx_files(&full_spec)
        .find(|file| file.role == ModelFileRole::Encoder)
        .ok_or("no encoder")?;
    let original_encoder = fetch_original(&full_spec, encoder)?;
    assert_eq!(original.parent(), original_encoder.parent());
    convert_file(&original, Some(&converted_dir), TablePrecision::Half)?;
    let original_dir = original.parent().ok_or("no original directory")?;
    validate(&full_spec, original_dir, &converted_dir, &["half-table"])
}
