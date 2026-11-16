//! Integration tests for purr

use assert_cmd::Command;
use purr_core::ModelManager;
use purr_golden::Recording;
use rstest::rstest;
use serde_json::Value;
use std::{
    fmt,
    path::{Path, PathBuf},
    process::Output,
    time::Duration,
};

type TestResult<T = ()> = Result<T, Box<dyn std::error::Error>>;

struct MissingBaseModel {
    dir: PathBuf,
}

impl fmt::Debug for MissingBaseModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl fmt::Display for MissingBaseModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "the `base` model is missing or incomplete in {}: run `purr models download base` or `just download-test-models`",
            self.dir.display()
        )
    }
}

impl std::error::Error for MissingBaseModel {}

async fn base_model() -> TestResult<PathBuf> {
    let models = ModelManager::new()?;
    let spec = models.model("base")?;
    if !models.is_model_downloaded(spec).await? {
        return Err(MissingBaseModel {
            dir: models.models_dir().to_path_buf(),
        }
        .into());
    }
    Ok(models.get_model_path(spec))
}

/// The CLI on one thread, CPU only, with the `base` model.
async fn purr_on(sample: &Path, format: &OutputFormat) -> TestResult<Output> {
    Ok(Command::cargo_bin("purr")?
        .arg(sample)
        .args(["--no-stream", "--no-gpu", "--threads", "1"])
        .args(["--output", <&str>::from(format)])
        .arg("--model")
        .arg(base_model().await?)
        .output()?)
}

/// Test missing audio file error - validates actual error handling
#[test]
fn test_missing_audio_file() {
    let mut cmd = Command::cargo_bin("purr").unwrap();
    cmd.arg("nonexistent_file.wav");
    cmd.assert().failure();
}

fn golden_path(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/golden")
        .join(name)
}

/// The same recording in every container the samples come in.
#[rstest]
#[case("jfk.wav")]
#[case("jfk.opus")]
#[case("jfk.mp3")]
#[case("jfk.ogg")]
#[timeout(Duration::from_secs(600))]
#[tokio::test]
async fn real_model_cli_transcribes_every_container(#[case] file: &str) -> TestResult {
    let sample = Path::new("../../samples").join(file);
    let output = purr_on(&sample, &OutputFormat::Json).await?;
    assert!(output.status.success(), "{file}");

    let result: Value = serde_json::from_slice(&output.stdout)?;
    let text = result["text"].as_str().ok_or("the output has no text")?;
    purr_golden::check(
        &golden_path(&format!("{file}.json")),
        &Recording {
            text: text.to_owned(),
            audio_duration: None,
        },
        0.0,
    )?;
    Ok(())
}

#[derive(Debug, strum::EnumString, strum::IntoStaticStr, strum::Display)]
#[strum(serialize_all = "lowercase")]
enum OutputFormat {
    Text,
    Json,
    Srt,
}

/// The text of an SRT file: every line that is neither a cue number nor a time range.
fn srt_text(srt: &str) -> String {
    srt.lines()
        .filter(|line| !line.is_empty() && !line.contains("-->") && line.parse::<u32>().is_err())
        .collect::<Vec<_>>()
        .join(" ")
}

#[rstest]
#[case(OutputFormat::Text)]
#[case(OutputFormat::Json)]
#[case(OutputFormat::Srt)]
#[timeout(Duration::from_secs(600))]
#[tokio::test]
async fn real_model_cli_output_formats(#[case] format: OutputFormat) -> TestResult {
    if purr_golden::updating() {
        return Ok(());
    }
    let output = purr_on(Path::new("../../samples/jfk.wav"), &format).await?;
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout)?;

    let text = match format {
        OutputFormat::Text => stdout,
        OutputFormat::Json => {
            let result: Value = serde_json::from_str(&stdout)?;
            assert!(
                result["segments"]
                    .as_array()
                    .is_some_and(|segments| !segments.is_empty())
            );
            result["text"]
                .as_str()
                .ok_or("the output has no text")?
                .to_owned()
        }
        OutputFormat::Srt => srt_text(&stdout),
    };
    let golden = Recording::read(&golden_path("jfk.wav.json"))?;
    purr_golden::assert_text_close(&golden.text, &text, 0.0);
    Ok(())
}

/// Test CLI error handling
#[rstest]
#[case("nonexistent.wav", "should fail with missing file")]
#[case("../../Cargo.toml", "should fail with invalid audio file")]
#[test]
fn test_cli_error_handling(#[case] invalid_path: &str, #[case] description: &str) {
    let mut cmd = Command::cargo_bin("purr").unwrap();
    cmd.arg(invalid_path).timeout(Duration::from_secs(30));

    let output = cmd.output().unwrap();
    assert!(!output.status.success(), "{description}: {invalid_path}");

    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(
        !stderr.is_empty(),
        "Should produce error message for: {invalid_path}"
    );

    println!(
        "✓ CLI correctly handled invalid file: {} -> {}",
        invalid_path,
        stderr.lines().next().unwrap_or("")
    );
}
