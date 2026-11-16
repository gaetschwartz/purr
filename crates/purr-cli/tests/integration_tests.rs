//! Integration tests for purr

use assert_cmd::Command;
use purr_core::ModelManager;
use rstest::rstest;
use serde_json::{Value, json};
use std::{
    fmt,
    path::{Path, PathBuf},
    process::Output,
    time::Duration,
};

const JFK: &str = "And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country.";

const JFK_WITHOUT_FIRST_COMMA: &str = "And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country.";

fn jfk_segments() -> Value {
    json!([
        {
            "text": "And so, my fellow Americans, ask not what your country can do for you,",
            "start": 0.0,
            "end": 8.0,
        },
        {
            "text": "ask what you can do for your country.",
            "start": 8.0,
            "end": 11.0,
        },
    ])
}

const JFK_SRT: &str = "1
00:00:00,000 --> 00:00:08,000
And so, my fellow Americans, ask not what your country can do for you,

2
00:00:08,000 --> 00:00:11,000
ask what you can do for your country.
";

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

/// The same recording in every container the samples come in; the lossy ones lose a comma.
#[rstest]
#[case("jfk.wav", JFK)]
#[case("jfk.opus", JFK)]
#[case("jfk.mp3", JFK_WITHOUT_FIRST_COMMA)]
#[case("jfk.ogg", JFK_WITHOUT_FIRST_COMMA)]
#[timeout(Duration::from_secs(600))]
#[tokio::test]
async fn real_model_cli_transcribes_every_container(
    #[case] file: &str,
    #[case] text: &str,
) -> TestResult {
    let sample = Path::new("../../samples").join(file);
    let output = purr_on(&sample, &OutputFormat::Json).await?;
    assert!(output.status.success(), "{file}");

    let result: Value = serde_json::from_slice(&output.stdout)?;
    assert_eq!(result["text"], text, "{file}");
    Ok(())
}

#[derive(Debug, strum::EnumString, strum::IntoStaticStr, strum::Display)]
#[strum(serialize_all = "lowercase")]
enum OutputFormat {
    Text,
    Json,
    Srt,
}

#[rstest]
#[case(OutputFormat::Text)]
#[case(OutputFormat::Json)]
#[case(OutputFormat::Srt)]
#[timeout(Duration::from_secs(600))]
#[tokio::test]
async fn real_model_cli_output_formats(#[case] format: OutputFormat) -> TestResult {
    let output = purr_on(Path::new("../../samples/jfk.wav"), &format).await?;
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout)?;

    match format {
        OutputFormat::Text => assert_eq!(stdout.trim_end(), JFK),
        OutputFormat::Json => {
            let result: Value = serde_json::from_str(&stdout)?;
            assert_eq!(result["text"], JFK);
            assert_eq!(result["segments"], jfk_segments());
        }
        OutputFormat::Srt => {
            assert_eq!(stdout.replace("\r\n", "\n").trim_end(), JFK_SRT.trim_end());
        }
    }
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
