//! Integration tests for purr

use assert_cmd::Command;
use rstest::rstest;
use std::{
    path::{Path, PathBuf},
    time::Duration,
};

const TINY_MODEL: &str = "../../.local/models/ggml-tiny.bin";

/// Test missing audio file error - validates actual error handling
#[test]
fn test_missing_audio_file() {
    let mut cmd = Command::cargo_bin("purr").unwrap();
    cmd.arg("nonexistent_file.wav");
    cmd.assert().failure();
}

/// Test CLI transcription with all sample files
#[rstest]
#[timeout(Duration::from_secs(600))]
#[tokio::test]
async fn slow_test_cli_transcribe_sample_files(#[files("../../samples/*")] sample_path: PathBuf) {
    use purr_core::SyncTranscriptionResult;
    if !Path::new(&TINY_MODEL).exists() {
        panic!(
            "Tiny model not found at {}. Please download it to run tests.",
            TINY_MODEL
        );
    }

    let sample_path = sample_path.canonicalize().unwrap();
    println!("Testing CLI transcription of: {}", sample_path.display());

    let mut cmd = Command::cargo_bin("purr").unwrap();
    cmd.arg(&sample_path)
        .arg("--no-stream") // Disable streaming for faster tests
        .arg("--output")
        .arg("json") // Use JSON for easier validation
        .arg("--temperature")
        .arg("0.0") // Deterministic results
        .arg("--model")
        .arg(TINY_MODEL);

    println!("Running command: {:?}", cmd);

    let output = cmd.output().unwrap();

    // Check if command succeeded or failed due to no model
    // Parse JSON output to validate structure
    let stdout = String::from_utf8(output.stdout).unwrap();

    let res = serde_json::from_str::<SyncTranscriptionResult>(&stdout).unwrap_or_else(|_| {
        panic!(
            "Failed to parse JSON output for file {}: {}",
            sample_path.display(),
            stdout
        )
    });

    println!(
        "✓ CLI successfully transcribed {}: \"{}...\"",
        sample_path.file_name().unwrap().to_string_lossy(),
        res.text.chars().take(50).collect::<String>()
    );
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
async fn slow_test_cli_output_formats(#[case] format: OutputFormat) {
    let sample_path = Path::new("../../samples/jfk.wav");
    if !Path::new(&TINY_MODEL).exists() {
        panic!(
            "Tiny model not found at {}. Please download it to run tests.",
            TINY_MODEL
        );
    }

    let mut cmd = Command::cargo_bin("purr").unwrap();
    cmd.arg(sample_path)
        .arg("--output")
        .arg(<&str>::from(&format))
        .arg("--no-stream") // Disable streaming for faster tests
        .arg("--temperature")
        .arg("0.0") // Deterministic results
        .arg("--model")
        .arg(TINY_MODEL);

    let output = cmd.output().unwrap();

    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(
        !stdout.trim().is_empty(),
        "Output should not be empty for format: {}",
        format
    );

    match format {
        OutputFormat::Json => {
            // Validate JSON structure
            let json_result = serde_json::from_str::<serde_json::Value>(&stdout);
            assert!(
                json_result.is_ok(),
                "Should produce valid JSON for format: {}",
                format
            );
            println!("✓ CLI JSON format test successful");
        }
        OutputFormat::Srt => {
            // Check for SRT format patterns (timestamps)
            assert!(
                stdout.contains("-->"),
                "SRT format should contain timestamp arrows"
            );
            println!("✓ CLI SRT format test successful");
        }
        OutputFormat::Text => {
            // Text should be readable
            assert!(
                stdout.chars().any(|c| c.is_alphabetic()),
                "Text format should contain readable text"
            );
            println!("✓ CLI text format test successful");
        }
    }
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
    assert!(
        !output.status.success(),
        "{}: {}",
        description,
        invalid_path
    );

    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(
        !stderr.is_empty(),
        "Should produce error message for: {}",
        invalid_path
    );

    println!(
        "✓ CLI correctly handled invalid file: {} -> {}",
        invalid_path,
        stderr.lines().next().unwrap_or("")
    );
}
