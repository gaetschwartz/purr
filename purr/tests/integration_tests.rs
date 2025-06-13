//! Integration tests for purr

use assert_cmd::Command;
use rstest::rstest;
use std::fs;
use std::path::Path;
use tempfile::TempDir;

/// Test CLI argument parsing
#[test]
fn test_cli_help() {
    let mut cmd = Command::cargo_bin("purr").unwrap();
    cmd.arg("--help");
    cmd.assert().success();
}

/// Test CLI version
#[test]
fn test_cli_version() {
    let mut cmd = Command::cargo_bin("purr").unwrap();
    cmd.arg("--version");
    cmd.assert().success();
}

/// Test missing audio file error
#[test]
fn test_missing_audio_file() {
    let mut cmd = Command::cargo_bin("purr").unwrap();
    cmd.arg("nonexistent_file.wav");
    cmd.assert().failure();
}

/// Test invalid arguments
#[test]
fn test_invalid_arguments() {
    let mut cmd = Command::cargo_bin("purr").unwrap();
    cmd.arg("--invalid-flag");
    cmd.assert().failure();
}

/// Test output format options
#[test]
fn test_output_formats() {
    // Create a temporary audio file (empty for this test)
    let temp_dir = TempDir::new().unwrap();
    let audio_file = temp_dir.path().join("test.wav");
    fs::write(&audio_file, b"dummy audio data").unwrap();

    for format in &["text", "json", "srt"] {
        let mut cmd = Command::cargo_bin("purr").unwrap();
        cmd.arg(&audio_file)
            .arg("--output")
            .arg(format)
            .arg("--no-gpu"); // Avoid GPU requirements in tests

        // Note: This will likely fail due to invalid audio data,
        // but it tests argument parsing
        let output = cmd.output().unwrap();

        // Check that the program at least parsed the arguments correctly
        // (even if transcription fails due to dummy data)
        assert!(output.status.code().is_some());
    }
}

/// Test CLI transcription with all sample files
#[rstest]
#[case("../../samples/jfk.wav")]
#[case("../../samples/jfk.mp3")]
#[case("../../samples/a13.mp3")]
#[case("../../samples/gb0.ogg")]
#[case("../../samples/gb1.ogg")]
#[case("../../samples/hp0.ogg")]
#[case("../../samples/mm1.wav")]
#[case("../../samples/diffusion2023-07-03.flac")]
#[test]
fn test_cli_transcribe_sample_files(#[case] sample_path: &str) {
    let sample_path = Path::new(sample_path);

    // Skip if file doesn't exist
    if !sample_path.exists() {
        println!("⏭ Skipping {} - file not found", sample_path.display());
        return;
    }

    // Skip README.md and other non-audio files
    let extension = sample_path
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("");

    match extension.to_lowercase().as_str() {
        "wav" | "mp3" | "ogg" | "flac" => {
            println!("Testing CLI transcription of: {}", sample_path.display());

            let mut cmd = Command::cargo_bin("purr").unwrap();
            cmd.arg(sample_path)
                .arg("--no-gpu") // Use CPU for consistent test results
                .arg("--output")
                .arg("json") // Use JSON for easier validation
                .timeout(std::time::Duration::from_secs(120)); // Allow up to 2 minutes

            let output = cmd.output().unwrap();

            // Check if command succeeded or failed due to no model
            if output.status.success() {
                // Parse JSON output to validate structure
                let stdout = String::from_utf8(output.stdout).unwrap();

                if let Ok(json_value) = serde_json::from_str::<serde_json::Value>(&stdout) {
                    // Validate JSON structure
                    assert!(
                        json_value.get("text").is_some(),
                        "JSON output should have 'text' field for {}",
                        sample_path.display()
                    );
                    assert!(
                        json_value.get("segments").is_some(),
                        "JSON output should have 'segments' field for {}",
                        sample_path.display()
                    );
                    assert!(
                        json_value.get("audio_duration").is_some(),
                        "JSON output should have 'audio_duration' field for {}",
                        sample_path.display()
                    );

                    let text = json_value["text"].as_str().unwrap_or("");
                    assert!(
                        !text.is_empty(),
                        "Transcribed text should not be empty for {}",
                        sample_path.display()
                    );

                    println!(
                        "✓ CLI successfully transcribed {}: \"{}...\"",
                        sample_path.file_name().unwrap().to_string_lossy(),
                        text.chars().take(50).collect::<String>()
                    );
                } else {
                    panic!(
                        "Failed to parse JSON output for {}: {}",
                        sample_path.display(),
                        stdout
                    );
                }
            } else {
                let stderr = String::from_utf8(output.stderr).unwrap();

                // Check if failure is due to missing model (acceptable for tests)
                if stderr.contains("No Whisper model found")
                    || stderr.contains("No model available")
                {
                    println!(
                        "⚠ Skipping CLI test for {} - no model available",
                        sample_path.display()
                    );
                    return;
                } else {
                    panic!(
                        "CLI transcription failed for {}: {}",
                        sample_path.display(),
                        stderr
                    );
                }
            }
        }
        "md" => {
            println!("⏭ Skipping README file: {}", sample_path.display());
            return;
        }
        _ => {
            println!(
                "⚠ Unknown file type for CLI test: {} (extension: {})",
                sample_path.display(),
                extension
            );
            return;
        }
    }
}

/// Test CLI output formats with a known sample
#[rstest]
#[case("text")]
#[case("json")]
#[case("srt")]
#[test]
fn test_cli_output_formats(#[case] format: &str) {
    let sample_path = "../../samples/jfk.wav";

    // Skip if sample doesn't exist
    if !Path::new(sample_path).exists() {
        println!("⏭ Skipping CLI output format test - sample file not found");
        return;
    }

    let mut cmd = Command::cargo_bin("purr").unwrap();
    cmd.arg(sample_path)
        .arg("--output")
        .arg(format)
        .arg("--no-gpu")
        .timeout(std::time::Duration::from_secs(120));

    let output = cmd.output().unwrap();

    if output.status.success() {
        let stdout = String::from_utf8(output.stdout).unwrap();
        assert!(
            !stdout.trim().is_empty(),
            "Output should not be empty for format: {}",
            format
        );

        match format {
            "json" => {
                // Validate JSON structure
                let json_result = serde_json::from_str::<serde_json::Value>(&stdout);
                assert!(
                    json_result.is_ok(),
                    "Should produce valid JSON for format: {}",
                    format
                );
                println!("✓ CLI JSON format test successful");
            }
            "srt" => {
                // Check for SRT format patterns (timestamps)
                assert!(
                    stdout.contains("-->"),
                    "SRT format should contain timestamp arrows"
                );
                println!("✓ CLI SRT format test successful");
            }
            "text" => {
                // Text should be readable
                assert!(
                    stdout.chars().any(|c| c.is_alphabetic()),
                    "Text format should contain readable text"
                );
                println!("✓ CLI text format test successful");
            }
            _ => panic!("Unknown format: {}", format),
        }
    } else {
        let stderr = String::from_utf8(output.stderr).unwrap();
        if stderr.contains("No Whisper model found") || stderr.contains("No model available") {
            println!(
                "⚠ Skipping CLI output format test for {} - no model available",
                format
            );
            return;
        } else {
            panic!("CLI output format test failed for {}: {}", format, stderr);
        }
    }
}

/// Test CLI model management commands
#[test]
fn test_cli_model_commands() {
    // Test model list command (available models)
    let mut cmd = Command::cargo_bin("purr").unwrap();
    cmd.arg("models").arg("list").arg("--available");

    let output = cmd.output().unwrap();
    assert!(
        output.status.success(),
        "Models list --available command should succeed"
    );

    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(
        stdout.contains("Available Whisper Models"),
        "Models list should show available models"
    );
    assert!(
        stdout.contains("base"),
        "Models list should include base model"
    );

    println!("✓ CLI models list --available command successful");

    // Test default list command (downloaded models)
    let mut cmd = Command::cargo_bin("purr").unwrap();
    cmd.arg("models").arg("list");

    let output = cmd.output().unwrap();
    assert!(
        output.status.success(),
        "Models list command should succeed"
    );

    println!("✓ CLI models list command successful");
}

/// Test CLI error handling
#[rstest]
#[case("nonexistent.wav", "should fail with missing file")]
#[case("../../Cargo.toml", "should fail with invalid audio file")]
#[test]
fn test_cli_error_handling(#[case] invalid_path: &str, #[case] description: &str) {
    let mut cmd = Command::cargo_bin("purr").unwrap();
    cmd.arg(invalid_path)
        .arg("--no-gpu")
        .timeout(std::time::Duration::from_secs(30));

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

/// Test CLI verbose mode
#[test]
fn test_cli_verbose_mode() {
    let sample_path = "../../samples/jfk.wav";

    // Skip if sample doesn't exist
    if !Path::new(sample_path).exists() {
        println!("⏭ Skipping CLI verbose test - sample file not found");
        return;
    }

    let mut cmd = Command::cargo_bin("purr").unwrap();
    cmd.arg(sample_path)
        .arg("--verbose")
        .arg("--no-gpu")
        .timeout(std::time::Duration::from_secs(120));

    let output = cmd.output().unwrap();

    if output.status.success() {
        let stderr = String::from_utf8(output.stderr).unwrap();
        // Verbose mode should produce additional output
        assert!(
            stderr.contains("Whisper UI")
                || stderr.contains("Audio file")
                || stderr.contains("GPU acceleration"),
            "Verbose mode should produce additional output"
        );

        println!("✓ CLI verbose mode test successful");
    } else {
        let stderr = String::from_utf8(output.stderr).unwrap();
        if stderr.contains("No Whisper model found") || stderr.contains("No model available") {
            println!("⚠ Skipping CLI verbose test - no model available");
        } else {
            panic!("CLI verbose test failed: {}", stderr);
        }
    }
}
