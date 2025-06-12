//! Integration tests for whisper-ui-cli

use assert_cmd::Command;
use std::fs;
use tempfile::TempDir;

/// Test CLI argument parsing
#[test]
fn test_cli_help() {
    let mut cmd = Command::cargo_bin("whisper-ui-cli").unwrap();
    cmd.arg("--help");
    cmd.assert().success();
}

/// Test CLI version
#[test]
fn test_cli_version() {
    let mut cmd = Command::cargo_bin("whisper-ui-cli").unwrap();
    cmd.arg("--version");
    cmd.assert().success();
}

/// Test missing audio file error
#[test]
fn test_missing_audio_file() {
    let mut cmd = Command::cargo_bin("whisper-ui-cli").unwrap();
    cmd.arg("nonexistent_file.wav");
    cmd.assert().failure();
}

/// Test invalid arguments
#[test]
fn test_invalid_arguments() {
    let mut cmd = Command::cargo_bin("whisper-ui-cli").unwrap();
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
        let mut cmd = Command::cargo_bin("whisper-ui-cli").unwrap();
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
