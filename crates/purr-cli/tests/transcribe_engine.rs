//! `purr sys` lists the transcribe.cpp engine when the binary is built with it.
//!
//! The staged libraries sit next to the binary on Windows; Linux and macOS binaries need the
//! library path set up, which the packaging steps do.

#![cfg(all(feature = "transcribe", windows))]

use assert_cmd::Command;
use serde_json::Value;

#[test]
fn sys_json_lists_the_transcribe_engine_with_its_devices() {
    let output = Command::cargo_bin("purr")
        .unwrap()
        .args(["sys", "--json"])
        .output()
        .unwrap();
    assert!(output.status.success());

    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    let engine = report["engines"]
        .as_array()
        .unwrap()
        .iter()
        .find(|engine| engine["engine"] == "transcribe-cpp")
        .unwrap();
    let ready = &engine["status"]["ready"];
    assert!(!ready["version"].as_str().unwrap().is_empty());
    assert!(
        ready["devices"]
            .as_array()
            .unwrap()
            .iter()
            .any(|device| device["backend"] == "cpu")
    );
}
