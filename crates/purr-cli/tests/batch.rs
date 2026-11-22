//! Batch runs of the CLI against the real `base` model.

mod common;

use assert_cmd::Command;
use common::{TestResult, base_model, golden_path};
use pretty_assertions::assert_eq;
use purr_golden::Recording;
use rstest::rstest;
use serde_json::Value;
use std::{
    fs,
    path::{Path, PathBuf},
    process::Output,
    time::Duration,
};
use tempfile::TempDir;

fn sample() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../samples/jfk.wav")
}

fn workspace_with(names: &[&str]) -> TestResult<TempDir> {
    let dir = TempDir::new()?;
    for name in names {
        fs::copy(sample(), dir.path().join(name))?;
    }
    Ok(dir)
}

async fn purr_in(dir: &Path, args: &[&str]) -> TestResult<Output> {
    Ok(Command::cargo_bin("purr")?
        .current_dir(dir)
        .args(["--no-gpu", "--threads", "1", "--model"])
        .arg(base_model().await?)
        .args(args)
        .timeout(Duration::from_secs(600))
        .output()?)
}

fn file_names(dir: &Path) -> TestResult<Vec<String>> {
    let mut names = fs::read_dir(dir)?
        .map(|entry| Ok(entry?.file_name().to_string_lossy().into_owned()))
        .collect::<TestResult<Vec<_>>>()?;
    names.sort();
    Ok(names)
}

fn srt_text(srt: &str) -> String {
    srt.split("\n\n")
        .filter(|block| !block.is_empty())
        .flat_map(|block| block.lines().skip(2))
        .collect::<Vec<_>>()
        .join(" ")
}

fn golden_text() -> TestResult<String> {
    Ok(Recording::read(&golden_path("jfk.wav.json"))?.text)
}

#[tokio::test]
async fn real_model_cli_batch_per_file_outputs() -> TestResult {
    let dir = workspace_with(&["a.wav", "b.wav"])?;
    let output = purr_in(
        dir.path(),
        &["a.wav", "b.wav", "-o", "srt", "-f", "{stem}.{ext}"],
    )
    .await?;
    assert_eq!(output.status.code(), Some(0));

    assert_eq!(
        file_names(dir.path())?,
        ["a.srt", "a.wav", "b.srt", "b.wav"]
    );
    for name in ["a.srt", "b.srt"] {
        let text = srt_text(&fs::read_to_string(dir.path().join(name))?);
        purr_golden::assert_text_close(&golden_text()?, &text, 0.0);
    }
    Ok(())
}

#[tokio::test]
async fn real_model_cli_batch_missing_input_fails_only_that_input() -> TestResult {
    let dir = workspace_with(&["a.wav", "b.wav"])?;
    let output = purr_in(
        dir.path(),
        &["a.wav", "missing.wav", "b.wav", "-f", "{stem}.txt"],
    )
    .await?;
    assert_eq!(output.status.code(), Some(1));

    assert_eq!(
        file_names(dir.path())?,
        ["a.txt", "a.wav", "b.txt", "b.wav"]
    );
    let stderr = String::from_utf8(output.stderr)?;
    assert!(
        stderr
            .lines()
            .any(|line| line == "  missing.wav: Media file not found: missing.wav"),
        "{stderr}"
    );
    assert!(
        stderr
            .lines()
            .any(|line| line == "Summary: 2 done, 1 failed"),
        "{stderr}"
    );
    Ok(())
}

#[tokio::test]
async fn real_model_cli_batch_merge_text() -> TestResult {
    let dir = workspace_with(&["a.wav", "b.wav"])?;
    let output = purr_in(dir.path(), &["a.wav", "b.wav", "--merge", "-f", "all.txt"]).await?;
    assert_eq!(output.status.code(), Some(0));

    assert_eq!(file_names(dir.path())?, ["a.wav", "all.txt", "b.wav"]);
    let merged = fs::read_to_string(dir.path().join("all.txt"))?;
    let lines: Vec<&str> = merged.lines().collect();
    let second = lines
        .iter()
        .position(|line| *line == "==> b.wav <==")
        .ok_or("the second header is missing")?;
    assert_eq!(lines.first(), Some(&"==> a.wav <=="));
    assert_eq!(lines.get(second - 1), Some(&""));
    for text in [&lines[1..second - 1], &lines[second + 1..]] {
        purr_golden::assert_text_close(&golden_text()?, &text.join(" "), 0.0);
    }
    Ok(())
}

#[tokio::test]
async fn real_model_cli_batch_skip_existing() -> TestResult {
    let dir = workspace_with(&["a.wav", "b.wav"])?;
    fs::write(dir.path().join("a.txt"), "kept")?;
    fs::write(dir.path().join("b.txt.part"), "stale")?;
    let output = purr_in(
        dir.path(),
        &["a.wav", "b.wav", "--skip-existing", "-f", "{stem}.txt"],
    )
    .await?;
    assert_eq!(output.status.code(), Some(0));

    assert_eq!(
        file_names(dir.path())?,
        ["a.txt", "a.wav", "b.txt", "b.wav"]
    );
    assert_eq!(fs::read_to_string(dir.path().join("a.txt"))?, "kept");
    assert_ne!(fs::read_to_string(dir.path().join("b.txt"))?, "");
    Ok(())
}

#[derive(Debug, Clone, Copy, strum::IntoStaticStr)]
#[strum(serialize_all = "lowercase")]
enum Format {
    Text,
    Json,
    Srt,
    Txt,
}

#[rstest]
#[case(Format::Text)]
#[case(Format::Json)]
#[case(Format::Srt)]
#[case(Format::Txt)]
#[tokio::test]
async fn real_model_cli_single_form_output_is_unchanged(#[case] format: Format) -> TestResult {
    let dir = workspace_with(&["a.wav"])?;
    let name: &str = format.into();
    let output = purr_in(dir.path(), &["a.wav", "-o", name]).await?;
    assert_eq!(output.status.code(), Some(0));
    let stdout = String::from_utf8(output.stdout)?;

    let text = match format {
        Format::Text => {
            assert!(stdout.ends_with(" \n"), "{stdout:?}");
            stdout
        }
        Format::Txt => {
            assert!(stdout.ends_with('\n'), "{stdout:?}");
            stdout
        }
        Format::Json => {
            assert!(stdout.ends_with("}\n\n"), "{stdout:?}");
            stdout
                .lines()
                .filter(|line| !line.is_empty())
                .map(|line| {
                    let chunk: Value = serde_json::from_str(line)?;
                    assert_eq!(chunk.get("input"), None);
                    Ok(chunk["text"].as_str().ok_or("no text")?.to_owned())
                })
                .collect::<TestResult<Vec<_>>>()?
                .join(" ")
        }
        Format::Srt => {
            assert_eq!(stdout.lines().next(), Some("1"));
            srt_text(&stdout)
        }
    };
    purr_golden::assert_text_close(&golden_text()?, &text, 0.0);
    assert_eq!(file_names(dir.path())?, ["a.wav"]);
    Ok(())
}

#[tokio::test]
async fn real_model_cli_single_form_output_file_is_written_whole() -> TestResult {
    let dir = workspace_with(&["a.wav"])?;
    let output = purr_in(dir.path(), &["a.wav", "-f", "{stem}.out.txt"]).await?;
    assert_eq!(output.status.code(), Some(0));
    assert_eq!(output.stdout, b"");
    assert_eq!(file_names(dir.path())?, ["a.out.txt", "a.wav"]);
    let written = fs::read_to_string(dir.path().join("a.out.txt"))?;
    purr_golden::assert_text_close(&golden_text()?, &written, 0.0);
    Ok(())
}
