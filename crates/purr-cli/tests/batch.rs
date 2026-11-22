//! Batch runs of the CLI against the real `base` model.

mod common;

use assert_cmd::Command;
use axum::{Router, body::Body, body::Bytes, http::header, routing::get};
use common::{TestResult, base_model, golden_path};
use futures::StreamExt as _;
use pretty_assertions::assert_eq;
use purr_core::AudioProcessor;
use purr_golden::Recording;
use rstest::rstest;
use serde_json::Value;
use std::{
    convert::Infallible,
    fs,
    future::IntoFuture as _,
    path::{Path, PathBuf},
    process::Output,
    time::Duration,
};
use tempfile::TempDir;
use tokio::net::TcpListener;

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

fn headers_and_body(stdout: &str) -> (Vec<&str>, String) {
    let (headers, body): (Vec<&str>, Vec<&str>) = stdout
        .lines()
        .filter(|line| !line.is_empty())
        .partition(|line| line.starts_with("==> "));
    (headers, body.join(" "))
}

#[tokio::test]
async fn real_model_cli_batch_with_progress_disabled_matches_default_output() -> TestResult {
    let dir = workspace_with(&["a.wav", "b.wav"])?;
    let default = purr_in(dir.path(), &["a.wav", "b.wav"]).await?;
    let disabled = purr_in(dir.path(), &["a.wav", "b.wav", "--no-progress"]).await?;
    assert_eq!(default.status.code(), Some(0));
    assert_eq!(disabled.status.code(), Some(0));

    let default = String::from_utf8(default.stdout)?;
    let disabled = String::from_utf8(disabled.stdout)?;
    let (default_headers, default_body) = headers_and_body(&default);
    let (disabled_headers, disabled_body) = headers_and_body(&disabled);
    assert_eq!(default_headers, ["==> a.wav <==", "==> b.wav <=="]);
    assert_eq!(disabled_headers, default_headers);
    purr_golden::assert_text_close(&default_body, &disabled_body, 0.0);
    let golden = golden_text()?;
    purr_golden::assert_text_close(&format!("{golden} {golden}"), &default_body, 0.0);
    Ok(())
}

const SAMPLE_RATE: u32 = 16_000;
const OPEN_ENDED: u32 = 0xFFFF_FFFF;

fn wav_header(data_size: u32, riff_size: u32) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(44);
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&riff_size.to_le_bytes());
    bytes.extend_from_slice(b"WAVEfmt ");
    bytes.extend_from_slice(&16u32.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&SAMPLE_RATE.to_le_bytes());
    bytes.extend_from_slice(&(SAMPLE_RATE * 2).to_le_bytes());
    bytes.extend_from_slice(&2u16.to_le_bytes());
    bytes.extend_from_slice(&16u16.to_le_bytes());
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&data_size.to_le_bytes());
    bytes
}

async fn sample_pcm() -> TestResult<Vec<u8>> {
    let mut stream = AudioProcessor::stream_file(&sample()).await?;
    let mut pcm = Vec::new();
    while let Some(chunk) = stream.next().await {
        for value in chunk?.samples {
            let scaled = (value.clamp(-1.0, 1.0) * f32::from(i16::MAX)).round();
            pcm.extend_from_slice(&(scaled as i16).to_le_bytes());
        }
    }
    Ok(pcm)
}

async fn write_long_sample(path: &Path, copies: usize) -> TestResult {
    let pcm = sample_pcm().await?;
    let size = u32::try_from(pcm.len() * copies)?;
    let mut bytes = wav_header(size, 36 + size);
    for _ in 0..copies {
        bytes.extend_from_slice(&pcm);
    }
    fs::write(path, bytes)?;
    Ok(())
}

const LIVE_PIECE: usize = 8_000;
const LIVE_PACE: Duration = Duration::from_millis(50);

async fn serve_live_sample() -> TestResult<String> {
    let pcm = sample_pcm().await?;
    let app = Router::new().route(
        "/audio",
        get(move || {
            let pcm = pcm.clone();
            async move {
                let live = futures::stream::unfold(
                    (Some(wav_header(OPEN_ENDED, OPEN_ENDED)), 0usize),
                    move |(header, at)| {
                        let pcm = pcm.clone();
                        async move {
                            if let Some(header) = header {
                                return Some((
                                    Ok::<_, Infallible>(Bytes::from(header)),
                                    (None, at),
                                ));
                            }
                            tokio::time::sleep(LIVE_PACE).await;
                            let piece: Vec<u8> = pcm
                                .iter()
                                .cycle()
                                .skip(at)
                                .take(LIVE_PIECE)
                                .copied()
                                .collect();
                            let next = (at + LIVE_PIECE) % pcm.len();
                            Some((Ok(Bytes::from(piece)), (None, next)))
                        }
                    },
                );
                (
                    [(header::CONTENT_TYPE, "audio/wav")],
                    Body::from_stream(live),
                )
            }
        }),
    );
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let address = listener.local_addr()?;
    tokio::spawn(axum::serve(listener, app).into_future());
    Ok(format!("http://{address}/audio"))
}

fn words(text: &str) -> usize {
    text.split_whitespace().count()
}

#[tokio::test(flavor = "multi_thread")]
async fn real_model_cli_interrupt_known_length_leaves_part_and_exits_130() -> TestResult {
    let dir = TempDir::new()?;
    write_long_sample(&dir.path().join("long.wav"), 40).await?;
    let output = purr_in(
        dir.path(),
        &["long.wav", "-f", "{stem}.txt", "--interrupt-after", "25000"],
    )
    .await?;
    assert_eq!(output.status.code(), Some(130));

    assert_eq!(file_names(dir.path())?, ["long.txt.part", "long.wav"]);
    assert_ne!(fs::read_to_string(dir.path().join("long.txt.part"))?, "");
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn real_model_cli_duration_cuts_only_streams_in_a_mix() -> TestResult {
    let dir = workspace_with(&["a.wav"])?;
    let url = serve_live_sample().await?;
    let output = purr_in(
        dir.path(),
        &["a.wav", &url, "--duration", "6s", "-f", "out_{index}.txt"],
    )
    .await?;
    assert_eq!(output.status.code(), Some(0));

    assert_eq!(file_names(dir.path())?, ["a.wav", "out_1.txt", "out_2.txt"]);
    let file = fs::read_to_string(dir.path().join("out_1.txt"))?;
    purr_golden::assert_text_close(&golden_text()?, &file, 0.0);
    let stream = fs::read_to_string(dir.path().join("out_2.txt"))?;
    assert!(words(&stream) > 0, "{stream:?}");
    assert!(words(&stream) < words(&file), "{stream:?}");

    let stderr = String::from_utf8(output.stderr)?;
    let lines: Vec<&str> = stderr.lines().collect();
    let heading = lines
        .iter()
        .position(|line| *line == "Ended at the --duration limit:")
        .ok_or("the summary does not name the input cut at the duration")?;
    assert_eq!(lines.get(heading + 1), Some(&format!("  {url}").as_str()));
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn real_model_cli_stop_on_unknown_length_completes_the_file() -> TestResult {
    let dir = TempDir::new()?;
    let url = serve_live_sample().await?;
    let output = purr_in(
        dir.path(),
        &[&url, "-f", "radio.txt", "--interrupt-after", "4000"],
    )
    .await?;
    assert_eq!(output.status.code(), Some(0));

    assert_eq!(file_names(dir.path())?, ["radio.txt"]);
    assert_ne!(fs::read_to_string(dir.path().join("radio.txt"))?, "");
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn real_model_cli_interrupt_at_the_start_leaves_the_second_input_unstarted() -> TestResult {
    let dir = workspace_with(&["a.wav", "b.wav"])?;
    let output = purr_in(
        dir.path(),
        &[
            "a.wav",
            "b.wav",
            "-f",
            "{stem}.txt",
            "--interrupt-after",
            "1",
        ],
    )
    .await?;
    assert_eq!(output.status.code(), Some(130));

    assert!(!dir.path().join("a.txt").exists());
    assert!(!dir.path().join("b.txt").exists());
    assert!(!dir.path().join("b.txt.part").exists());
    let stderr = String::from_utf8(output.stderr)?;
    let summary = stderr.lines().find(|line| {
        *line == "Summary: 2 not started" || *line == "Summary: 1 interrupted, 1 not started"
    });
    assert!(summary.is_some(), "{stderr}");
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn real_model_cli_no_stream_stop_on_unknown_length_writes_what_arrived() -> TestResult {
    let dir = TempDir::new()?;
    let url = serve_live_sample().await?;
    let output = purr_in(
        dir.path(),
        &[&url, "-S", "-f", "radio.txt", "--interrupt-after", "4000"],
    )
    .await?;
    assert_eq!(output.status.code(), Some(0));

    assert_eq!(file_names(dir.path())?, ["radio.txt"]);
    assert_ne!(fs::read_to_string(dir.path().join("radio.txt"))?, "");
    Ok(())
}
