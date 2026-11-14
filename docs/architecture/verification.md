# Verification

What is checked, how, and what is known to hold on the Windows development machine. A status
is only "verified" with a date and the command that produced it; everything else says NOT
verified. Dates are 2026-11-14 unless another is given. Machine: Windows 11, Ryzen 9 7950X,
debug builds (`cargo`, profile `dev`), FFmpeg shared from scoop, no GPU feature.

Kinds of check: **unit** (`cargo nextest`, no model, runs in CI `test (linux)`), **model-gated**
(`#[ignore]` tests that need models in the model cache), **script** (`scripts/*`, run by CI jobs
and by hand), **CI** (a workflow step, only a runner proves it), **manual** (someone runs the app).

## Matrix

| Area | How it is checked | Windows status today |
|---|---|---|
| Whisper output, streaming and `--no-stream`, with and without VAD, auto and fixed language | model-gated: `golden.rs`, seven goldens in `crates/purr-engine-whisper/tests/golden/` | **Verified**, 7 of 7 goldens identical: `PURR_TEST_THREADS=4 cargo test -p purr-engine-whisper --test golden -- --ignored`, 1 passed in 55.6 s. First run since the Parakeet work and the shared pipeline were merged |
| Whisper cancel, live stream with tentative text | model-gated: `cancel.rs`, `live_stream.rs` | **Verified**: `cargo test -p purr-engine-whisper --test cancel --test live_stream -- --ignored`, 2 passed (1.4 s, 25.6 s) |
| Parakeet end to end through the shared pipeline (stream with and without VAD, whole, tentative text with words and scores) | model-gated: `crates/purr-engine-transcribe/tests/parakeet.rs` | **Verified**: `cargo test -p purr-engine-transcribe --test parakeet -- --ignored`, 2 passed in 12.5 s. The text is checked by assertions, not by goldens: there are no Parakeet goldens |
| Qwen3-ASR end to end through the shared pipeline (stream with and without VAD, `Whole`, language reported) | model-gated: `crates/purr-engine-transcribe/tests/qwen3.rs` | **Verified**: `cargo nextest run -p purr-engine-transcribe --test qwen3 --run-ignored only`, 1 passed in 25.8 s (0.6B Q8_0, 4 threads). JFK is the exact sentence with language `en`; 60 s of a13 stays in order in all three modes. The decoder's limit is `UpTo(5218.56 s)` |
| A window the engine stopped early on (`Completion::Truncated` / `Repeating`) | unit: scripted decoders in `purr-core/src/pipeline/tests.rs` and `pipeline/retry.rs`; `purr-engine-transcribe/src/error.rs` | **Verified** as unit tests. The path is proven with scripted engines only: no real Qwen3-ASR run on the samples tried so far truncated or looped |
| Parakeet on Linux | manual | **Verified once**, WSL Ubuntu 26.04, debug build: `purr -m parakeet-tdt-0.6b-v3-Q4_K_M.gguf --accel cpu samples/jfk.wav` prints the JFK sentence, next to a Whisper run in the same binary |
| Device union (one CPU seen by both engines, `--device` resolved against the engine that runs the model) | unit: `accel/hardware.rs`, `accel/selector.rs`, `accel/plan.rs`; script: `check-windows-sys.ps1 -ExpectTranscribe`; manual: `purr sys --json` | **Verified**: unit tests in the 663 below; `purr sys --json` on the staged folder lists one CPU with `seen_by` whisper-cpp and transcribe-cpp. Windows GPU devices through the second engine: not applicable, the second engine is CPU only |
| Development run finds the libraries without `PURR_RUNTIME_DIR` (`dx serve`, `cargo run`) | unit: `purr-core` `accel/runtime_dir.rs`, `purr-delayload` `emit.rs`; manual: `purr sys` of a debug `purr.exe` copied alone into an empty folder | **Verified**: it reports transcribe.cpp 0.4.2 loaded from the `outin` of the `transcribe-cpp-sys` build; with that folder renamed away, the error lists the searched folders, the build output among them. A release build has no such folder: unit-tested in `purr-delayload`; `dx serve` itself NOT run here |
| Failed load of transcribe.cpp is retried | unit: `purr-engine-transcribe` `runtime.rs`, with a scripted loader | **Verified**: a missing library is looked for again, a success and a "no compute backend" are kept, a missing library opens nothing. The app's "Try again" on a real run: NOT verified here |
| Delay-load, libraries absent | script: `check-windows-sys.ps1 -ExpectTranscribe` (executable alone) and `check-windows-imports.ps1 -ExpectDelayLoaded transcribe.dll` | **Verified** (on a debug build, before a debug build searched its own build output; the script now needs a release build) with a local copy of the sys script that also copies the FFmpeg DLLs (this machine's FFmpeg is dynamic, CI's is static): exit 0, "transcribe.cpp engine - Not loadable: the library transcribe.dll was not found", whisper.cpp available. The imports check passes as is |
| Delay-load, libraries present | the same scripts, executable next to the staged DLLs | **Verified**: transcribe-cpp `ready`, version 0.4.2, one CPU device, shared with whisper.cpp |
| Delay-load, libraries broken | manual, on a copy of the staged folder | **Verified**, `purr sys --json` exits 0 each time: a text file named `transcribe.dll` gives `library_unloadable`; `ggml.dll` removed gives `library_unloadable`; all `ggml-cpu-*.dll` removed gives `no_backend`. Whisper stays `ready` in all three. Missing Visual C++ runtime: NOT verified (the machine has it) |
| `purr sys` with the feature off and on | manual | **Verified**: off prints no Engines section and the legacy JSON keys (`backends`, `cpu`, `picked_gpus`, `default_order`, `devices`); on prints the Engines section and `engines` and `hardware` with `seen_by` |
| Model catalog, download, verification, engine routing of files outside the catalog | unit: `purr-core/src/model/registry/tests.rs`, `tests/model_manager_test.rs`, `tests/model_download_test.rs`; CI `test (linux)` | **Verified**: 663 passed, 16 skipped: `cargo nextest run -j 4 --no-fail-fast -p purr-delayload -p purr-common -p purr-core -p purr-engine-transcribe -p purr-cli --features purr-cli/transcribe` (includes `purr-cli::transcribe_engine`, which starts the staged binary). A real download of a catalog file from Hugging Face is verified for Qwen3-ASR: `purr models download qwen3-asr-0.6b-q8_0` and `qwen3-asr-1.7b-q8_0` fetched the pinned URLs and passed the size and sha256 of the manifest. Parakeet: NOT verified here |
| UI states (greyed-out models of an engine that is not built, device rows, engine notes in Settings, start plan) | unit: `purr-app` `model_catalog`, `transcribe`, `views/settings` tests; manual | Unit tests of `purr-app` NOT run in this session. Rendering in the app: NOT verified (no app, browser or screenshot run) |
| wasm builds (`purr-wasm`, `purr-app` web) | CI job `web` (`dx bundle --web`); `just test-wasm` | NOT verified here |
| Linux executable keeps whisper.cpp's ggml apart from transcribe.cpp's | script/CI: step "The two engines keep their own ggml" of `test (linux)` | **Verified in WSL** (debug build, GNU ld 2.46) after the fix in `purr-delayload`: no ggml or gguf symbol in `nm -D`, no binding of a transcribe.cpp library to the executable. Before the fix: 161 exported symbols and 11 bindings of `libtranscribe.so` to `purr`. The workflow step itself: NOT run on a runner |
| Packaging, Windows CLI zip | script: `stage-transcribe-runtime.ps1`, `check-windows-imports.ps1`, `check-windows-sys.ps1`; CI `cli (windows, *)` | Staging (from cargo's JSON messages and from the build directory, also after a no-op rebuild that left the profile folder without the DLLs) and both checks **verified** on a debug CLI, before a debug build searched its build output (see below; the sys check now needs a release build to see the libraries absent). Release build, CUDA and Vulkan variants and the zip made by CI: NOT verified |
| Packaging, Windows NSIS installer and portable archive | CI `desktop (windows, cuda-vulkan)` with `PURR_DESKTOP_TRANSCRIBE` on | NOT verified. `dx bundle` was not run (the NSIS tools are downloaded by it), so the `dist-runtime` resources are unproven. Off by default |
| Packaging, Linux and macOS | CI | NOT done for Linux packages; macOS CLI job edits NOT run |
| `cargo tree -p purr-core` | command | **Verified**: `cargo tree -p purr-core --offline \| rg -i 'whisper\|ggml\|transcribe'` prints nothing, also with `-e features` |

## Re-running the model-gated tests

The models are looked up in the model cache, the folder printed as "XDG data directory" by
`purr models list`: `%APPDATA%\purr\data\models` on Windows, `~/.local/share/purr/models` on
Linux. A test whose model is missing prints "skipping" and passes, so look for that word.

| Suite | Models | Command | Cost |
|---|---|---|---|
| Whisper goldens | `ggml-base.bin`, `ggml-silero-v6.2.0.bin` | `PURR_TEST_THREADS=4 cargo test -p purr-engine-whisper --test golden -- --ignored` | about 1 minute decoding, 8 threads by default |
| Whisper cancel and live stream | `ggml-base.bin` | `cargo test -p purr-engine-whisper --test cancel --test live_stream -- --ignored` | half a minute |
| Qwen3-ASR | `Qwen3-ASR-0.6B-Q8_0.gguf`, the Silero model, and the transcribe.cpp libraries next to the test binaries | `cargo test -p purr-engine-transcribe --test qwen3 -- --ignored --nocapture` | 26 seconds, 4 threads |
| Parakeet | `parakeet-tdt-0.6b-v3-Q4_K_M.gguf`, the Silero model, and the transcribe.cpp libraries next to the test binaries (`transcribe-cpp-sys` stages them on Windows) | `cargo test -p purr-engine-transcribe --test parakeet -- --ignored --nocapture` | 12 seconds, 4 threads |

`cargo nextest run -p <crate> --test <name> --run-ignored only` does the same. After a change to
the pipeline, the Whisper goldens are the regression gate: a mismatch prints a diff per case, and
`PURR_GOLDEN=update` records them again once the change is understood. The goldens passed at 4
threads although the constant is 8; if a mismatch ever shows with another count, re-run it at 8
before believing it.

On a Claude desktop session the Windows profile folders are virtualised per package, so a
model downloaded by the app and one seen by a tool run from the session can be two different
folders.

## Packaging evidence

Debug CLI built with `cargo build -p purr-cli --features transcribe`, staged with
`scripts/stage-transcribe-runtime.ps1 -CargoMessages <cargo json> -Destination <dir>`:

| File | Bytes |
|---|---:|
| `transcribe.dll` | 2,052,608 |
| `ggml.dll` | 61,952 |
| `ggml-base.dll` | 675,328 |
| `ggml-cpu-x64.dll` | 843,264 |
| `ggml-cpu-sse42.dll` | 842,240 |
| `ggml-cpu-sandybridge.dll` | 1,076,224 |
| `ggml-cpu-haswell.dll` | 1,105,408 |
| `ggml-cpu-skylakex.dll` | 1,206,784 |
| `ggml-cpu-cannonlake.dll` | 1,206,784 |
| `ggml-cpu-cascadelake.dll` | 1,206,272 |
| `ggml-cpu-icelake.dll` | 1,206,272 |
| `ggml-cpu-alderlake.dll` | 1,103,360 |
| **total** | **12,586,496** |

Deflated into a zip they are 4,562,294 bytes. The installer's LZMA size is not measured. Release
builds were not made, so these sizes are those of the debug build's libraries (ggml is built in
Release either way, so a release build should be close).

## Open checks

Each of these needs something this machine cannot give; the exact check is in
[ci.md](ci.md#what-is-not-validated).

1. Run the workflow: `build.yml` Windows CLI jobs with `transcribe`, `ci.yml` Linux steps.
2. Turn on `PURR_DESKTOP_TRANSCRIBE` once a run shows the second `dx bundle` and the `_staging` check pass.
3. Linux and macOS packages of the libraries, and a macOS `purr sys --json` run.
4. Parakeet goldens, if the output should be pinned the way Whisper's is.

## Model comparison (2026-11-14)

CLI debug build, CPU, 4 threads, VAD on, stream mode, Q8_0 for Qwen3-ASR and Parakeet, F16 Whisper. Agreement is a proxy: word error rate of each model against Whisper large-v3-turbo on the same input (case and punctuation removed), not against a reference transcript. Speed is audio seconds per decode second; Whisper ran from this build's statically linked whisper.cpp, whose numbers here are far below what the model should reach and are not a statement about Whisper.

| Input | Parakeet v3 | Qwen3-ASR 0.6B | Qwen3-ASR 1.7B | Whisper base | Whisper turbo |
|---|---|---|---|---|---|
| JFK, 11 s (English) | 0% , 18x | 0%, 7.8x | 0%, 3.7x | 0%, 4.5x | reference, 0.3x |
| Apollo 13, first 60 s (English, noisy) | 29%, 14x | 32%, 6.8x | 30%, 3.4x | 61%, 6.8x | reference, 0.7x |
| French voice note, 39 s | 83%, English output | 6%, 4.5x, French | 7%, 2.6x, French | 10%, 5.3x | reference, 0.5x |
| French voice note, first 60 s | 79%, English output | 8%, 5.5x, French | 5%, 2.6x, French | 8%, 7.6x | reference, 0.7x |

Qwen3-ASR with `--language fr` gives the same French text. With `--language en` on the French note the 0.6B model answers in English and the 1.7B model keeps transcribing French. A 179 s file with `--no-stream` on the 0.6B model ran as seven pause-cut windows in 24 s. No truncated or repeating output was seen in any run.
