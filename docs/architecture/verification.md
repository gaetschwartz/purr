# Verification

What is checked, how, and what is known to hold on the Windows development machine. A status
is only "verified" with a date and the command that produced it; everything else says NOT
verified. Dates are 2026-11-14 unless another is given. Machine: Windows 11, Ryzen 9 7950X,
debug builds (`cargo`, profile `dev`), FFmpeg shared from scoop, no GPU feature, except where a row says otherwise (the GPU rows, dated 2026-11-16, use
release builds with Vulkan and CUDA on an RTX 2070 SUPER; see [GPU through transcribe.cpp](#gpu-through-transcribecpp-2026-11-16)).

Kinds of check: **unit** (`cargo nextest`, no model, runs in CI `test (linux)`), **model-gated**
(`real_model_*` tests that load models from the model cache; the default nextest profile leaves them out), **script** (`scripts/*`, run by CI jobs
and by hand), **CI** (a workflow step, only a runner proves it), **manual** (someone runs the app).

## Matrix

| Area | How it is checked | Windows status today |
|---|---|---|
| Whisper output on `base` and `small`, streaming and whole, with and without VAD, auto and fixed language, on jfk (11 s), mm1 (30 s), the first minute of a13 and all of a13 (179 s) | model-gated: `real_model_golden.rs`, 20 goldens in `crates/purr-engine-whisper/tests/golden/` named `<model>-<sample>-<mode>-<language>.json` | **Verified**, 20 of 20 goldens identical in two runs, one thread per test: `just test-models -p purr-engine-whisper --test real_model_golden`. The six goldens of a13 streams were re-recorded when a Whisper decode stopped reusing a used state (see engines.md); every other golden is unchanged. The identity at 1 and at 8 threads was measured before that change and not repeated |
| Whisper cancel, live stream with tentative text, a sentence spoken twice, progress | model-gated: `real_model_cancel.rs`, `real_model_live_stream.rs`, `real_model_whisper.rs` | **Verified**: 5 passed in the last run (the cancel test, the live stream, the repeated sentence, progress, and a window decoding the same after other decodes). Cancel races a one second timer against the decode of 179 s of audio, and the live stream feeds 11 s of audio at the pace of speech: both depend on time by nature |
| Parakeet end to end through the shared pipeline (stream with and without VAD, whole, tentative text with words and scores) | model-gated: `crates/purr-engine-transcribe/tests/real_model_parakeet.rs`, 7 goldens in `tests/golden/` | **Verified**, 8 passed in the run below. Goldens on jfk, mm1 and the first minute of a13 |
| Qwen3-ASR end to end through the shared pipeline (stream with and without VAD, `Whole`, language reported) | model-gated: `crates/purr-engine-transcribe/tests/real_model_qwen3.rs`, 7 goldens in `tests/golden/` | **Verified**, 8 passed in the run below (0.6B Q8_0). Text without times: the goldens carry no words, the language is `en`. The decoder's limit is `UpTo(_)` |
| A window the engine stopped early on (`Completion::Truncated` / `Repeating`) | unit: scripted decoders in `purr-core/src/pipeline/tests.rs` and `pipeline/retry.rs`; `purr-engine-transcribe/src/error.rs` | **Verified** as unit tests. The path is proven with scripted engines only: no real Qwen3-ASR run on the samples tried so far truncated or looped |
| Parakeet on Linux | manual | **Verified once**, WSL Ubuntu 26.04, debug build: `purr -m parakeet-tdt-0.6b-v3-Q4_K_M.gguf --accel cpu samples/jfk.wav` prints the JFK sentence, next to a Whisper run in the same binary |
| Device union (one CPU seen by both engines, `--device` resolved against the engine that runs the model) | unit: `accel/hardware.rs`, `accel/selector.rs`, `accel/plan.rs`; script: `check-windows-sys.ps1 -ExpectTranscribe`; manual: `purr sys --json` | **Verified**: unit tests in the 663 below; `purr sys --json` on the staged folder lists one CPU with `seen_by` whisper-cpp and transcribe-cpp. **Verified** on 2026-11-16 with Vulkan and CUDA: `purr sys` lists the RTX 2070 SUPER once per backend and the Radeon once, each with `seen_by` whisper-cpp and transcribe-cpp; unit: `Candidate::find_gpu` through the whisper.cpp tests, `purr-engine-transcribe` `placement.rs`, `purr-app` `device_rows` (a GPU both engines see is offered to a transcribe.cpp model). The app's Hardware page and model picker NOT run |
| Development run finds the libraries without `PURR_RUNTIME_DIR` (`dx serve`, `cargo run`) | unit: `purr-core` `accel/runtime_dir.rs`, `purr-delayload` `emit.rs`; manual: `purr sys` of a debug `purr.exe` copied alone into an empty folder | **Verified**: it reports transcribe.cpp 0.4.2 loaded from the `outin` of the `transcribe-cpp-sys` build; with that folder renamed away, the error lists the searched folders, the build output among them. A release build has no such folder: unit-tested in `purr-delayload`; `dx serve` itself NOT run here |
| Failed load of transcribe.cpp is retried | unit: `purr-engine-transcribe` `runtime.rs`, with a scripted loader | **Verified**: a missing library is looked for again, a success and a "no compute backend" are kept, a missing library opens nothing. The app's "Try again" on a real run: NOT verified here |
| GPU on the transcribe.cpp engine: Vulkan and CUDA, device order, `--accel cpu` | model-gated: `crates/purr-engine-transcribe/tests/real_model_gpu.rs` (one test per model, built with `--features vulkan` or `cuda`); manual: `purr -v -m <model> --accel vulkan|cuda|cpu` | **Verified** 2026-11-16: 2 of 2 pass on Vulkan and on CUDA, the GPU text equals the CPU golden (jfk, whole); the CPU goldens still pass in a Vulkan build (`--features vulkan`, jfk cases). Timings and the differences on a13 in the GPU section below. Metal, Linux and the other backends NOT run |
| GPU runtime missing or broken (Vulkan loader, cuBLAS) | script: `check-windows-sys.ps1 -ExpectTranscribeBackends`; manual: `purr sys`, `purr engines --json` on a copy with the runtime hidden | **Verified** 2026-11-16 for CUDA (cuBLAS hidden) and Vulkan (loader replaced by a text file): exit 0, typed reason, engine ready on the CPU, run falls back. Output in the GPU section below. The Visual C++ runtime still NOT verified |
| Delay-load, libraries absent | script: `check-windows-sys.ps1 -ExpectTranscribe` (executable alone) and `check-windows-imports.ps1 -ExpectDelayLoaded transcribe.dll` | **Verified** (on a debug build, before a debug build searched its own build output; the script now needs a release build) with a local copy of the sys script that also copies the FFmpeg DLLs (this machine's FFmpeg is dynamic, CI's is static): exit 0, "transcribe.cpp engine - Not loadable: the library transcribe.dll was not found", whisper.cpp available. The imports check passes as is |
| Delay-load, libraries present | the same scripts, executable next to the staged DLLs | **Verified**: transcribe-cpp `ready`, version 0.4.2, one CPU device, shared with whisper.cpp |
| Delay-load, libraries broken | manual, on a copy of the staged folder | **Verified**, `purr sys --json` exits 0 each time: a text file named `transcribe.dll` gives `library_unloadable`; `ggml.dll` removed gives `library_unloadable`; all `ggml-cpu-*.dll` removed gives `no_backend`. Whisper stays `ready` in all three. Missing Visual C++ runtime: NOT verified (the machine has it) |
| `purr sys` with the feature off and on | manual | **Verified**: off prints no Engines section and the legacy JSON keys (`backends`, `cpu`, `picked_gpus`, `default_order`, `devices`); on prints the Engines section and `engines` and `hardware` with `seen_by` |
| Model catalog, download, verification, engine routing of files outside the catalog | unit: `purr-core/src/model/registry/tests.rs`, `tests/model_manager_test.rs`, `tests/model_download_test.rs`; CI `test (linux)` | **Verified**: 663 passed, 16 skipped: `cargo nextest run -j 4 --no-fail-fast -p purr-delayload -p purr-common -p purr-core -p purr-engine-transcribe -p purr-cli --features purr-cli/transcribe` (includes `purr-cli::transcribe_engine`, which starts the staged binary). A real download of a catalog file from Hugging Face is verified for Qwen3-ASR: `purr models download qwen3-asr-0.6b-q8_0` and `qwen3-asr-1.7b-q8_0` fetched the pinned URLs and passed the size and sha256 of the manifest. Parakeet: NOT verified here |
| UI states (greyed-out models of an engine that is not built, device rows, engine notes in Settings, start plan) | unit: `purr-app` `model_catalog`, `transcribe`, `views/settings` tests; manual | Unit tests of `purr-app` NOT run in this session. Rendering in the app: NOT verified (no app, browser or screenshot run) |
| wasm builds (`purr-wasm`, `purr-app` web) | CI job `web` (`dx bundle --web`); `just test-wasm` | NOT verified here |
| Linux executable keeps whisper.cpp's ggml apart from transcribe.cpp's | script/CI: step "The two engines keep their own ggml" of `test (linux)` | **Verified in WSL** (debug build, GNU ld 2.46) after the fix in `purr-delayload`: no ggml or gguf symbol in `nm -D`, no binding of a transcribe.cpp library to the executable. Before the fix: 161 exported symbols and 11 bindings of `libtranscribe.so` to `purr`. The workflow step itself: NOT run on a runner |
| Packaging, GPU modules | script: `stage-transcribe-runtime.ps1 -Backends`, `check-windows-imports.ps1 -Module`, `check-windows-sys.ps1 -ExpectTranscribeBackends`; CI `cli (windows, cuda-vulkan)` | **Verified** 2026-11-16 on a release `transcribe,cuda,vulkan` and a release `transcribe,vulkan` CLI: staging with `-Backends` lists 14 or 13 files and fails without it or with a backend too many (`-Check` too); `-Module` passes for `ggml-vulkan.dll` and `ggml-cuda.dll` and fails for `purr.exe`; `check-windows-sys.ps1` passes on both folders with this machine's FFmpeg DLLs copied into its temporary folders. `package-release-assets.sh` only syntax-checked (no `zip` here); the workflow NOT run |
| Packaging, Windows CLI zip | script: `stage-transcribe-runtime.ps1`, `check-windows-imports.ps1`, `check-windows-sys.ps1`; CI `cli (windows, *)` | Staging (from cargo's JSON messages and from the build directory, also after a no-op rebuild that left the profile folder without the DLLs) and both checks **verified** on a debug CLI, before a debug build searched its build output (see below; the sys check now needs a release build to see the libraries absent). Release build, CUDA and Vulkan variants and the zip made by CI: NOT verified |
| Packaging, Windows NSIS installer and portable archive | CI `desktop (windows, cuda-vulkan)` with `PURR_DESKTOP_TRANSCRIBE` on | NOT verified. `dx bundle` was not run (the NSIS tools are downloaded by it), so the `dist-runtime` resources are unproven. Off by default |
| Packaging, Linux and macOS | CI | NOT run: the `cli (linux, vulkan)` and `cli (macos, metal)` jobs now carry the engine and its GPU module (workflow edits read, not run); NOT done for the Linux and macOS desktop packages |
| `cargo tree -p purr-core` | command | **Verified**: `cargo tree -p purr-core --offline \| rg -i 'whisper\|ggml\|transcribe'` prints nothing, also with `-e features` |

## Re-running the model-gated tests

The models are looked up in the model cache, the folder printed as "XDG data directory" by
`purr models list`: `%APPDATA%\purr\data\models` on Windows, `~/.local/share/purr/models` on
Linux. A real-model test whose model is missing fails, with the `purr models download <name>`
command in its message. The tests are the `real_model_*` binaries and tests, which the default
nextest profile leaves out; `just test-models` runs them (`cargo nextest run --profile all -E
'test(/real_model_/) | binary_id(/real_model_/)'`), each on one thread, CPU only. Setup scripts in
`.config/nextest.toml` build `purr-cli` and run `purr models download` for the models of the tests
selected, so a run downloads what is missing and keeps a file whose size matches the catalog (the
checksum is checked on download, and by `purr models verify`); `just download-test-models` does the
same by hand. A failed download of a model fails the run; a failed download of the Silero model
does not, the CLI only logs it, and the tests then fail with their missing-model message.

| Suite | Models | Cost, one thread per test, all tests at once |
|---|---|---|
| Whisper goldens (`real_model_golden`) | `base`, `small`, the Silero model | 20 cases, 10 s (`base`, jfk) to 196 s (`small`, all of a13) each |
| Whisper cancel, live stream, repeated sentence, progress | `base`, the Silero model | 9 s, 39 s, 33 s, 18 s; the live stream lasts as long as the 11 s sample plays, twice |
| Parakeet | `parakeet-tdt-0.6b-v3-q4_k_m`, the Silero model, and the transcribe.cpp libraries next to the test binaries (`transcribe-cpp-sys` stages them on Windows) | 3 s to 23 s per case |
| Qwen3-ASR | `qwen3-asr-0.6b-q8_0`, the Silero model, same libraries | 12 s to 51 s per case |
| CLI (`purr-cli` `integration_tests`) | `base` | 16 s to 17 s per case |
| GPU text (`real_model_gpu`, only in a build with `--features vulkan`, `cuda` or `metal` of `purr-engine-transcribe`) | `parakeet-tdt-0.6b-v3-q4_k_m`, `qwen3-asr-0.6b-q8_0`, same libraries | 2 tests, 1 s to 11 s on this GPU once its kernels are cached. `just test-models` builds without a GPU feature, where the binary is empty; run `cargo nextest run --profile all -p purr-engine-transcribe --features vulkan -E 'binary_id(/real_model_gpu/)'` |

The whole group: 47 passed in 215 s on the 7950X, three runs in a row (`cargo nextest run --profile all -E
'test(/real_model_/) | binary_id(/real_model_/)'`, 32 tests at a time). After a change to the
pipeline, the goldens are the regression gate: a mismatch prints a diff, and `PURR_GOLDEN=update`
records them again once the change is understood. A golden holds exactly what one engine on this
machine's CPU build produced; the Whisper goldens were identical at 1 and at 8 threads (before the
re-recording of the a13 stream goldens), so the thread count is not part of a golden.

The live stream test compares the final chunks exactly with and without tentative text: a tentative
decode must not change what is committed, which holds because a Whisper decode never runs on a state
an earlier decode used (`a_window_decodes_the_same_after_other_decodes`).

With cargo 1.100's build layout the test binaries of `purr-engine-transcribe` are written under
`target/debug/build/purr-engine-transcribe/<hash>/out/`, but `transcribe-cpp-sys` stages the DLLs
in `target/debug/build/deps/`: the binaries import `transcribe.dll` and the loader does not find it
(`0xc0000135` when nextest lists them). Putting `target\debug\build\deps` on `PATH` for the run
works around it; the default run never lists these binaries.

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

With the Vulkan backend there is one more file, `ggml-vulkan.dll` (45,311,488 bytes, 14,185,127 deflated; the thirteen files deflate to 18,747,409 bytes), and with the CUDA backend `ggml-cuda.dll`
(148,412,416 bytes in a debug build's tree, 148,406,784 in the release build, 133,947,288 deflated: its kernels do not compress). Fourteen files make 206,304,768 bytes (2026-11-16, release).

Deflated into a zip the twelve CPU files are 4,562,294 bytes. The installer's LZMA size is not measured. Release
builds were not made, so these sizes are those of the debug build's libraries (ggml is built in
Release either way, so a release build should be close).

## Open checks

Each of these needs something this machine cannot give; the exact check is in
[ci.md](ci.md#what-is-not-validated).

1. Run the workflow: `build.yml` Windows CLI jobs with `transcribe`, `ci.yml` Linux steps.
2. Turn on `PURR_DESKTOP_TRANSCRIBE` once a run shows the second `dx bundle` and the `_staging` check pass.
3. Linux and macOS packages of the libraries, and a macOS `purr sys --json` run.
4. Metal on an M1 Mac: the build and run commands are in [Metal on an M1 Mac](#metal-on-an-m1-mac-not-run-here).
5. Vulkan on Linux (`cli (linux, vulkan)` builds the engine with `libggml-vulkan.so`; the bare Ubuntu image must show a CPU and no Vulkan device), and CUDA on Linux, which no job builds.
6. Whether `cli (windows, cuda-vulkan)` fits its time budget with the transcribe.cpp CUDA build added; see [ci.md](ci.md#what-the-gpu-modules-cost-the-jobs).

## GPU through transcribe.cpp (2026-11-16)

Machine: Windows 11, Ryzen 9 7950X, RTX 2070 SUPER (8 GB, driver 621.14, Vulkan 1.4, compute capability 7.5), the integrated Radeon
Graphics of the CPU, CUDA 13.5, other work on the machine idle. Release build of `purr-cli` with `transcribe,cuda,vulkan` (FFmpeg shared from scoop, the
libraries staged next to `purr.exe` by `stage-transcribe-runtime.ps1 -Backends cuda, vulkan`), `target` outside the worktree, because the whisper.cpp Vulkan
build failed in the worktree's deep path (a CMake error in `vulkan-shaders-gen`). Each cell is one untimed run followed by three timed ones of the CLI in its default streaming mode with the Silero model,
`purr -m <model> --accel cpu|vulkan|cuda -o json <file>`; "wall" is the whole process, "decode" the `processing_time` the CLI reports
(pipeline start to end, so the model load and the process start are out of it). The CPU runs use the CLI default of one thread per
physical core (16). A first run is the untimed one, so what the GPU paid once is in it.

| Model | Input | Device | Wall, median of 3 (s) | Realtime, wall | Decode, median of 3 (s) | Realtime, decode | First run (s) | Text against the CPU run |
|---|---|---|---:|---:|---:|---:|---:|---|
| Parakeet Q4_K_M | jfk (11 s) | CPU (16 threads) | 0.91 | 12.1x | 0.28 | 39.6x | 1.1 | reference |
| Parakeet Q4_K_M | jfk (11 s) | Vulkan, RTX 2070 SUPER | 0.98 | 11.2x | 0.10 | 108.4x | 1.2 | identical |
| Parakeet Q4_K_M | jfk (11 s) | CUDA, RTX 2070 SUPER | 1.02 | 10.8x | 0.24 | 46.0x | 1.1 | identical |
| Parakeet Q4_K_M | a13, first 60 s | CPU (16 threads) | 1.91 | 31.3x | 1.29 | 46.6x | 1.9 | reference |
| Parakeet Q4_K_M | a13, first 60 s | Vulkan, RTX 2070 SUPER | 1.33 | 45.1x | 0.46 | 131.4x | 1.4 | differs, see below |
| Parakeet Q4_K_M | a13, first 60 s | CUDA, RTX 2070 SUPER | 1.54 | 38.9x | 0.75 | 79.5x | 1.6 | differs, see below |
| Qwen3-ASR 0.6B Q8_0 | jfk (11 s) | CPU (16 threads) | 1.62 | 6.8x | 0.99 | 11.1x | 1.7 | reference |
| Qwen3-ASR 0.6B Q8_0 | jfk (11 s) | Vulkan, RTX 2070 SUPER | 1.41 | 7.8x | 0.26 | 43.1x | 1.5 | identical |
| Qwen3-ASR 0.6B Q8_0 | jfk (11 s) | CUDA, RTX 2070 SUPER | 1.41 | 7.8x | 0.36 | 31.0x | 6.0 | identical |
| Qwen3-ASR 0.6B Q8_0 | a13, first 60 s | CPU (16 threads) | 5.52 | 10.9x | 4.89 | 12.3x | 5.5 | reference |
| Qwen3-ASR 0.6B Q8_0 | a13, first 60 s | Vulkan, RTX 2070 SUPER | 2.15 | 27.9x | 0.98 | 61.5x | 2.2 | differs, see below |
| Qwen3-ASR 0.6B Q8_0 | a13, first 60 s | CUDA, RTX 2070 SUPER | 2.19 | 27.5x | 1.12 | 53.4x | 2.2 | differs, see below |

- Text: every run of a cell gave the same text. On jfk both models give the CPU's text exactly, on Vulkan and on CUDA, as the
  `real_model_gpu` tests check against the CPU goldens (whole-source mode). On the first 60 s of a13 (noisy radio, many sentences) the text differs
  from the CPU's. Parakeet: three small differences on Vulkan (CPU/GPU: `stand by 13` / `stand by a 13`, `me was` / `me B was`, `that j that that jolt` / `that j that jolt`,
  `sensor,` / `sensor`) and the same minus one on CUDA plus a trailing `line`. Qwen3-ASR (a language model, so sensitive in the stretches it half hears): both GPUs
  differ in the same places, for example `a call B P with the one that had a Hampton. I've gone to once before` on the CPU against `a very cold baby with the one that had a
  campsite gone in once before` on Vulkan, and CUDA adds a sentence near the start. Which of them is closer to what was said was not judged here.
- A Vulkan-only release build (`transcribe,vulkan`) gives the same decode times within 5 %, and wall times 0.1 to 0.4 s lower on every row, CPU runs included (probably because the
  CUDA build opens cuBLAS at startup, `Inventory::detect`; not checked). Its medians (wall/decode, s): Parakeet jfk CPU 0.69/0.27 and Vulkan 0.77/0.10, a13 CPU 1.77/1.27 and Vulkan 1.12/0.47;
  Qwen3-ASR jfk CPU 1.38/0.97 and Vulkan 1.19/0.25, a13 CPU 5.13/4.73 and Vulkan 1.90/0.97.
- The first time a configuration runs it pays for shader or kernel compilation, which the driver then caches: the very first Vulkan run of Parakeet on jfk took 26.5 s of
  decode time (the second took 0.45 s in a debug build), and the first Qwen3-ASR run took about 6 s on Vulkan and on CUDA.
- The integrated Radeon runs both models on jfk with the same text as the CPU (`--device vulkan/igpu`). Not timed.
- CUDA is slower than Vulkan here in decode (Parakeet a13: 0.75 s against 0.46 s); a fixed cost shows already on jfk (0.24 s against 0.10 s). Why was not investigated.
- With 16 CPU threads on this CPU the GPU wins mostly on the longer inputs: wall 5.5 s to 2.2 s for Qwen3-ASR on a13. For a short file the process start and the model load, which cost 0.25 to 0.55 s more on the GPU
  than on the CPU (wall minus decode), cancel what the decode saves.

### What failed, and what was verified of the fallbacks

- The GPU runtime missing, hidden for real, with `purr.exe` and the 14 libraries copied to an empty folder, `CUDA_PATH` unset and `PATH` reduced to the system folders:
  `purr engines --json` and `purr sys --json` exit 0; transcribe-cpp lists `vulkan, vulkan, cpu` (the CUDA module was left out by ggml; `build` says CUDA is built) and
  whisper-cpp lists `cuda, vulkan, vulkan, cpu`; `purr sys --json` reports `backends.cuda` as `RuntimeLibraryMissing` for `cublasLt64_13.dll`
  with the folders searched; `purr -m parakeet-tdt-0.6b-v3-q4_k_m --accel cuda jfk.wav` prints the JFK sentence and the log says `Acceleration: Vulkan (NVIDIA GeForce RTX 2070 SUPER)`.
- A Vulkan loader that is not a library (a text file named `vulkan-1.dll` next to `purr.exe`): `purr sys` exits 0, `Vulkan - Built, but the runtime library vulkan-1.dll
  cannot be loaded`, transcribe-cpp lists the CPU alone, and `--accel vulkan` runs on the CPU (`Acceleration: CPU`).
- `TRANSCRIBE_BACKENDS=cpu purr --device vulkan/0000:01:00.0 -m parakeet-... jfk.wav` stops with the typed `NotOnEngine` error (`describes a device the transcribe.cpp engine cannot run on; only the whisper.cpp
  engine lists it`) and `--accel vulkan` with the same variable runs on the CPU.
- Found by running it: the engine first scanned transcribe.cpp's backend modules before Purr had opened the GPU runtime libraries from its folders, and ggml scans once, so with cuBLAS only under `%CUDA_PATH%`
  (not on `PATH`) the CUDA module was skipped for good while Whisper used CUDA. `runtime::load` now opens them first: `real_model_gpu` failed before and passes after with `PATH` reduced to the system
  folders, and the CLI runs Qwen3-ASR on `Acceleration: CUDA` with only `CUDA_PATH` set.
- NOT verified with a real failure: a GPU that exists but cannot load the model (`AttemptFailure::DeviceInit`, `DeviceOutOfMemory`) is unit-tested through `open_error`; no way was found to make this GPU fail the load
  (`GGML_VK_FORCE_MAX_ALLOCATION_SIZE` did not). A GPU that loads the model and fails while decoding is a decode error and does not fall back: there is no warm-up decode.
- Not run: ROCm, SYCL, Linux (any backend), Intel GPUs, two NVIDIA GPUs, a GPU with less memory than the model.

### Metal on an M1 Mac (not run here)

Needs Xcode command line tools, `brew install cmake ffmpeg pkg-config`, the models of the two engines (`purr models download parakeet-tdt-0.6b-v3-q4_k_m qwen3-asr-0.6b-q8_0`).

```
cargo build -p purr-cli --release --features metal,transcribe --message-format json-render-diagnostics > cargo-messages.jsonl
lib_dir="$(jq -rs '[.[] | select(.reason == "build-script-executed" and (.package_id | contains("transcribe-cpp-sys"))) | .out_dir] | unique | .[0]' cargo-messages.jsonl)/lib"
mkdir -p /tmp/purr-metal/lib && cp target/release/purr /tmp/purr-metal/
find "$lib_dir" -maxdepth 1 \( -type f -o -type l \) -exec cp -P {} /tmp/purr-metal/lib/ \;
install_name_tool -add_rpath @executable_path/lib /tmp/purr-metal/purr
ls /tmp/purr-metal/lib | grep metal                                   # a libggml-metal* module
/tmp/purr-metal/purr sys                                             # Metal device seen by whisper.cpp engine and transcribe.cpp engine
/tmp/purr-metal/purr engines --json | jq '.engines[] | {engine, built: (.build.backends[] | select(.backend == "metal"))}'
/tmp/purr-metal/purr -v -m parakeet-tdt-0.6b-v3-q4_k_m samples/jfk.wav          # log: Acceleration: Metal (Apple M1)
/tmp/purr-metal/purr -v -m qwen3-asr-0.6b-q8_0 samples/jfk.wav
/tmp/purr-metal/purr -v -m parakeet-tdt-0.6b-v3-q4_k_m --accel cpu samples/jfk.wav   # log: Acceleration: CPU
cargo test -p purr-engine-transcribe --features metal --test real_model_gpu     # Metal text equals the CPU golden of jfk
```

Expected: the transcribe.cpp engine lists `Metal` with the same name as whisper.cpp, so one hardware entry with both engines in `seen_by`; both models print the jfk text and the log says Metal; `--accel cpu` says CPU;
`real_model_gpu` passes, or fails with a text diff, in which case record the diff here. Also time `purr -m ... --accel cpu|metal -o json samples/jfk.wav` as in the table above (`--accel metal` is the backend name). The build has `GGML_METAL_EMBED_LIBRARY` on, so no
`default.metallib` has to sit next to the library. The `desktop (macos, metal)` bundle has no transcribe.cpp libraries and is not covered.

## Model comparison (2026-11-14)

CLI debug build, CPU, 4 threads, VAD on, stream mode, Q8_0 for Qwen3-ASR and Parakeet, F16 Whisper. Agreement is a proxy: word error rate of each model against Whisper large-v3-turbo on the same input (case and punctuation removed), not against a reference transcript. Speed is audio seconds per decode second; Whisper ran from this build's statically linked whisper.cpp, whose numbers here are far below what the model should reach and are not a statement about Whisper.

| Input | Parakeet v3 | Qwen3-ASR 0.6B | Qwen3-ASR 1.7B | Whisper base | Whisper turbo |
|---|---|---|---|---|---|
| JFK, 11 s (English) | 0% , 18x | 0%, 7.8x | 0%, 3.7x | 0%, 4.5x | reference, 0.3x |
| Apollo 13, first 60 s (English, noisy) | 29%, 14x | 32%, 6.8x | 30%, 3.4x | 61%, 6.8x | reference, 0.7x |
| French voice note, 39 s | 83%, English output | 6%, 4.5x, French | 7%, 2.6x, French | 10%, 5.3x | reference, 0.5x |
| French voice note, first 60 s | 79%, English output | 8%, 5.5x, French | 5%, 2.6x, French | 8%, 7.6x | reference, 0.7x |

Qwen3-ASR with `--language fr` gives the same French text. With `--language en` on the French note the 0.6B model answers in English and the 1.7B model keeps transcribing French. A 179 s file with `--no-stream` on the 0.6B model ran as seven pause-cut windows in 24 s. No truncated or repeating output was seen in any run.
