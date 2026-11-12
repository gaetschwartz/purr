# whisper.cpp 1.8.3 to 1.9.5: what could matter to Purr

Curated by hand on 2026-11-12. The mechanical lists are [whisper-cpp-1.8.3-to-1.9.5.md](whisper-cpp-1.8.3-to-1.9.5.md) (1604 commits, tags v1.8.4 to v1.9.5) and [ggml-1.8.3-to-1.9.5.md](ggml-1.8.3-to-1.9.5.md) (ggml 0.9.5 to 0.26.0, 1623 commits). Only 23 of the 1604 whisper.cpp commits touch `src/whisper.cpp` or `include/whisper.h`; the rest is ggml syncs, CI, examples and bindings.

Evidence notation: `W:n` is line n of `src/whisper.cpp` and `H:n` line n of `include/whisper.h`, both at tag v1.9.5 (`d1be6fde11ac`), fetched from ggml-org/whisper.cpp. `W183:n` is the same file at v1.8.3 (`2eeeba56e9ed`). `#nnnn` is a whisper.cpp PR, a bare 8-hex string a commit. Nothing was built or run: every behavioural statement below comes from reading the source.

## The six earlier findings, checked against v1.9.5

| | Finding at 1.8.3 | At 1.9.5 | Evidence |
|---|---|---|---|
| a | `no_speech_prob` is computed from logits row 0, but only the last prompt row is filled | Unchanged. Wording refined: row 0 is never written by the prompt decode, so it is zero on a fresh state with a fixed language (probability exactly 1/n_vocab), otherwise stale data left by the last decode that wrote row 0 (the language-detect SOT pass, or a previous step or window) | computation `W:7346-7356` reads `state->logits` from index 0 (`W:6298-6316`); copy-out only for rows with `batch.logits[i] != 0` (`W:3012-3018`), set only on the last prompt token (`W:528`); each sampling step rewrites row 0 (`W:7640-7653`); identical code at `W183:7152-7160`, `W183:2949-2954` |
| b | VAD params have no effect through `whisper_full_with_state` | Unchanged | `params.vad` is read only at `W:7953` (`whisper_full`) and `W:7981` (`whisper_full_parallel`); `whisper_full_with_state` starts at `W:6973` and never reads it. The new VAD getters (see Observability) depend on `has_vad_segments`, set only in the private `whisper_vad` (`W:6823`, `W:6845`), so they are empty on the per-state path |
| c | `whisper_get_timings` returns NULL without a default state | Unchanged; no `_from_state` variant added | `W:4395-4398` (`W183:4245-4248`); `H:437-445` declares the struct, no free function. whisper-rs builds contexts with `*_no_state` (whisper-rs `src/whisper_ctx.rs:48`, `:76`), so Purr hits the NULL case. whisper-rs only wraps `print_timings`/`reset_timings` |
| d | `grammar_rules` is `const whisper_grammar_element **` | Unchanged | `H:581-582`; consumed at `W:7273-7274`. `include/whisper.h` differs between the tags by four additive hunks only |
| e | Abort callback polled once after each encoder and decoder call, not per ggml node | Partly changed: still per call, but now also covers language detection | polls at `W:2518` (encode) and `W:3040` (decode); the scheduler compute helper `W:196-219` never installs the callback on backends (only the CPU-only helper at `W:174-194` does, and it is called with null at `W:8603`). New: abort passed to language detection `W:4162-4188`, `W:7003` (#4077, 398997ed) |
| f | Beam search draws k tokens from a discrete distribution instead of taking top-k | Unchanged | `W:6745-6748` (`W183:6542`); the `partial_sort` at `W:6709-6715` is computed and never used for the result |

## Observability

| Change | Reachable from whisper-rs | Evidence |
|---|---|---|
| `whisper_full_get_token_t0/t1` map token times back to the original timeline when VAD is on | raw-api only; VAD path only | `H:669-678`, `W:8292-8319` (#3910, 5cad7abb) |
| `whisper_full_n_vad_segments`, `whisper_full_get_vad_segment_t0/t1` | raw-api only; VAD path only | `H:684-694`, `W:8321-8343` (#3916, 6fc7c33b) |
| `whisper_vad_detect_speech_no_reset`, `whisper_vad_reset_state` for streaming VAD | raw-api only; usable by Purr (stateless, no `whisper_full` involved) | `H:718-727`, `W:5260-5348` (#3677, 166c20b4) |
| `encoder_begin_callback` now also fires before language auto-detect, so twice on the first window | yes | `W:6996-7001` (#3936, c44b60b8) |
| ggml: GPU-profiler debug labels and pipeline-compile logging in Vulkan, `ggml_backend_dev_caps.mmap_support`, device type `META` | via `ggml_log_set` and `dev.rs` | a6e85dd5, 8291ab84; `ggml/include/ggml-backend.h:146`, `:160` |

Not added: no timings in segments, no per-token or per-step callback, no new fields in `whisper_token_data` or `whisper_full_params` (`H:131-151`, `H:460-582`), no logging API change (`H:763`). The progress callback is still per 30 s window (`W:7195-7200`).

## Fine control

Changed:

- Decoder 0's RNG is re-seeded on every `whisper_full_with_state` call; before, the same audio decoded twice could differ on the temperature-fallback path (`W:7076`, #4025, 6d0ed914).
- `max_tokens` no longer skips the rest of the audio (`W:6393-6399`, `W:7923-7928`, #3798, f08258ab).
- Wrong segment timestamps near silences fixed (`W:7858`, #2279, c33c5618).
- UTF-8 aware `max_len` wrapping (`W:6236-6260`, #3592, f53dc748) and `voice_length` for CJK (#3915, 909307c8).
- VAD: mapping drift from overlap samples fixed (#3711, dc961166); VAD model load rejects `n_encoder_layers != 4` (`W:4969`, #4064, a44e0784).
- `whisper_full_parallel` chunk offsets in 64-bit (`W:8047-8049`, #4044, da545722).

Unchanged: `carry_initial_prompt` (`W:7120`, default `W:6122`), `n_max_text_ctx` and previous-text handling (`W:7101`, `W:7285`, cutoff `W:151`), `suppress_regex` (`H:521`), grammar, the default decoding thresholds (`W:6134-6139`), VAD defaults (`W:4597-4607`), `flash_attn=true` default (`W:3716`). There is no hot-words parameter anywhere in `whisper.h`. Language detection still costs one extra encoder pass on the first window: it encodes at `W:4177`, and the main loop encodes the same window again at `W:7215`. No commit subject in the range mentions repetition or hallucination loops.

## Quality, speed, backends

- ggml 0.9.5 to 0.26.0 is nearly all synced from llama.cpp (`llama/NNNN` subjects) and aimed at LLM kernels (MoE, sparse flash attention). Nothing in the range measures whisper. Items on Purr's backends: CUDA flash-attention tuning (846af526, 8bb85d96) and thin f16 mat-vec (91da4702); Vulkan int8 coopmat for RDNA3/4 (cf0852d2), sparse FA (f69f5904), NVIDIA driver workarounds (f59047cf, 67cf515a), hidden internal symbols to avoid duplicate-dlopen state (fbcb94d0); Metal tensor-API FA (d0dc4363); faster model loading (f449de94); Windows ARM64 MSVC build (b7b84950). Reverts appear inside the range (for example 563a9f7a, 34f53363), so expect churn.
- New encoder offload backends in whisper.cpp itself: AMD VitisAI NPU (#3608, 52dec9d8), Apple ANEForge (#3905, 4afa0094); OpenVINO cross-attention fix (`W:2349`, #3997). Purr wires none of them.
- NVIDIA Parakeet: new library target and API (`include/parakeet.h`, `src/parakeet.cpp`, #3735, 9efddafb). whisper-rs-sys `wrapper.h` includes only `whisper.h` and `ggml.h`, so it is not bound.
- No whisper-specific quantisation or model-format change in the range; `flash_attn` stays default-on.
- Hardening: C++ exceptions during model load (for example Vulkan `vk::SystemError`) now return NULL instead of aborting the host process (`W:3832-3844`, #3831, 610e664b); invalid tensor type and dims rejected (`W:1899`, `W:5179`, #4091, #3957); heap OOB on very short audio (#3956); null loader read (#3982); uninitialised mel (#3981).

## API, ABI and build

- `whisper.h`: additive only (the functions listed above). No removal, rename, struct layout or default change.
- ggml headers Purr touches: `ggml_backend_dev_caps` gains `bool mmap_support` (`ggml-backend.h:160`, after `events`), `ggml_backend_dev_type` gains `META`. Signatures of `ggml_backend_dev_get/init/count/get_props`, `ggml_backend_free`, `ggml_cpu_has_avx/avx2/fma/f16c`, `ggml_log_set` and the `GGML_LOG_LEVEL_*` values are unchanged. `ggml_add1`, `ggml_mul_mat_set_prec` and `ggml_flash_attn_ext_set_prec` are now deprecated; whisper-rs and Purr use none.
- CMake: the version moved from `project("whisper.cpp" VERSION x.y.z)` to `set(WHISPER_VERSION_MAJOR/MINOR/PATCH ...)` (`CMakeLists.txt:5-9`). Minimum versions are unchanged for whisper (3.5) and Vulkan (3.19); ggml becomes `3.14...3.28`; CUDA adds `90-virtual` to the default architectures. New option defaults: `GGML_CUDA_NCCL` ON but optional (`ggml/CMakeLists.txt:211`, `ggml-cuda/CMakeLists.txt:184-191`). Removed options: `GGML_HIP_ROCWMMA_FATTN`. A `parakeet` library target is now built unconditionally (`src/CMakeLists.txt:196-199`) and nothing links it.

## whisper-rs PR #284 ("Update whisper.cpp to v1.9.5", open since 2026-11-08, mergeable, no comments)

The diff is two files, 15 additions and 4 deletions: `sys/whisper.cpp` moves from `2eeeba56` to `d1be6fde` (the v1.9.5 tag), and `sys/build.rs` `get_whisper_cpp_version` reads `WHISPER_VERSION_MAJOR/MINOR/PATCH` instead of the `project(... VERSION` line. Without that second change the build script panics (parser at `sys/build.rs:382-395`, `expect("Could not find whisper.cpp version declaration")` at `:350-353`).

- Bindings: regenerated at build time by bindgen from `wrapper.h` (`sys/build.rs:119-174`), so the new functions and the changed ggml structs appear automatically. The checked-in `sys/src/bindings.rs` is not refreshed (it has none of the new VAD functions). It is used when bindgen fails (`:166-171`) or `WHISPER_DONT_GENERATE_BINDINGS` is set; with it, `ggml_backend_dev_caps` has one bool too few, and `ggml_backend_dev_get_props` into a zeroed `ggml_backend_dev_props` would write past the end of the struct.
- whisper-rs itself: no change needed and none made. No safe wrappers exist for any new function, so Purr reaches them through `raw-api`.
- Versions and packaging: neither crate version is bumped (sys stays 0.15.0) and there is no CHANGELOG entry. `sys/Cargo.toml` `include` lists only `whisper.cpp/include/whisper.h`, while `src/CMakeLists.txt:196-199` lists `../include/parakeet.h` as a source of the unconditional `parakeet` target. A crate packaged from #284 as it stands would very likely fail at CMake configure (inferred, not built). A git dependency or `[patch]` on the repository, which carries the full submodule, would not have that problem. Purr uses crates.io 0.16.0 today.
- Purr: `dev.rs:141-155` has a `_ => Unknown` arm, so `META` is safe. `dev.rs:197-229` serialises four caps fields and ignores `mmap_support`; its comment block at `:201-206` would be stale. `probe.rs` and `logging.rs` use only unchanged signatures. `whisper/sync.rs` and `streaming.rs` set language, temperature and print flags, and no VAD, abort, timings or grammar. Nothing in Purr's use is expected to break, but this was not compiled.

## Worth migrating?

Ranked by value to Purr:

1. Model load no longer aborts the process on C++ exceptions (#3831, `W:3832-3844`). It fits the GPU probe and fallback chain, since the Vulkan backend can throw during device and buffer allocation (comment at `W:3832-3836`).
2. Reproducible decoding on reused states (#4025, `W:7076`); matters whenever temperature fallback runs (`temperature_inc` defaults to 0.2, `W:6136`); Purr's streaming calls `full` repeatedly on one `WhisperState` (`streaming.rs:303`).
3. Decode fixes: timestamps near silences (#2279), `max_tokens` (#3798, only if used), UTF-8 wrapping (#3592).
4. Cancellable language detection (#4077, #3936); only useful once Purr installs abort or encoder-begin callbacks, which it does not today.
5. Crash hardening on malformed models and short audio (#3957, #4091, #3956, #4044).
6. Streaming VAD primitives (#3677) through `raw-api`, if Purr ever wants whisper's own VAD without `whisper_full`.
7. ggml GPU speedups: plausible, unmeasured, mostly LLM-targeted.

Not delivered by 1.9.5: a fix for `no_speech_prob`, VAD on the per-state path, per-node abort, timings without a default state, top-k beam search, hot words.

Risks and blockers: a crates.io-based release of #284 probably fails on `parakeet.h` (use a git dependency until sys is repackaged); stale `bindings.rs` fallback is memory-unsafe for `dev.rs`; 1623 ggml commits with several reverts and no whisper benchmark, so GPU regressions on CUDA/Vulkan are possible; CUDA builds pick up NCCL if present on a Linux runner and `build.rs` emits no link flag for it (inferred from the CMake lines above).

Could not verify: any build or run; the numeric effect of the ggml changes on whisper speed or VRAM; the `parakeet.h` packaging failure and NCCL link failure (inferred); whether the stale-row behaviour in (a) shows up as a constant or varying value in practice; that `cargo` bindgen output compiles against whisper-rs 0.16.0 unchanged.
