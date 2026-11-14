# Engines and the pipeline

An engine is speech recognition software that runs a model on a device: today whisper.cpp
(`purr-engine-whisper`). The browser's engine (`purr-wasm`) stays outside this design. Everything
else, from cutting audio into windows to deciding what text is final, is one pipeline in
`purr-core` that does not know which engine it drives.

## Crates

```
purr-common          vocabulary: EngineId, ModelProfile, LanguageTag, Word, ChunkKind
purr-core            audio, models, hardware (plan, report, runtime libraries, delay-load hook),
                     the engine traits and the pipeline; no ggml, no whisper-rs
purr-engine-whisper  whisper-rs: model loading, ggml devices and probe, token-to-word grouping,
                     Silero VAD, whisper.cpp logging, abort callback, CoreML encoder path
purr-cli, purr-app   composition roots: list the engines (`engines.rs`) and forward the cargo
                     features cuda, vulkan, metal, coreml to the engine crate, static-ffmpeg to
                     purr-core
```

## The engine boundary (`purr_core::engine`)

- `Engine`: `id`, `build_info` (backends and CPU instruction sets compiled in), `devices` (its own
  device list; a device's id is its index there), `open` (load a model on one planned device),
  `open_speech_detector`, `install_logging`.
- `WindowDecoder`: one loaded model. `decode(DecodeRequest)` takes 16 kHz mono samples, a
  `LanguageRequest` (`Detect` or `Use(tag)`), the finest `Timing` wanted and a `CancelToken`, and
  returns a `Hypothesis` (segments with optional `Span`s and byte-ranged `Word`s, the detected
  language with its confidence when the engine gives one, and a `Segmentation` marker) with a
  `Completion` (`Complete`, or `Cancelled` holding what the engine finished). Language detection is
  part of the decode. `limits()` gives the longest input one decode takes (`InputLimit`) and the
  longest window worth decoding while streaming. `supports(tag)` is checked before the run starts.
- `SpeechDetector`: the spans of a window that hold speech.
- Errors: `OpenError` (`NoDevice`, `Device(AttemptFailure)`, `Engine`), `DecodeError`
  (`InputTooLong`, `InputTooShort`, `UnsupportedLanguage`, `OutOfMemory`, `Engine`), `DetectError`.
  Each engine keeps its own error enum and converts into these; `purr_core::Error` holds only them.

Hardware stays neutral: the plan (`accel::plan`) orders `Candidate`s by backend, name and PCI id,
and `engine::open::open_on_best_device` tries them in order, calling `Engine::open` for each and
recording every attempt in the `AccelReport`. The engine finds its own handle for a candidate among
the devices it listed.

`Inventory::detect(engine)` opens the GPU runtime libraries before it asks the engine for its
devices. The order matters: listing devices registers ggml's backends, and their delay-loaded
imports must bind to the libraries Purr opened from its search directories. Anything that makes an
engine touch its devices goes through `Inventory::detect` first.

## The pipeline (`purr_core::pipeline`)

```
Input (file, URL, AudioStream)
  -> window policy (policy.rs) + optional SpeechDetector: Full | AtPauses | Whole
  -> WindowDecoder::decode
  -> Segmenter (segmenter.rs): engine segments with honest spans
  -> commit.rs, tentative.rs, language.rs: what is final, tentative text, language lock
  -> StreamEvent (Chunk | Progress), gathered into a SyncTranscriptionResult by result::collect
```

- `run::transcribe(input, RunConfig, &ModelRef, engines)` picks the engine the `ModelRef` names,
  loads the model while the audio starts to arrive, and returns a `Transcription` stream. Dropping
  it cancels the run inside the window being decoded.
- `WindowPolicy::choose` derives the windows from the decoder's limits, whether a speech detector
  is loaded and `RunConfig::windowing`. `Whole` (the whole source in one decode, what
  `purr --no-stream` asks for) is only chosen when the decoder takes input of any length;
  otherwise the run streams.
- The `Segmenter` keeps `Segmentation::Native` segments as the engine gave them. For
  `Synthesize` it splits at sentence punctuation (Latin, CJK, Arabic, Devanagari, Ethiopic), at a
  silence of a second between words and at ten seconds, and ends each segment at its last word,
  never past the window. With honest spans, `commit`'s margin rule keeps a sentence still being
  spoken uncommitted. `engine::text::TextAssembler` builds segment text and word ranges for engines
  that report words rather than text.
- Language: a detection with confidence locks the stream at 0.8 or more on a window with speech; an
  engine that gives no confidence locks on its first detection with speech.

## Whisper in this design

- `limits()`: input `Unbounded` (whisper.cpp's `full()` slides over long audio itself, carrying the
  previous text within one call) and a 30 s streaming window, because the encoder always sees 30 s.
- Segments are `Native`; spans come from whisper.cpp's segment timestamps, words from its tokens
  (`words.rs`).
- Detection runs `pcm_to_mel` and `lang_detect` on the window, then `full()` in the detected
  language; this is the same detection `full()` makes itself.
- Cancel: whisper.cpp's abort callback reads the `CancelToken`. An aborted window of 30 s or less
  yields no segment; an aborted whole-file decode keeps the segments of the 30 s stretches it
  finished.
- Beam search with 5 beams, always.

## Adding an engine

1. Create `crates/purr-engine-<name>` depending on `purr-core` and `purr-common`, with its own
   cargo features for the GPU backends it builds; forward `cuda` and `vulkan` to `purr-core` so the
   delay-load hook and the runtime-library names are compiled in.
2. Implement `Engine`, `WindowDecoder` and, if it has one, `SpeechDetector`; convert the engine's
   own error enum into `OpenError`, `DecodeError` and `DetectError`.
3. Report honest `limits()`, and `Segmentation::Synthesize` when the engine gives long stretches with
   timed words.
4. Add it to `engines::all()` in `purr-cli` and `purr-app`, and forward the binaries' features to it.
5. Give its catalog models that `EngineId`, so `ModelManager::reference` and the app pick it.
6. Run the Whisper goldens (`cargo nextest run -p purr-engine-whisper --test golden
   --run-ignored only`) to show the shared pipeline did not move.
