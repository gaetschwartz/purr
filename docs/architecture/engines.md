# Engines and the pipeline

An engine is speech recognition software that runs a model on a device: whisper.cpp
(`purr-engine-whisper`) for the Whisper models, and transcribe.cpp (`purr-engine-transcribe`,
behind the `transcribe` feature) for the Parakeet and Qwen3-ASR models. The browser's engine (`purr-wasm`) stays
outside this design. Everything else, from cutting audio into windows to deciding what text is final, is one pipeline in
`purr-core` that does not know which engine it drives.

## Crates

```
purr-common          vocabulary: EngineId, ModelProfile, LanguageTag, Word, ChunkKind
purr-core            audio, models, hardware (plan, report, runtime libraries, delay-load hook),
                     the engine traits and the pipeline; no ggml, no whisper-rs
purr-engine-whisper  whisper-rs: model loading, ggml devices and probe, token-to-word grouping,
                     Silero VAD, whisper.cpp logging, abort callback, CoreML encoder path
purr-engine-transcribe  transcribe.cpp (transcribe-cpp, with its own ggml in shared libraries):
                     guarded loading of those libraries, backend and device listing, logging,
                     decoding of any transcribe.cpp model, Silero VAD
purr-cli, purr-app   composition roots: list the engines (`engines.rs`) and forward the cargo
                     features cuda, vulkan, metal, coreml to the whisper engine crate,
                     static-ffmpeg to purr-core, and `transcribe` to add the transcribe.cpp engine
```

## The engine boundary (`purr_core::engine`)

- `Engine`: `id`, `availability` (whether its software can be used, see below), `build_info`
  (backends and CPU instruction sets compiled in), `devices` (its own
  device list; a device's id is its index there), `open` (load a model on one planned device),
  `open_speech_detector`, `install_logging`.
- `WindowDecoder`: one loaded model. `decode(DecodeRequest)` takes 16 kHz mono samples, a
  `LanguageRequest` (`Detect` or `Use(tag)`), the finest `Timing` wanted and a `CancelToken`, and
  returns a `Hypothesis` (segments with optional `Span`s and byte-ranged `Word`s, the detected
  language with its confidence when the engine gives one, and a `Segmentation` marker) with a
  `Completion`: `Complete`; `Cancelled`, holding what the engine finished; or `Truncated` /
  `Repeating`, holding the text of a language model that ran out of room for its output or
  began to loop. That text does not say how far into the window it reached, so the pipeline never
  trusts where it ends (see "When the engine stops early" below). Language detection is
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

## Devices across engines

Each engine has its own ggml, so a device's index is a position in one engine's list and a saved
device (`DeviceKey`: backend, name, occurrence) means the same device to every engine.

- `Engine::availability` says whether the engine's software can be used: `Loaded` (with a version
  and the folder it came from, when it has them) or the typed `EngineUnavailable` (library missing,
  library not loadable, no compute backend). An engine linked into the executable is always loaded.
- `accel::engine_report::EngineReport::of(engine)` is one engine's whole report: availability, its
  `Inventory` (build, devices, runtime libraries) and the devices a run can use. `reports(engines,
  expected)` makes one per engine, and one marked `NotBuilt` for each `expected` engine that is not
  in the list. Only a loaded engine lists devices; asking it is what loads its library.
- `accel::hardware::merge(reports)` is the union that `purr sys` and the Hardware page show: one
  `HardwareDevice` per physical device, with a `Sighting` (engine, the engine's own key and index)
  for every engine that lists it. Two entries are the same device when the backend and name agree
  and, if both report a PCI id, the ids agree, else the position among devices of that name agrees;
  the processor is always one device. A merged key is the first engine's key and is unique in the
  list. `hardware::backend_status` says a backend runs when any loaded engine runs it.
- The union is for reporting and for the user's order. A run never builds it: `open_on_best_device`
  asks only the engine that runs the model, so the plan, `OpenSpec::devices` and the device indexes
  are that engine's. A device of the saved order that this engine does not list is a skipped step
  with the engine's own reason, recorded in the `AccelReport`.
- `--device` is resolved against the engine that runs the model (`DeviceSelector::resolve` takes its
  `EngineDevices` and the other engines'). Matching nothing there is `SelectorError::NoMatch`, or
  `SelectorError::NotOnEngine` when another engine lists a device the selector describes; an index is
  never looked for in another engine's list.

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
- A model whose text carries no times (`Timing::None`) gives segments without spans, and the
  segmenter places each over its whole window. With a speech detector the window ends in a pause
  (`AtPauses`), so a chunk's span is the stretch of audio between two pauses; the CLI loads the
  Silero model by default, the app after its background download. Without a detector the windows
  are `Full`: the chunks are as wide as the window (30 s) and say so.
- `Windowing::Whole` on a decoder with `InputLimit::UpTo` streams, as before, with windows no
  longer than the streaming window or the limit, whichever is shorter. The limit of a language model
  is the context it shares between audio and text (87 minutes for Qwen3-ASR): a decode that long
  would run out of room for its text, so the limit bounds a window and does not enlarge it.

### When the engine stops early

An engine that returns `Truncated` or `Repeating` for a window gave text but no end: for a model
without times, nothing tells where in the audio the text stops. The session therefore decodes the
window again at half its length, from the same start, ending in the last pause of that half when
a detector is loaded (the window is then `Paused`, else `Full`), and drops the first attempt's
text. It stops halving when half a window would be under 5 s, and then commits the text it has and
advances over the whole window, with a warning in the log. A `Last` window that is cut stops being
`Last`, so the rest of the audio is still decoded. Tentative text is decoded once, never again.
Each failed attempt costs a full encode of its window.

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
- Beam search with 5 beams, always. whisper.cpp draws its beam candidates from a random generator
  kept in the state and seeded once, so each decode runs on a state no earlier decode used (the one
  made when the model opens serves the first); otherwise tentative decodes would change the text a
  window commits.

## transcribe.cpp (`purr-engine-transcribe`)

- The two engines never share a ggml. whisper.cpp's is linked statically into the executable;
  transcribe.cpp is built by `transcribe-cpp-sys` with its `dynamic-backends` feature: `transcribe.dll`
  (`libtranscribe.so`) with `ggml` and `ggml-base` next to it, and each CPU instruction set tier (and
  Vulkan, when built) as a loadable module that the library registers at run time. On Windows the
  executable imports nothing from `ggml.dll`, only `transcribe.dll`.
- Nothing of transcribe.cpp may be called before `runtime::get()` has opened the library. It opens
  `transcribe.dll` from the runtime folders of [runtime-libraries.md](runtime-libraries.md), never through
  the system's search, then has transcribe.cpp load its backend modules from that folder. See there for why.
- `id`, `build_info` and `install_logging` do not touch the library (`install_logging` only records the
  request; the log callback is set right after the library is loaded). `devices` loads it and lists the
  devices of transcribe.cpp's own registry, which are not the ones `--device` indexes for a whisper.cpp
  model. `availability()` is the typed outcome that `purr sys` prints under "Engines".
- `open` loads the GGUF with `Backend::Cpu`: the build has no GPU backend module, so a plan candidate
  of any other backend is `OpenError::NoDevice`. A load that fails in a backend is `OpenError::Device`;
  any other failure stops the device search with `OpenError::Engine`.
- `open_speech_detector` loads the same Silero `.bin` the Whisper models use, through transcribe.cpp.
- One decoder (`TranscribeDecoder`) serves every family, driven by what the loaded model says:
  `capabilities().languages` is the list `supports()` reads and the codes a hint goes out in (the
  library compares the hint with its list as text, so `fr-CH` is sent as `fr`, whatever the model's
  spelling); `session.limits().effective_max_audio_ms` is the input limit (0: `Unbounded`);
  timestamps are asked for as `Auto`, the finest the model gives, and the kind it returns picks the
  `Segmentation`: words (Parakeet) are `Synthesize`, coarser or none are `Native`. A
  `LanguageRequest::Use` is passed to the library as the hint and a `Detect` reports the language the
  library names. Which models a hint steers is the profile's `language_hint`, applied before the run
  by the app and the CLI (`LanguageChoice::for_profile`, `language_arg::check`), so a hint that
  reaches the decoder is one the model takes.

### Parakeet in this design

- `limits()`: input from the model's `max_audio_ms` (0 for Parakeet: `Unbounded`, so `--no-stream`
  decodes the whole source in one call) and a 30 s streaming window. The library switches the
  full-attention checkpoint to local attention (256, 256), so time grows linearly with the audio,
  and so does the compute buffer: about 0.35 MB per second of audio, which is 1.7 GB peak for a
  three-minute file in one call and an estimated 12 GB for an hour.
- Segments are `Synthesize`: one segment per call, split by the pipeline from the word times.
  Words come from the library's words, spelled by their tokens (which carry the space before a
  word), timed at 80 ms resolution, scored as their least certain token.
- Language: the model finds it by itself and the library does not report it
  (`transcribe_detected_language` is empty for Parakeet), so the profile says `language_hint:
  ignored` and `detection: silent`. The app and the CLI drop a hint for it, so none is passed;
  `supports()` reads the model's own list of 25 languages. The stream stays in language detection and chunks carry no language.
- Cancel: the library checks its abort callback once at the top of each run, and its token cannot be
  linked to Purr's, so `decode` returns `Cancelled` before it starts when the token is set and cannot
  stop a run already going. Four threads decode about 17 seconds of audio per second, so a window of
  30 s takes under two.
- Errors: `InputTooLong`, `InputTooShort` and `OutOfMemory` map to their `DecodeError` arms,
  everything else to `DecodeError::Engine`. A truncated or repeating output is a `Completion` with
  the partial transcript the library keeps, not an error.
- Models: three GGUF files of `parakeet-tdt-0.6b-v3` (F16, Q8_0, Q4_K_M) from the `handy-computer`
  repository pinned in the manifest, CC BY 4.0. `ModelFamily::attribution` carries who made a
  family and what the downloaded files change; `purr models info` prints it and the app's Licenses
  page lists it (`views::licenses::models`). Nothing else in the app or the CLI listings shows it.

### Qwen3-ASR in this design

- An audio encoder in front of a Qwen3 language model, two sizes (0.6B, 1.7B), 30 languages,
  Apache-2.0, three GGUF files each (F16, Q8_0, Q4_K_M) from the `handy-computer` repositories
  pinned in the manifest. The profile: the hint `steers` (a forced language is written into the
  prompt), the language is `reported` by name and the library maps it to a tag, `timing` is `none`,
  no word scores, `input` is `up-to` 5218 s, the library's `effective_max_audio_ms`.
- The library reports no confidence, so a stream locks on the first window with speech. A wrong
  first window (a few words of noise, a name) forces its language on the rest of the stream; the
  language setting is the fix.
- The output is free text from a language model, so it can stop short (the generation budget) or
  loop. Both come back as `Completion::Truncated` / `Repeating` with the partial text.
- Cancel is checked once before a run, as for Parakeet.

## What the app and the CLI take from the model

Nothing in the screens or the commands names an engine or a family to decide what to show; they
read the model's descriptor and `ModelProfile`, and the engine is the model's (`ModelDescriptor::engine`,
`ModelRef::engine`). There is no engine setting and no `--engine` flag: a model file has one format,
a format has one engine (`EngineId::loading`), and `ModelManager::reference` reads the engine from
the catalog or from the file's first bytes.

- Language control (`transcribe::language_choice`): `Only(list)` offers exactly that list; a model
  with `language_hint: ignored` disables the Spoken language pill, which shows "Automatic" with the
  tooltip "<model> detects the language itself". The run uses Automatic for such a model
  (`LanguageChoice::for_profile`) and the saved language stays in settings for other models. One with
  `detection: silent` never makes a run wait for a detected language and every place that shows a
  language (More menu, History rows, JSON export) says "not reported" or shows nothing. Stored
  languages are rewritten under their current spelling when settings load
  (`Settings::canonicalize_languages`).
- CLI `--language` is checked against the catalog model's profile before the model loads
  (`language_arg::check`): a language the model does not list is `LanguageProblem::Unsupported`, one it
  lists but ignores is a warning and the hint is dropped from the run. A file outside the catalog has
  no profile, so the pipeline's own `supports` check applies.
- Models (`model_catalog`): sections per size for Whisper and one each for Parakeet and Qwen3-ASR. The picker heads
  each line of models when it lists several. A model whose engine is not in `engines::all()` shows
  "Not in this build" and cannot be picked. Licences are not shown there: the Licenses page has a
  Models section with the credit of every family whose licence asks for it
  (`ModelLicense::requires_attribution`), read from `ModelFamily::attribution`.
- Devices: the Transcribe screen's device override offers the devices the selected model's engine can
  use (`DeviceRow::usable_by`, from `seen_by`); a pick that the new model's engine cannot use is dropped.
- An engine that is built in but cannot be used when a run starts (`Engine::availability`) fails the
  run with `TranscriptionError::EngineUnavailable` / `PlatformError::EngineUnavailable`, which names the
  folders searched; the Transcribe screen shows it as the failure message with the reason as its hint.

## Adding an engine

1. Create `crates/purr-engine-<name>` depending on `purr-core` and `purr-common`, with its own
   cargo features for the GPU backends it builds; forward `cuda` and `vulkan` to `purr-core` so the
   delay-load hook and the runtime-library names are compiled in.
2. Implement `Engine`, `WindowDecoder` and, if it has one, `SpeechDetector`; convert the engine's
   own error enum into `OpenError`, `DecodeError` and `DetectError`.
3. Report honest `limits()`, and `Segmentation::Synthesize` when the engine gives long stretches with
   timed words.
4. Add it to `engines::all()` in `purr-cli` and `purr-app`, and forward the binaries' features to it.
5. Give its catalog models that `EngineId`, so the app and `ModelManager::reference` pick it; a file
   outside the catalog is routed by `EngineId::loading(ModelFormat::of_header(..))`. A model whose
   engine is not in `engines::all()` is `Unavailable` in the app (`ModelInfo::engine_built`) and
   skipped by `find_first_available_model`.
6. Run the Whisper goldens (`just test-models -p purr-engine-whisper --test real_model_golden`) to
   show the shared pipeline did not move. What else to run, and what holds today, is in
   [verification.md](verification.md).
