# Browser models

The browser platform (`purr-wasm`) runs Whisper through transformers.js. Its models are pinned in
`crates/purr-common/manifests/web-models.json` and installed by Rust, not by transformers.js.

## Why files are converted

iOS kills a Safari tab at 2,048 MB, and two things in the stock `onnx-community` files pushed the
WebGPU load over it:

- A merged decoder has an `If` node whose branches each transpose the tied embedding table for the
  output `MatMul`. ONNX Runtime folds each `Transpose` into a copy of the table, so it is uploaded to
  the GPU three times.
- Weights inside the `.onnx` pass through the WebAssembly heap about three times while loading. With
  external data, ONNX Runtime Web uploads them from JavaScript to the GPU directly.

## The converter

`crates/purr-onnx-convert` reads an ONNX file once, as a stream, and writes two files: a graph
(`<name>.onnx`) and its weights (`<name>.onnx_data`).

- It reads with `rust-protobuf` over the vendored `onnx.proto3` (tag `v1.23.2`, Apache-2.0 licence
  kept beside it). Initializers of 1 KiB or more are copied to the data sink in 1 MiB chunks, aligned
  to 4 KiB, and their entry in the graph points at `location`, `offset` and `length`. The rest of the
  model is the small graph held in memory.
- Where an `If` branch transposes an initializer for a `MatMul` and also gathers rows from it, the
  `Transpose` and `MatMul` become `Reshape`, `Gemm` with `transB`, `Reshape`. The result is bit
  identical on the CPU. A model without that pattern passes through unchanged.
- The data is written first and the graph last, because the graph holds the offsets. The output is
  deterministic.

The crate also builds as a `cdylib` for `wasm32-unknown-unknown` with a C interface (`src/abi.rs`):
the host supplies `purr_read`, `purr_write`, `purr_location` and `purr_report_error`, and calls
`purr_convert`. `purr-wasm`'s `build.rs` compiles it in its own target directory and ships it as
`dist/purr_onnx_convert.wasm` next to `dist/convert_worker.js`.

## Installing a model

`model_install.rs` downloads every manifest file from Hugging Face and writes the results into the
`transformers-cache` Cache Storage under the keys transformers.js asks for.

1. The original is hashed while it streams in and must match the manifest.
2. Small files are held in memory and stored as they are.
3. An ONNX file is fed to a short-lived worker (`js/convert_worker.js`, `converter_worker.rs`)
   running the converter module. The page writes into a 4 MiB ring in a `SharedArrayBuffer`; the
   worker reads it synchronously, blocking in `Atomics.wait`. Outputs return as transferred
   messages, and the page acknowledges each one so the worker stays within 8 MiB of what has been
   stored. The page needs cross-origin isolation for this. The worker is terminated afterwards.
4. The data output streams into a `TransformStream` whose readable side is the body of the cache
   entry. Its size and hash must match `converted.data` before the stream closes; otherwise it is
   aborted and nothing is stored. The graph is stored last, after its own check.

A model counts as installed when every file is cached under all its keys, a converted file as both
the graph and `<path>_data`. A cache filled by an older version (unconverted `.onnx` only) therefore
reads as not installed and is installed again. `model_cache::verify` hashes every cached entry
against the sizes and digests the manifest records for what is stored.

The transcription worker never downloads: `env.allowRemoteModels` is off and the pipeline is created
with `use_external_data_format`, so transformers.js reads the graph and `<path>_data` from the cache.

## Manifest

Schema 3. Each file has `size_bytes` and `sha256` of the original. Each ONNX file also has
`converted.graph` and `converted.data`, each with `size_bytes` and `sha256`; the registry rejects an
ONNX file without them and a plain file with them. Never edit these by hand:

```
cargo run -p purr-core --example update_web_manifest [-- --bump] [--reconvert]
```

downloads every ONNX file (a few GB; kept under `target/web-model-cache`), verifies it against its
digest and runs the converter to record the outputs. A file whose original digest is unchanged keeps
its recorded conversion unless `--reconvert` is given.

## Tests

- `cargo nextest run -p purr-onnx-convert`: synthetic models for the split and the rewrite, and a
  counting allocator that bounds the converter's peak memory.
- `cargo nextest run -p purr-onnx-convert --features tooling --profile all`: the `real_model_` tests
  fetch `tiny`, check the converter's output against the manifest, and run
  `scripts/validate_onnx_conversion.py` (via `uv`) to compare encoder output and every decoder logit
  with the originals in ONNX Runtime on the CPU.
