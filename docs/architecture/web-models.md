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

## Variants

Every model is offered as up to two variants, each with its own ONNX files, sizes and digests:

- `fp16`: the half-precision encoder (`encoder_model_fp16.onnx`, or the model's quantized encoder
  where that is what it uses) and the q4 merged decoder whose tied embedding table is stored once as
  half precision. It needs the WebGPU `shader-f16` feature.
- `fp32`: the encoder as it was (full precision or quantized) and the same decoder with its table in
  full precision. This is the only variant the WebAssembly CPU runtime runs.

Rust picks the variant (`compute_choice.rs`, `VariantKind`): WebGPU takes `fp16` when the adapter has
`shader-f16` and `fp32` otherwise; the CPU, and a run that falls back to it after WebGPU failed, take
`fp32` and install it first. Only the variant that will run is installed. The two variants of a model
share their configuration files and, for a quantized encoder, the encoder; the decoder has the same
cache key in both, so installing one replaces the other, and a file counts as installed only when its
cache entry has the length the manifest expects. The Models page shows the size and installed state
of the variant a run would use.

Models without a variant that does not need `shader-f16` list only `fp16`: `large-v3-turbo` (an fp32
encoder would be about 2.4 GB) and `large-v3-turbo-q4f16`, which is half precision throughout. The
quantized-encoder models (`*-q4`) have both variants: they differ in the decoder.

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
- With the half-table option a table that the nodes show to be tied (the `Gemm` form above) is
  rounded to half precision, to nearest even, as its bytes stream through in the same 1 MiB chunks,
  and declared `FLOAT16`. Each branch's lookup then reads the half table and a `Cast` to full precision
  produces the original output; the head casts the hidden state to half, runs the `Gemm` and casts the
  logits back. The table is never cast on its own: ONNX Runtime would fold that into a full-precision
  copy. Which tensor is the table must be known before its bytes arrive, and the graph's nodes are
  serialized before its initializers, so it is decided from the nodes read so far. A table that anything
  else reads, or that would overflow half precision, is an error or stays in full precision, and a
  model whose nodes follow its weights is rejected.
- An encoder's self-attention, exported as separate operators, keeps two full `[batch * heads, S, S]`
  tensors per layer in memory: the scores and their softmax (68.7 MiB each for Base, in full
  precision). Each block is replaced by one `com.microsoft.MultiHeadAttention` node (`attention_fusion.rs`),
  which ONNX Runtime Web runs on WebGPU and the CPU without them. The block is recognised by
  operators and edges: `Softmax` over the last axis of a `MatMul` of the scaled query and the
  transposed key, a `MatMul` with the values, then the heads merged back by `Reshape` and `Transpose`.
  Two layouts of the heads occur: folded into the batch (`[batch * heads, S, size]`, with a regrouping
  `Reshape` on the way back; Tiny to Small) and kept as an axis (`[batch, heads, S, size]`; Large V3
  Turbo). The head count and size come from the `Reshape` shape constants and
  must agree across query, key, value and the merge, which must be as wide as heads times size. The
  node takes the unscaled query; the export's scale constant (a scalar of either precision) is the
  node's default `1/sqrt(head_size)` or, when it differs, passed as `scale`. The same pattern holds
  in the half-precision encoders (their constants are `float16`), so both variants are fused. Nodes
  that nothing reads afterwards are removed by walking back from the graph outputs, subgraph
  reads included. A graph without the pattern (the decoders) is
  left as it is. The quantized encoders keep float attention, so they are fused as well. It works on the graph the converter already holds, so the weights stream untouched.
  Measured on Base in headless Chrome with WebGPU, it took the GPU buffer peak from 617 to 357 MiB
  and the encoder output stays within 5e-4 of the original on the CPU.
- The data is written first and the graph last, because the graph holds the offsets. The output is
  deterministic.

The crate also builds as a `cdylib` for `wasm32-unknown-unknown` with a C interface (`src/abi.rs`):
the host supplies `purr_read`, `purr_write`, `purr_location`, `purr_table_precision` and
`purr_report_error`, and calls
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

The transcription worker never downloads: `js/offline.js` replaces transformers.js's `fetch` with a
lookup in the cache (a miss is a 404, and the few requests it makes for `main` are answered from the
pinned revision), and the pipeline is created with `use_external_data_format`, so it reads the graph
and `<path>_data` from the cache. The Hugging Face token is used only by the page's downloads.

## Developer models

With developer mode on (Settings, Advanced) a model can be loaded without a manifest entry: from a
Hugging Face repository and revision, or from files picked on disk. `developer_install.rs` stores
the files without any hash check, converting the ONNX files with the same converter only when asked.

- A developer model's cache keys use a repository of their own, `purr-developer/<id>`, so they never
  collide with a catalog model of the same source. Verification, eviction and `is_installed` of the
  catalog never look at them.
- A developer model is a `WebModel` of one variant and no files (`developer_model::web_model`): the
  dtypes and, when converting, the table precision the user chose decide whether that variant is
  `fp16` or `fp32`. Nothing switches between precisions for it, not even the WebGPU to CPU fallback,
  and the choice of the converter's table precision (full or half) is part of the model.
- Files used as they are keep their external weights only when the repository or the picked files
  have them; the worker then asks transformers.js for external data per ONNX file.
- The session options typed in the settings are merged into the ones `model_loader.js` builds:
  objects key by key, execution providers by name, anything else replaced.
- The worker collects the developer panel figures only when started with developer mode on
  (`developer.js`): it wraps `WebAssembly.Memory`, `WebAssembly.instantiate*`,
  `GPUDevice.prototype.createBuffer` and `GPUBuffer.prototype.destroy` before ONNX Runtime starts.
  The GPU figure counts buffers allocated and not yet destroyed, since browsers expose no real VRAM
  figure. An idle worker started the other way is replaced when the setting changes.

## Manifest

Schema 4. A model has `files` (configuration and tokenizer) and `variants`. A variant has its
`precision` (`fp16` or `fp32`), the encoder and decoder dtypes, `table` (`fp16` when the converter
stores the tied table as half precision; omitted for `fp32`) and its ONNX `files`. Each file has
`size_bytes` and `sha256` of the original. Each ONNX file also has `converted.graph` and
`converted.data`, each with `size_bytes` and `sha256`; the registry rejects an ONNX file without them
and a plain file with them, and an `fp32` variant that needs `shader-f16`. Never edit these by hand:

```
cargo run -p purr-core --example update_web_manifest [-- --bump] [--reconvert]
```

downloads every ONNX file of every variant (a few GB; kept under `target/web-model-cache`), verifies
it against its digest and runs the converter with the variant's `table` to record the outputs. A file
whose original digest is unchanged keeps its recorded conversion unless `--reconvert` is given; pass it
after changing a variant's `table` or the converter.

## Tests

- `cargo nextest run -p purr-onnx-convert`: synthetic models for the split and the rewrites (full and
  half table, and the attention fusion: match, no match, scale, subgraph reads), and a counting allocator that bounds the converter's peak memory.
- `cargo nextest run -p purr-onnx-convert --features tooling --profile all`: the `real_model_` tests
  fetch `tiny`, check the converter's output against the manifest, and run
  `scripts/validate_onnx_conversion.py` (via `uv`) to compare encoder output and every decoder logit
  with the originals in ONNX Runtime on the CPU. The `fp32` conversion must give the original's
  decoder logits bit for bit and an encoder output within 2e-3 of it (the attention is fused); the
  half-table decoder must pick the same token on at least 98% of the teacher-forced steps with no logit
  moving by more than 0.25.
