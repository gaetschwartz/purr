//! Browser implementation of the Purr platform: audio decoding with the Web Audio API,
//! transcription with Whisper through transformers.js in a Web Worker, and model caching in
//! the browser's Cache Storage.

use manganis::{Asset, AssetOptions, asset};

pub mod audio;
pub mod byte_stream;
pub mod compute_choice;
pub mod converter_worker;
pub mod developer;
pub mod developer_files;
pub mod developer_install;
pub mod developer_model;
pub mod developer_run;
pub mod error;
pub mod history_audio;
pub mod history_store;
pub mod idb;
pub mod integrity;
pub mod local;
pub mod logging;
pub mod model;
pub mod model_cache;
pub mod model_info;
pub mod model_install;
pub mod model_progress;
pub mod platform;
pub mod protocol;
pub mod status_mapper;
pub mod storage;
pub mod storage_space;
pub mod worker_client;

static WORKER_SCRIPT: Asset = asset!(
    "/dist/worker.js",
    AssetOptions::js().with_minify(false).into_asset_options()
);

static ORT_MODULE: Asset = asset!(
    "/node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.asyncify.mjs",
    AssetOptions::js().with_minify(false).into_asset_options()
);

static ORT_WASM: Asset =
    asset!("/node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.asyncify.wasm");

/// URL paths of the ONNX runtime files the worker loads.
#[must_use]
pub fn ort_urls() -> (String, String) {
    (ORT_MODULE.to_string(), ORT_WASM.to_string())
}

static CONVERTER_WORKER_SCRIPT: Asset = asset!(
    "/dist/convert_worker.js",
    AssetOptions::js().with_minify(false).into_asset_options()
);

static CONVERTER_WASM: Asset = asset!("/dist/purr_onnx_convert.wasm");

/// Absolute URL paths of the script of the worker that converts ONNX files while a model installs, and of its WebAssembly module.
#[must_use]
pub fn converter_urls() -> (String, String) {
    (
        CONVERTER_WORKER_SCRIPT.to_string(),
        CONVERTER_WASM.to_string(),
    )
}

static WARM_UP: Asset = asset!("/assets/warm_up.pcm");

/// URL path of the recording the worker transcribes to warm up a WebGPU model: 16 kHz mono signed 16-bit samples.
#[must_use]
pub fn warm_up_url() -> String {
    WARM_UP.to_string()
}

/// Absolute URL path of the bundled transcription worker script.
#[must_use]
pub fn worker_url() -> String {
    WORKER_SCRIPT.to_string()
}

/// Installs the panic hook. Logging is set up by the application, see [`logging`].
pub fn init() {
    console_error_panic_hook::set_once();
}
