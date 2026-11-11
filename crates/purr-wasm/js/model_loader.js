import { env, pipeline } from "@huggingface/transformers";
import { emit, FailureKind, WorkerFailure } from "./events.js";
import { requireWebGpu } from "./gpu.js";

env.allowLocalModels = false;
env.backends.onnx.logLevel = "error";

let loaded = { key: null, promise: null };

export function configureRuntime(setup) {
    env.backends.onnx.wasm.wasmPaths = {
        mjs: setup.ort_module_url,
        wasm: setup.ort_wasm_url,
    };
}

function modelKey(model) {
    return `${model.repo}@${model.revision}:${model.encoder_dtype}/${model.decoder_dtype}`;
}

function progressReporter(requestId) {
    return (data) => {
        switch (data.status) {
            case "progress":
                emit({
                    type: "ModelFile",
                    request_id: requestId,
                    file: data.file,
                    loaded: Math.round(data.loaded),
                    total: Math.round(data.total),
                });
                break;
            case "done":
                emit({ type: "ModelFileDone", request_id: requestId, file: data.file });
                break;
            default:
                break;
        }
    };
}

async function release(entry) {
    if (entry.promise === null) {
        return;
    }
    try {
        const instance = await entry.promise;
        await instance.dispose();
    } catch (error) {
        console.warn("Releasing the previous model failed", error);
    }
}

export async function loadTranscriber(model, requestId) {
    await requireWebGpu(model);
    const key = modelKey(model);
    if (loaded.key !== key) {
        const previous = loaded;
        loaded = { key: null, promise: null };
        await release(previous);
        loaded = {
            key,
            promise: pipeline("automatic-speech-recognition", model.repo, {
                revision: model.revision,
                dtype: {
                    encoder_model: model.encoder_dtype,
                    decoder_model_merged: model.decoder_dtype,
                },
                device: "webgpu",
                session_options: { logSeverityLevel: 3 },
                progress_callback: progressReporter(requestId),
            }),
        };
    }
    try {
        const transcriber = await loaded.promise;
        emit({ type: "ModelReady", request_id: requestId });
        return transcriber;
    } catch (error) {
        loaded = { key: null, promise: null };
        throw new WorkerFailure(
            FailureKind.ModelLoad,
            `Loading ${model.repo} failed: ${error instanceof Error ? error.message : error}`,
        );
    }
}
