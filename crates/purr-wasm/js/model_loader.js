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

export const Provider = Object.freeze({
    WebGpu: "WebGpu",
    Cpu: "Cpu",
});

const DEVICES = Object.freeze({
    [Provider.WebGpu]: "webgpu",
    [Provider.Cpu]: "wasm",
});

function modelKey(model, provider) {
    return `${model.repo}@${model.revision}:${model.encoder_dtype}/${model.decoder_dtype}:${provider}`;
}

export function asProviderFailure(provider, error) {
    if (provider !== Provider.WebGpu || error instanceof WorkerFailure) {
        return error;
    }
    return new WorkerFailure(
        FailureKind.WebGpuFailed,
        `WebGPU failed: ${error instanceof Error ? error.message : error}`,
    );
}

const PROGRESS_INTERVAL_MS = 100;

function wholeNumber(value) {
    return Number.isFinite(value) ? Math.round(value) : 0;
}

function progressReporter(requestId) {
    const lastSent = new Map();
    return (data) => {
        switch (data.status) {
            case "progress": {
                const now = performance.now();
                if (now - (lastSent.get(data.file) ?? 0) < PROGRESS_INTERVAL_MS) {
                    break;
                }
                lastSent.set(data.file, now);
                emit({
                    type: "ModelFile",
                    request_id: requestId,
                    file: data.file,
                    loaded: wholeNumber(data.loaded),
                    total: wholeNumber(data.total),
                });
                break;
            }
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

export async function loadTranscriber(model, provider, requestId) {
    const device = DEVICES[provider];
    if (device === undefined) {
        throw new WorkerFailure(FailureKind.Protocol, `Unknown execution provider: ${provider}`);
    }
    if (provider === Provider.WebGpu) {
        await requireWebGpu(model);
    }
    const key = modelKey(model, provider);
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
                device,
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
        const message = `Loading ${model.repo} failed: ${error instanceof Error ? error.message : error}`;
        throw new WorkerFailure(
            provider === Provider.WebGpu ? FailureKind.WebGpuFailed : FailureKind.ModelLoad,
            message,
        );
    }
}
