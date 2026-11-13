import "./console_filter.js";
import { useHuggingFaceToken } from "./hugging_face_auth.js";
import { emit, emitFailure, FailureKind, WorkerFailure } from "./events.js";
import { asProviderFailure, configureRuntime, loadTranscriber, Provider } from "./model_loader.js";
import { transcribe } from "./transcriber.js";
import { configureWarmUp, warmUp } from "./warm_up.js";
import { keepWebGpuPolled } from "./webgpu_polling.js";

keepWebGpuPolled();

let queue = Promise.resolve();

// Firefox builds every WebGPU pipeline again on each page load, one at a time: https://bugzilla.mozilla.org/show_bug.cgi?id=2083378
async function readyTranscriber(message) {
    useHuggingFaceToken(message.hugging_face_token);
    const transcriber = await loadTranscriber(message.model, message.provider, message.request_id);
    if (message.provider === Provider.WebGpu) {
        try {
            await warmUp(transcriber, message.english_only);
        } catch (error) {
            console.warn("Warming up the GPU failed; the run itself will report the cause", error);
        }
    }
    return transcriber;
}

async function handle(message) {
    switch (message.type) {
        case "Configure":
            configureRuntime(message);
            configureWarmUp(message.warm_up_url);
            return;
        case "LoadModel":
            useHuggingFaceToken(message.hugging_face_token);
            await loadTranscriber(message.model, message.provider, message.request_id);
            return;
        case "Transcribe": {
            const transcriber = await readyTranscriber(message);
            emit({ type: "Transcribing", request_id: message.request_id });
            try {
                await transcribe(transcriber, message);
            } catch (error) {
                throw asProviderFailure(message.provider, error);
            }
            return;
        }
        default:
            throw new WorkerFailure(
                FailureKind.Protocol,
                `Unknown worker request type: ${message.type}`,
            );
    }
}

function decode(data) {
    const message = JSON.parse(data.request);
    if (data.audio !== undefined) {
        message.audio = data.audio;
    }
    return message;
}

self.addEventListener("message", (event) => {
    let message = null;
    queue = queue
        .then(() => {
            message = decode(event.data);
            return handle(message);
        })
        .catch((error) => {
            console.error("Worker request failed", error);
            emitFailure(message?.request_id ?? null, error);
        });
});

self.addEventListener("unhandledrejection", (event) => {
    console.error("Unhandled rejection in worker", event.reason);
    emitFailure(null, event.reason);
});
