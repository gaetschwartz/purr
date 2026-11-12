import "./console_filter.js";
import { configureCanary, transcribeCanary } from "./canary.js";
import { useHuggingFaceToken } from "./hugging_face_auth.js";
import { emit, emitFailure, FailureKind, WorkerFailure } from "./events.js";
import { probeGpu } from "./gpu.js";
import { asProviderFailure, configureRuntime, loadTranscriber, Provider } from "./model_loader.js";
import { transcribe } from "./transcriber.js";
import { keepWebGpuPolled } from "./webgpu_polling.js";

keepWebGpuPolled();

let queue = Promise.resolve();

async function handle(message) {
    switch (message.type) {
        case "Configure":
            configureRuntime(message);
            configureCanary(message.canary_url);
            return;
        case "LoadModel":
            useHuggingFaceToken(message.hugging_face_token);
            await loadTranscriber(message.model, message.provider, message.request_id);
            return;
        case "ProbeGpu":
            emit({ type: "GpuProbed", request_id: message.request_id, probe: await probeGpu() });
            return;
        case "TranscribeCanary": {
            useHuggingFaceToken(message.hugging_face_token);
            const transcriber = await loadTranscriber(
                message.model,
                message.provider,
                message.request_id,
            );
            try {
                if (message.warm_up) {
                    await transcribeCanary(transcriber, message.english_only);
                }
                const startedAt = performance.now();
                const text = await transcribeCanary(transcriber, message.english_only);
                emit({
                    type: "CanaryTranscribed",
                    request_id: message.request_id,
                    text,
                    elapsed_ms: performance.now() - startedAt,
                });
            } catch (error) {
                throw asProviderFailure(message.provider, error);
            }
            return;
        }
        case "Transcribe": {
            useHuggingFaceToken(message.hugging_face_token);
            const transcriber = await loadTranscriber(
                message.model,
                message.provider,
                message.request_id,
            );
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
