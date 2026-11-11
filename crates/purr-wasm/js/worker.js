import "./console_filter.js";
import { emit, emitFailure, FailureKind, WorkerFailure } from "./events.js";
import { configureRuntime, loadTranscriber } from "./model_loader.js";
import { transcribe } from "./transcriber.js";

let queue = Promise.resolve();

async function handle(message) {
    switch (message.type) {
        case "Configure":
            configureRuntime(message);
            return;
        case "LoadModel":
            await loadTranscriber(message.model, message.request_id);
            return;
        case "Transcribe": {
            const transcriber = await loadTranscriber(message.model, message.request_id);
            emit({ type: "Transcribing", request_id: message.request_id });
            await transcribe(transcriber, message);
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
