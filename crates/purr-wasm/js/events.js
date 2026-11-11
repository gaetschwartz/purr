export const FailureKind = Object.freeze({
    WebGpuUnavailable: "WebGpuUnavailable",
    ShaderF16Unavailable: "ShaderF16Unavailable",
    ModelLoad: "ModelLoad",
    Inference: "Inference",
    Protocol: "Protocol",
});

export class WorkerFailure extends Error {
    constructor(kind, message) {
        super(message);
        this.kind = kind;
    }
}

export function emit(event) {
    self.postMessage(JSON.stringify(event));
}

export function emitFailure(requestId, error) {
    const kind = error instanceof WorkerFailure ? error.kind : FailureKind.Inference;
    emit({
        type: "Failed",
        request_id: requestId,
        kind,
        message: error instanceof Error ? error.message : String(error),
    });
}
