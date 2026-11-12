import { env } from "@huggingface/transformers";
import { FailureKind, WorkerFailure } from "./events.js";

async function requestAdapter() {
    if (!self.navigator?.gpu) {
        return null;
    }
    return navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
}

function limits(adapter) {
    const values = {};
    for (const key in adapter.limits) {
        values[key] = adapter.limits[key];
    }
    return values;
}

export async function probeGpu() {
    const adapter = await requestAdapter();
    if (!adapter) {
        return null;
    }
    const info = adapter.info ?? {};
    const fingerprint = JSON.stringify({
        vendor: info.vendor ?? null,
        architecture: info.architecture ?? null,
        device: info.device ?? null,
        description: info.description ?? null,
        features: [...adapter.features].sort(),
        limits: limits(adapter),
        browser: navigator.userAgent,
        runtime: env.backends.onnx.versions ?? null,
        transformers: env.version,
    });
    return { fingerprint };
}

export async function requireWebGpu(model) {
    if (!self.navigator?.gpu) {
        throw new WorkerFailure(
            FailureKind.WebGpuUnavailable,
            "WebGPU is not available in this browser",
        );
    }
    const adapter = await requestAdapter();
    if (!adapter) {
        throw new WorkerFailure(
            FailureKind.WebGpuUnavailable,
            "The browser found no WebGPU adapter on this device",
        );
    }
    if (model.requires_shader_f16 && !adapter.features.has("shader-f16")) {
        throw new WorkerFailure(
            FailureKind.ShaderF16Unavailable,
            "This model needs the WebGPU shader-f16 feature, which this GPU or browser does not provide",
        );
    }
}
