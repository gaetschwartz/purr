import { FailureKind, WorkerFailure } from "./events.js";

export async function requireWebGpu(model) {
    if (!self.navigator?.gpu) {
        throw new WorkerFailure(
            FailureKind.WebGpuUnavailable,
            "WebGPU is not available in this browser",
        );
    }
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
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
