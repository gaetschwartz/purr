import { FailureKind, WorkerFailure } from "./events.js";

const SHADER_F16_DTYPES = new Set(["fp16", "q4f16"]);

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
    const needsF16 = [model.encoder_dtype, model.decoder_dtype].some((dtype) =>
        SHADER_F16_DTYPES.has(dtype),
    );
    if (needsF16 && !adapter.features.has("shader-f16")) {
        throw new WorkerFailure(
            FailureKind.ShaderF16Unavailable,
            "This model needs the WebGPU shader-f16 feature, which this GPU or browser does not provide",
        );
    }
}
