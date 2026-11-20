const SAMPLE_INTERVAL_MS = 100;

const memories = new Set();
const buffers = new WeakMap();
const state = {
    installed: false,
    model: null,
    provider: null,
    wasmPeak: 0,
    jsHeapPeak: 0,
    gpuBytes: 0,
    gpuBytesPeak: 0,
    gpuLive: 0,
    gpuLivePeak: 0,
    adapter: null,
};

function wasmHeapBytes() {
    let total = 0;
    for (const memory of memories) {
        total += memory.buffer.byteLength;
    }
    return total;
}

function jsHeap() {
    const memory = self.performance?.memory;
    return memory ? { used: memory.usedJSHeapSize, limit: memory.jsHeapSizeLimit } : null;
}

function sample() {
    state.wasmPeak = Math.max(state.wasmPeak, wasmHeapBytes());
    const heap = jsHeap();
    if (heap) {
        state.jsHeapPeak = Math.max(state.jsHeapPeak, heap.used);
    }
}

function captureMemory(value) {
    const memory = value?.exports?.memory ?? value?.instance?.exports?.memory;
    if (memory instanceof WebAssembly.Memory) {
        memories.add(memory);
    }
    return value;
}

function hookWebAssembly() {
    const NativeMemory = WebAssembly.Memory;
    WebAssembly.Memory = new Proxy(NativeMemory, {
        construct(target, args, newTarget) {
            const memory = Reflect.construct(target, args, newTarget);
            memories.add(memory);
            return memory;
        },
    });
    for (const name of ["instantiate", "instantiateStreaming"]) {
        const native = WebAssembly[name];
        if (typeof native === "function") {
            WebAssembly[name] = (...args) => native.apply(WebAssembly, args).then(captureMemory);
        }
    }
}

function hookGpuBuffers() {
    if (typeof GPUDevice === "undefined" || typeof GPUBuffer === "undefined") {
        return;
    }
    const create = GPUDevice.prototype.createBuffer;
    GPUDevice.prototype.createBuffer = function (descriptor) {
        const buffer = create.call(this, descriptor);
        buffers.set(buffer, descriptor.size);
        state.gpuBytes += descriptor.size;
        state.gpuLive += 1;
        state.gpuBytesPeak = Math.max(state.gpuBytesPeak, state.gpuBytes);
        state.gpuLivePeak = Math.max(state.gpuLivePeak, state.gpuLive);
        return buffer;
    };
    const destroy = GPUBuffer.prototype.destroy;
    GPUBuffer.prototype.destroy = function () {
        const size = buffers.get(this);
        if (size !== undefined) {
            buffers.delete(this);
            state.gpuBytes -= size;
            state.gpuLive -= 1;
        }
        return destroy.call(this);
    };
}

// must run before ONNX Runtime creates its WebAssembly memory and its WebGPU device
export function installDeveloperHooks() {
    if (state.installed) {
        return;
    }
    state.installed = true;
    hookWebAssembly();
    hookGpuBuffers();
    setInterval(sample, SAMPLE_INTERVAL_MS);
}

export function noteModel(model, provider) {
    state.model = `${model.repo}@${model.revision} ${model.encoder_dtype}/${model.decoder_dtype}`;
    state.provider = provider;
}

function fields(source) {
    const entries = [];
    for (const name in source) {
        const value = source[name];
        if (typeof value !== "function") {
            entries.push({ name, value: String(value) });
        }
    }
    return entries;
}

async function readAdapter() {
    if (state.adapter !== null) {
        return state.adapter;
    }
    if (!self.navigator?.gpu) {
        return null;
    }
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) {
        return null;
    }
    state.adapter = {
        info: fields(adapter.info),
        features: [...adapter.features],
        limits: fields(adapter.limits),
    };
    return state.adapter;
}

function peak(current, highest) {
    return { current, peak: Math.max(current, highest) };
}

export function resetPeaks() {
    state.wasmPeak = wasmHeapBytes();
    state.jsHeapPeak = jsHeap()?.used ?? 0;
    state.gpuBytesPeak = state.gpuBytes;
    state.gpuLivePeak = state.gpuLive;
}

export async function snapshot(resetPeakValues) {
    if (resetPeakValues) {
        resetPeaks();
    }
    sample();
    const heap = jsHeap();
    return {
        hooks: state.installed,
        wasm_heap: memories.size > 0 ? peak(wasmHeapBytes(), state.wasmPeak) : null,
        gpu_buffers: {
            bytes: peak(state.gpuBytes, state.gpuBytesPeak),
            live: peak(state.gpuLive, state.gpuLivePeak),
        },
        js_heap: heap ? { used: peak(heap.used, state.jsHeapPeak), limit: heap.limit } : null,
        model: state.model,
        provider: state.provider,
        adapter: await readAdapter(),
    };
}

function isPlainObject(value) {
    return value !== null && typeof value === "object" && !Array.isArray(value);
}

function providerName(entry) {
    return typeof entry === "string" ? entry : entry?.name;
}

function asProvider(entry) {
    return typeof entry === "string" ? { name: entry } : entry;
}

function mergeProviders(base, extra) {
    const merged = base.map(asProvider);
    for (const entry of extra) {
        const index = merged.findIndex((candidate) => providerName(candidate) === providerName(entry));
        if (index < 0) {
            merged.push(asProvider(entry));
        } else {
            merged[index] = { ...merged[index], ...asProvider(entry) };
        }
    }
    return merged;
}

// objects merge key by key; execution providers merge by name; anything else is replaced
export function mergeSessionOptions(base, extra) {
    const merged = { ...base };
    for (const [key, value] of Object.entries(extra ?? {})) {
        if (key === "executionProviders" && Array.isArray(value) && Array.isArray(merged[key])) {
            merged[key] = mergeProviders(merged[key], value);
        } else if (isPlainObject(value) && isPlainObject(merged[key])) {
            merged[key] = mergeSessionOptions(merged[key], value);
        } else {
            merged[key] = value;
        }
    }
    return merged;
}
