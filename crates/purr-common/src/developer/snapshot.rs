use super::webgpu::FeatureRow;
use crate::{
    accel::{AccelBackend, LastRun},
    model::engine::EngineId,
    platform::DeviceType,
};

/// A figure with the highest value seen since the peak was last reset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Peaked {
    pub current: u64,
    pub peak: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionProvider {
    WebGpu,
    Cpu,
}

/// A named value the browser reports about the adapter, shown as text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Field {
    pub name: String,
    pub value: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdapterReading {
    pub info: Vec<Field>,
    pub features: Vec<FeatureRow>,
    pub limits: Vec<Field>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JsHeapReading {
    pub used: Peaked,
    pub limit: u64,
}

/// GPU memory as ONNX Runtime allocates it: buffers created and not yet destroyed, since the
/// browser exposes no real figure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuBuffers {
    pub bytes: Peaked,
    pub live: Peaked,
}

/// What the transcription worker reports.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkerReading {
    /// `None` while no WebAssembly memory was captured, for instance before a model loads.
    pub wasm_heap: Option<Peaked>,
    pub gpu_buffers: GpuBuffers,
    pub js_heap: Option<JsHeapReading>,
    pub provider: Option<ExecutionProvider>,
    pub model: Option<String>,
    pub adapter: Option<AdapterReading>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BrowserSnapshot {
    pub worker: WorkerReading,
    /// The converter's WebAssembly heap while it runs or after its last run.
    pub converter_heap: Option<Peaked>,
    pub page_js_heap: Option<JsHeapReading>,
}

/// A GPU device an engine reports, with its names exactly as reported.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeDevice {
    pub engine: EngineId,
    pub name: String,
    pub description: String,
    pub device_id: Option<String>,
    pub backend: Option<AccelBackend>,
    pub device_type: DeviceType,
    /// `None` where the backend reports no figure.
    pub memory_free: Option<u64>,
    pub memory_total: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeSnapshot {
    pub devices: Vec<NativeDevice>,
    pub last_run: Option<LastRun>,
    /// Working set of this process; the peak is the system's own and cannot be reset.
    pub process_memory: Option<Peaked>,
    pub process_virtual: Option<Peaked>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeveloperSnapshot {
    Browser(BrowserSnapshot),
    Native(NativeSnapshot),
}
