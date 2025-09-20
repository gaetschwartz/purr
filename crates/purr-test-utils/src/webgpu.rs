//! WebGPU testing utilities and helpers

#[cfg(feature = "web")]
use wasm_bindgen::prelude::*;
#[cfg(feature = "web")]
use web_sys::*;
use std::collections::HashMap;

/// WebGPU test environment setup
#[cfg(feature = "web")]
pub struct WebGpuTestEnv {
    pub adapter: Option<GpuAdapter>,
    pub device: Option<GpuDevice>,
    pub queue: Option<GpuQueue>,
}

#[cfg(feature = "web")]
impl WebGpuTestEnv {
    /// Create a new WebGPU test environment
    pub async fn new() -> Result<Self, JsValue> {
        let window = web_sys::window().ok_or("No window available")?;
        let navigator = window.navigator();
        let gpu = navigator.gpu().ok_or("WebGPU not supported")?;

        // Request adapter
        let adapter = wasm_bindgen_futures::JsFuture::from(
            gpu.request_adapter()
        ).await?;

        let adapter: GpuAdapter = adapter.into();

        // Request device
        let device_descriptor = GpuDeviceDescriptor::new();
        let device = wasm_bindgen_futures::JsFuture::from(
            adapter.request_device_with_descriptor(&device_descriptor)
        ).await?;

        let device: GpuDevice = device.into();
        let queue = device.queue();

        Ok(Self {
            adapter: Some(adapter),
            device: Some(device),
            queue: Some(queue),
        })
    }

    /// Create a compute shader for testing
    pub fn create_test_compute_shader(&self, source: &str) -> Result<GpuComputePipeline, JsValue> {
        let device = self.device.as_ref().ok_or("Device not available")?;

        let shader_module_descriptor = GpuShaderModuleDescriptor::new(source);
        let shader_module = device.create_shader_module(&shader_module_descriptor);

        let compute_stage = GpuProgrammableStage::new("main", &shader_module);
        let pipeline_descriptor = GpuComputePipelineDescriptor::new(&compute_stage);

        Ok(device.create_compute_pipeline(&pipeline_descriptor))
    }

    /// Create a buffer for testing
    pub fn create_test_buffer(&self, size: u64, usage: u32) -> Result<GpuBuffer, JsValue> {
        let device = self.device.as_ref().ok_or("Device not available")?;

        let buffer_descriptor = GpuBufferDescriptor::new(size, usage);
        Ok(device.create_buffer(&buffer_descriptor))
    }

    /// Run a simple compute shader test
    pub async fn run_compute_test(
        &self,
        shader_source: &str,
        input_data: &[f32],
        workgroup_size: (u32, u32, u32),
    ) -> Result<Vec<f32>, JsValue> {
        let device = self.device.as_ref().ok_or("Device not available")?;
        let queue = self.queue.as_ref().ok_or("Queue not available")?;

        // Create buffers
        let buffer_size = (input_data.len() * 4) as u64; // f32 = 4 bytes
        let storage_buffer = self.create_test_buffer(
            buffer_size,
            GpuBufferUsage::STORAGE | GpuBufferUsage::COPY_SRC | GpuBufferUsage::COPY_DST,
        )?;

        let output_buffer = self.create_test_buffer(
            buffer_size,
            GpuBufferUsage::MAP_READ | GpuBufferUsage::COPY_DST,
        )?;

        // Write input data
        let input_bytes = bytemuck::cast_slice(input_data);
        queue.write_buffer_with_u8_array(&storage_buffer, 0, input_bytes);

        // Create compute pipeline
        let pipeline = self.create_test_compute_shader(shader_source)?;

        // Create bind group
        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group_descriptor = GpuBindGroupDescriptor::new(&bind_group_layout);

        // Add buffer binding
        let buffer_binding = GpuBufferBinding::new(&storage_buffer);
        buffer_binding.set_offset(0);
        buffer_binding.set_size(buffer_size);

        let binding_resource = GpuBindingResource::from(buffer_binding);
        let bind_group_entry = GpuBindGroupEntry::new(0, &binding_resource);

        let entries = js_sys::Array::new();
        entries.push(&bind_group_entry);
        bind_group_descriptor.set_entries(&entries);

        let bind_group = device.create_bind_group(&bind_group_descriptor);

        // Create command encoder
        let command_encoder = device.create_command_encoder();
        let compute_pass = command_encoder.begin_compute_pass();

        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, &bind_group);
        compute_pass.dispatch_workgroups(workgroup_size.0, workgroup_size.1, workgroup_size.2);
        compute_pass.end();

        // Copy result to output buffer
        command_encoder.copy_buffer_to_buffer(&storage_buffer, 0, &output_buffer, 0, buffer_size);

        // Submit commands
        let commands = js_sys::Array::new();
        commands.push(&command_encoder.finish());
        queue.submit(&commands);

        // Read result
        let _map_result = wasm_bindgen_futures::JsFuture::from(
            output_buffer.map_async(GpuMapMode::READ())
        ).await?;

        let array_buffer = output_buffer.get_mapped_range(0, buffer_size);
        let uint8_array = js_sys::Uint8Array::new(&array_buffer);
        let mut result_bytes = vec![0u8; uint8_array.length() as usize];
        uint8_array.copy_to(&mut result_bytes);

        output_buffer.unmap();

        // Convert bytes back to f32
        let result_f32 = bytemuck::cast_slice::<u8, f32>(&result_bytes).to_vec();
        Ok(result_f32)
    }
}

/// Mock WebGPU implementation for testing without actual WebGPU
pub struct MockWebGpu {
    operations: HashMap<String, Box<dyn Fn(&[f32]) -> Vec<f32> + Send + Sync>>,
}

impl MockWebGpu {
    pub fn new() -> Self {
        let mut operations = HashMap::new();

        // Add some default operations
        operations.insert(
            "multiply".to_string(),
            Box::new(|data: &[f32]| data.iter().map(|&x| x * 2.0).collect())
        );

        operations.insert(
            "add".to_string(),
            Box::new(|data: &[f32]| data.iter().map(|&x| x + 1.0).collect())
        );

        operations.insert(
            "square".to_string(),
            Box::new(|data: &[f32]| data.iter().map(|&x| x * x).collect())
        );

        Self { operations }
    }

    pub fn add_operation<F>(&mut self, name: String, operation: F)
    where
        F: Fn(&[f32]) -> Vec<f32> + Send + Sync + 'static,
    {
        self.operations.insert(name, Box::new(operation));
    }

    pub fn execute_operation(&self, name: &str, data: &[f32]) -> Result<Vec<f32>, String> {
        self.operations
            .get(name)
            .map(|op| op(data))
            .ok_or_else(|| format!("Operation '{}' not found", name))
    }

    pub fn list_operations(&self) -> Vec<String> {
        self.operations.keys().cloned().collect()
    }
}

impl Default for MockWebGpu {
    fn default() -> Self {
        Self::new()
    }
}

/// Test utilities for audio processing with WebGPU
pub struct AudioWebGpuTester;

impl AudioWebGpuTester {
    /// Generate a test compute shader for audio processing
    pub fn fft_shader() -> &'static str {
        r#"
        @group(0) @binding(0)
        var<storage, read_write> audio_data: array<f32>;

        @group(0) @binding(1)
        var<storage, read_write> fft_output: array<f32>;

        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let index = global_id.x;
            if (index >= arrayLength(&audio_data)) {
                return;
            }

            // Simplified FFT operation for testing
            let real_part = audio_data[index];
            let imag_part = 0.0;

            fft_output[index * 2] = real_part;
            fft_output[index * 2 + 1] = imag_part;
        }
        "#
    }

    /// Generate a test compute shader for audio filtering
    pub fn lowpass_filter_shader() -> &'static str {
        r#"
        @group(0) @binding(0)
        var<storage, read_write> audio_data: array<f32>;

        @group(0) @binding(1)
        var<uniform> filter_params: FilterParams;

        struct FilterParams {
            cutoff_frequency: f32,
            sample_rate: f32,
        }

        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let index = global_id.x;
            if (index >= arrayLength(&audio_data) || index == 0) {
                return;
            }

            // Simple lowpass filter implementation
            let alpha = filter_params.cutoff_frequency / filter_params.sample_rate;
            audio_data[index] = alpha * audio_data[index] + (1.0 - alpha) * audio_data[index - 1];
        }
        "#
    }

    /// Generate test audio data for WebGPU processing
    pub fn generate_test_audio(samples: usize, frequency: f32, sample_rate: f32) -> Vec<f32> {
        (0..samples)
            .map(|i| {
                let t = i as f32 / sample_rate;
                (2.0 * std::f32::consts::PI * frequency * t).sin()
            })
            .collect()
    }
}

/// Performance testing utilities for WebGPU operations
pub struct WebGpuPerformanceTester {
    pub operation_times: HashMap<String, std::time::Duration>,
}

impl WebGpuPerformanceTester {
    pub fn new() -> Self {
        Self {
            operation_times: HashMap::new(),
        }
    }

    pub async fn benchmark_operation<F, Fut>(&mut self, name: &str, operation: F) -> Result<std::time::Duration, String>
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = Result<(), String>>,
    {
        let start = std::time::Instant::now();
        operation().await?;
        let duration = start.elapsed();

        self.operation_times.insert(name.to_string(), duration);
        Ok(duration)
    }

    pub fn get_benchmark_results(&self) -> &HashMap<String, std::time::Duration> {
        &self.operation_times
    }

    pub fn average_time(&self) -> Option<std::time::Duration> {
        if self.operation_times.is_empty() {
            return None;
        }

        let total: std::time::Duration = self.operation_times.values().sum();
        Some(total / self.operation_times.len() as u32)
    }
}

impl Default for WebGpuPerformanceTester {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(all(test, feature = "web"))]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    wasm_bindgen_test_configure!(run_in_browser);

    #[wasm_bindgen_test]
    async fn test_webgpu_environment_creation() {
        let env_result = WebGpuTestEnv::new().await;
        // This test might fail in environments without WebGPU support
        if env_result.is_ok() {
            let env = env_result.unwrap();
            assert!(env.device.is_some());
            assert!(env.queue.is_some());
        }
    }

    #[test]
    fn test_mock_webgpu() {
        let mock = MockWebGpu::new();
        let test_data = vec![1.0, 2.0, 3.0, 4.0];

        let result = mock.execute_operation("multiply", &test_data);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), vec![2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_audio_webgpu_tester() {
        let audio_data = AudioWebGpuTester::generate_test_audio(1000, 440.0, 44100.0);
        assert_eq!(audio_data.len(), 1000);
        assert!(audio_data.iter().all(|&sample| sample >= -1.0 && sample <= 1.0));
    }
}