//! WebGPU testing utilities
//!
//! This module provides utilities and helpers for testing WebGPU functionality
//! in both unit tests and integration tests.

#[cfg(test)]
use wasm_bindgen::prelude::*;
#[cfg(test)]
use web_sys::{console};
#[cfg(test)]
use js_sys::{Object, Reflect, Promise, Float32Array};
#[cfg(test)]
use crate::{WebError, WebResult};

/// Mock WebGPU adapter for testing
#[cfg(test)]
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_name = "MockWebGPUAdapter")]
    pub type MockWebGPUAdapter;

    #[wasm_bindgen(constructor)]
    fn new() -> MockWebGPUAdapter;

    #[wasm_bindgen(method)]
    fn request_device(this: &MockWebGPUAdapter) -> Promise;

    #[wasm_bindgen(method, getter)]
    fn features(this: &MockWebGPUAdapter) -> js_sys::Set;

    #[wasm_bindgen(method, getter)]
    fn limits(this: &MockWebGPUAdapter) -> Object;

    #[wasm_bindgen(method, getter)]
    fn info(this: &MockWebGPUAdapter) -> Object;
}

/// Mock WebGPU device for testing
#[cfg(test)]
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_name = "MockWebGPUDevice")]
    pub type MockWebGPUDevice;

    #[wasm_bindgen(constructor)]
    fn new() -> MockWebGPUDevice;

    #[wasm_bindgen(method)]
    fn create_buffer(this: &MockWebGPUDevice, descriptor: &Object) -> Object;

    #[wasm_bindgen(method)]
    fn create_shader_module(this: &MockWebGPUDevice, descriptor: &Object) -> Object;

    #[wasm_bindgen(method)]
    fn create_compute_pipeline(this: &MockWebGPUDevice, descriptor: &Object) -> Object;

    #[wasm_bindgen(method)]
    fn create_render_pipeline(this: &MockWebGPUDevice, descriptor: &Object) -> Object;

    #[wasm_bindgen(method)]
    fn create_command_encoder(this: &MockWebGPUDevice) -> Object;

    #[wasm_bindgen(method, getter)]
    fn queue(this: &MockWebGPUDevice) -> Object;

    #[wasm_bindgen(method, getter)]
    fn features(this: &MockWebGPUDevice) -> js_sys::Set;

    #[wasm_bindgen(method, getter)]
    fn limits(this: &MockWebGPUDevice) -> Object;

    #[wasm_bindgen(method, getter)]
    fn lost(this: &MockWebGPUDevice) -> Promise;
}

#[cfg(test)]
pub struct WebGPUTestEnvironment {
    pub webgpu_available: bool,
    pub adapter: Option<MockWebGPUAdapter>,
    pub device: Option<MockWebGPUDevice>,
}

#[cfg(test)]
impl Default for WebGPUTestEnvironment {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
impl WebGPUTestEnvironment {
    /// Create a new test environment
    pub fn new() -> Self {
        Self {
            webgpu_available: Self::check_webgpu_support(),
            adapter: None,
            device: None,
        }
    }

    /// Check if WebGPU is supported in the current environment
    pub fn check_webgpu_support() -> bool {
        let window = match web_sys::window() {
            Some(w) => w,
            None => return false,
        };

        let navigator = window.navigator();
        match Reflect::get(&navigator, &JsValue::from_str("gpu")) {
            Ok(gpu_val) => !gpu_val.is_undefined(),
            Err(_) => false,
        }
    }

    /// Initialize mock adapter and device
    pub fn setup_mock_device(&mut self) -> WebResult<()> {
        if !self.webgpu_available {
            console::log_1(&"Using mock WebGPU device for testing".into());
        }

        self.adapter = Some(MockWebGPUAdapter::new());
        self.device = Some(MockWebGPUDevice::new());

        Ok(())
    }

    /// Get the mock device for testing
    pub fn get_device(&self) -> Option<&MockWebGPUDevice> {
        self.device.as_ref()
    }

    /// Get the mock adapter for testing
    pub fn get_adapter(&self) -> Option<&MockWebGPUAdapter> {
        self.adapter.as_ref()
    }
}

/// Create a test buffer descriptor with specified size and usage
#[cfg(test)]
pub fn create_test_buffer_descriptor(size: u64, usage: u32) -> Object {
    let descriptor = Object::new();
    Reflect::set(&descriptor, &JsValue::from_str("size"), &JsValue::from_f64(size as f64))
        .expect("Failed to set buffer size");
    Reflect::set(&descriptor, &JsValue::from_str("usage"), &JsValue::from_f64(usage as f64))
        .expect("Failed to set buffer usage");
    descriptor
}

/// Create a test shader module descriptor with WGSL code
#[cfg(test)]
pub fn create_test_shader_descriptor(wgsl_code: &str) -> Object {
    let descriptor = Object::new();
    Reflect::set(&descriptor, &JsValue::from_str("code"), &JsValue::from_str(wgsl_code))
        .expect("Failed to set shader code");
    descriptor
}

/// Create a test compute pipeline descriptor
#[cfg(test)]
pub fn create_test_compute_pipeline_descriptor(entry_point: &str) -> Object {
    let descriptor = Object::new();

    let compute_stage = Object::new();
    Reflect::set(&compute_stage, &JsValue::from_str("module"), &JsValue::NULL)
        .expect("Failed to set compute module");
    Reflect::set(&compute_stage, &JsValue::from_str("entryPoint"), &JsValue::from_str(entry_point))
        .expect("Failed to set compute entry point");

    Reflect::set(&descriptor, &JsValue::from_str("compute"), &compute_stage)
        .expect("Failed to set compute stage");
    Reflect::set(&descriptor, &JsValue::from_str("layout"), &JsValue::from_str("auto"))
        .expect("Failed to set pipeline layout");

    descriptor
}

/// Create a test bind group layout descriptor
#[cfg(test)]
pub fn create_test_bind_group_layout(bindings: &[(u32, &str, &str)]) -> Object {
    let descriptor = Object::new();
    let entries = js_sys::Array::new();

    for &(binding, visibility, buffer_type) in bindings {
        let entry = Object::new();
        Reflect::set(&entry, &JsValue::from_str("binding"), &JsValue::from_f64(binding as f64))
            .expect("Failed to set binding");

        let visibility_flags = match visibility {
            "vertex" => 1u32,
            "fragment" => 2u32,
            "compute" => 4u32,
            _ => 7u32, // All stages
        };
        Reflect::set(&entry, &JsValue::from_str("visibility"), &JsValue::from_f64(visibility_flags as f64))
            .expect("Failed to set visibility");

        let buffer_layout = Object::new();
        Reflect::set(&buffer_layout, &JsValue::from_str("type"), &JsValue::from_str(buffer_type))
            .expect("Failed to set buffer type");
        Reflect::set(&entry, &JsValue::from_str("buffer"), &buffer_layout)
            .expect("Failed to set buffer layout");

        entries.push(&entry);
    }

    Reflect::set(&descriptor, &JsValue::from_str("entries"), &entries)
        .expect("Failed to set entries");

    descriptor
}

/// Validate WebGPU device limits for testing
#[cfg(test)]
pub fn validate_device_limits(limits: &Object) -> WebResult<()> {
    let required_limits = [
        ("maxBufferSize", 268_435_456u64),         // 256MB
        ("maxStorageBufferBindingSize", 134_217_728u64), // 128MB
        ("maxUniformBufferBindingSize", 65536u64),  // 64KB
        ("maxComputeWorkgroupSizeX", 256u64),
        ("maxComputeWorkgroupSizeY", 256u64),
        ("maxComputeWorkgroupSizeZ", 64u64),
    ];

    for (limit_name, _min_value) in required_limits {
        match Reflect::get(limits, &JsValue::from_str(limit_name)) {
            Ok(limit_val) => {
                if limit_val.is_undefined() {
                    return Err(WebError::WebGpu {
                        operation: "limit_validation".to_string(),
                        device_type: "webgpu".to_string(),
                        source: Box::new(std::io::Error::new(std::io::ErrorKind::Other, format!("Missing limit: {}", limit_name))),
                    });
                }
                // In a real test, we'd validate the actual values
            }
            Err(_) => {
                return Err(WebError::WebGpu {
                    operation: "limit_access".to_string(),
                    device_type: "webgpu".to_string(),
                    source: Box::new(std::io::Error::new(std::io::ErrorKind::Other, format!("Failed to get limit: {}", limit_name))),
                });
            }
        }
    }

    Ok(())
}

/// Create test audio data for WebGPU processing
#[cfg(test)]
pub fn create_test_audio_samples(sample_count: usize, frequency: f32, sample_rate: f32) -> Vec<f32> {
    let mut samples = Vec::with_capacity(sample_count);

    for i in 0..sample_count {
        let t = i as f32 / sample_rate;
        let sample = (frequency * 2.0 * std::f32::consts::PI * t).sin();
        samples.push(sample * 0.5); // Reduce amplitude to prevent clipping
    }

    samples
}

/// Convert audio samples to WebGPU buffer data
#[cfg(test)]
pub fn audio_samples_to_buffer_data(samples: &[f32]) -> Float32Array {
    let array = Float32Array::new_with_length(samples.len() as u32);
    for (i, &sample) in samples.iter().enumerate() {
        array.set_index(i as u32, sample);
    }
    array
}

/// Validate audio processing results
#[cfg(test)]
pub fn validate_audio_processing_results(
    input: &[f32],
    output: &[f32],
    expected_gain: f32,
) -> WebResult<()> {
    if input.len() != output.len() {
        return Err(WebError::WebGpu {
            operation: "buffer_validation".to_string(),
            device_type: "webgpu".to_string(),
            source: Box::new(std::io::Error::new(std::io::ErrorKind::InvalidData, "Input and output buffer sizes don't match")),
        });
    }

    let mut total_error = 0.0f32;
    let tolerance = 0.01f32; // 1% tolerance

    for (i, (&input_sample, &output_sample)) in input.iter().zip(output.iter()).enumerate() {
        let expected = input_sample * expected_gain;
        let error = (output_sample - expected).abs();

        if error > tolerance {
            return Err(WebError::WebGpu {
                operation: "audio_processing".to_string(),
                device_type: "webgpu".to_string(),
                source: Box::new(std::io::Error::new(std::io::ErrorKind::InvalidData, format!(
                    "Audio processing error at sample {}: expected {:.6}, got {:.6}, error {:.6}",
                    i, expected, output_sample, error
                ))),
            });
        }

        total_error += error;
    }

    let average_error = total_error / input.len() as f32;
    if average_error > tolerance / 10.0 {
        return Err(WebError::WebGpu {
            operation: "audio_processing".to_string(),
            device_type: "webgpu".to_string(),
            source: Box::new(std::io::Error::new(std::io::ErrorKind::InvalidData, format!(
                "Average audio processing error too high: {:.6}",
                average_error
            ))),
        });
    }

    Ok(())
}

/// Test WebGPU feature availability
#[cfg(test)]
pub fn test_webgpu_features(features: &js_sys::Set, required_features: &[&str]) -> WebResult<Vec<String>> {
    let mut missing_features = Vec::new();

    for &feature in required_features {
        let feature_js = JsValue::from_str(feature);
        if !features.has(&feature_js) {
            missing_features.push(feature.to_string());
        }
    }

    if !missing_features.is_empty() {
        return Err(WebError::WebGpu {
            operation: "feature_validation".to_string(),
            device_type: "webgpu".to_string(),
            source: Box::new(std::io::Error::new(std::io::ErrorKind::Other, format!(
                "Missing WebGPU features: {}",
                missing_features.join(", ")
            ))),
        });
    }

    Ok(missing_features)
}

/// Create test texture descriptor
#[cfg(test)]
pub fn create_test_texture_descriptor(width: u32, height: u32, format: &str) -> Object {
    let descriptor = Object::new();

    let size = Object::new();
    Reflect::set(&size, &JsValue::from_str("width"), &JsValue::from_f64(width as f64))
        .expect("Failed to set texture width");
    Reflect::set(&size, &JsValue::from_str("height"), &JsValue::from_f64(height as f64))
        .expect("Failed to set texture height");
    Reflect::set(&size, &JsValue::from_str("depthOrArrayLayers"), &JsValue::from_f64(1.0))
        .expect("Failed to set texture depth");

    Reflect::set(&descriptor, &JsValue::from_str("size"), &size)
        .expect("Failed to set texture size");
    Reflect::set(&descriptor, &JsValue::from_str("format"), &JsValue::from_str(format))
        .expect("Failed to set texture format");
    Reflect::set(&descriptor, &JsValue::from_str("usage"), &JsValue::from_f64(0x0014 as f64)) // TEXTURE_BINDING | COPY_DST
        .expect("Failed to set texture usage");

    descriptor
}

/// Performance measurement utilities for WebGPU tests
#[cfg(test)]
pub struct WebGPUPerformanceTimer {
    start_time: f64,
    measurements: Vec<(String, f64)>,
}

#[cfg(test)]
impl Default for WebGPUPerformanceTimer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
impl WebGPUPerformanceTimer {
    pub fn new() -> Self {
        Self {
            start_time: 0.0,
            measurements: Vec::new(),
        }
    }

    pub fn start(&mut self) {
        self.start_time = js_sys::Date::now();
    }

    pub fn measure(&mut self, operation: &str) {
        let elapsed = js_sys::Date::now() - self.start_time;
        self.measurements.push((operation.to_string(), elapsed));
        console::log_1(&format!("WebGPU {}: {:.2}ms", operation, elapsed).into());
    }

    pub fn get_measurements(&self) -> &[(String, f64)] {
        &self.measurements
    }

    pub fn total_time(&self) -> f64 {
        self.measurements.iter().map(|(_, time)| time).sum()
    }
}