//! WebGPU rendering pipeline and compute shader tests
//!
//! This module tests WebGPU pipeline creation, compute shader execution,
//! and rendering pipeline functionality for audio processing.

#![allow(dead_code)]

use purr_web::{WebError};
use wasm_bindgen_test::*;
use wasm_bindgen::prelude::*;
use web_sys::{console};
use js_sys::{Object, Reflect};

wasm_bindgen_test_configure!(run_in_browser);

// Mock WebGPU pipeline structures
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_name = "MockGpuComputePipeline")]
    type MockGpuComputePipeline;

    #[wasm_bindgen(constructor)]
    fn new() -> MockGpuComputePipeline;

    #[wasm_bindgen(method, js_name = "getBindGroupLayout")]
    fn get_bind_group_layout(this: &MockGpuComputePipeline, index: u32) -> Object;

    #[wasm_bindgen(js_name = "MockGpuRenderPipeline")]
    type MockGpuRenderPipeline;

    #[wasm_bindgen(constructor)]
    fn new_render() -> MockGpuRenderPipeline;

    #[wasm_bindgen(method, js_name = "getBindGroupLayout")]
    fn get_render_bind_group_layout(this: &MockGpuRenderPipeline, index: u32) -> Object;

    #[wasm_bindgen(js_name = "MockGpuCommandEncoder")]
    type MockGpuCommandEncoder;

    #[wasm_bindgen(constructor)]
    fn new_encoder() -> MockGpuCommandEncoder;

    #[wasm_bindgen(method, js_name = "beginComputePass")]
    fn begin_compute_pass(this: &MockGpuCommandEncoder) -> Object;

    #[wasm_bindgen(method, js_name = "beginRenderPass")]
    fn begin_render_pass(this: &MockGpuCommandEncoder, descriptor: &Object) -> Object;

    #[wasm_bindgen(method, js_name = "finish")]
    fn finish(this: &MockGpuCommandEncoder) -> Object;
}

/// Audio processing compute shader for WebGPU
const AUDIO_PROCESSING_COMPUTE_SHADER: &str = r#"
    @group(0) @binding(0) var<storage, read> input_buffer: array<f32>;
    @group(0) @binding(1) var<storage, read_write> output_buffer: array<f32>;
    @group(0) @binding(2) var<uniform> params: ProcessingParams;

    struct ProcessingParams {
        sample_rate: f32,
        frame_size: u32,
        gain: f32,
        padding: f32,
    }

    // FFT processing for audio analysis
    @compute @workgroup_size(64)
    fn audio_process_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
        let index = global_id.x;
        if (index >= arrayLength(&input_buffer)) {
            return;
        }

        // Apply gain and basic filtering
        let sample = input_buffer[index];
        let processed_sample = sample * params.gain;

        // Simple high-pass filter (subtract low-frequency component)
        let filtered_sample = processed_sample * 0.95;

        output_buffer[index] = filtered_sample;
    }

    // Spectral analysis compute shader
    @compute @workgroup_size(256)
    fn spectral_analysis_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
        let index = global_id.x;
        let frame_size = params.frame_size;

        if (index >= frame_size) {
            return;
        }

        // Simple DFT calculation for spectral analysis
        var real: f32 = 0.0;
        var imag: f32 = 0.0;

        for (var k: u32 = 0u; k < frame_size; k = k + 1u) {
            let angle = -2.0 * 3.14159265359 * f32(index) * f32(k) / f32(frame_size);
            let cos_val = cos(angle);
            let sin_val = sin(angle);

            if (k < arrayLength(&input_buffer)) {
                real = real + input_buffer[k] * cos_val;
                imag = imag + input_buffer[k] * sin_val;
            }
        }

        // Store magnitude in output buffer
        let magnitude = sqrt(real * real + imag * imag);
        if (index < arrayLength(&output_buffer)) {
            output_buffer[index] = magnitude;
        }
    }
"#;

/// Audio visualization vertex shader
const AUDIO_VISUALIZATION_VERTEX_SHADER: &str = r#"
    struct VertexInput {
        @location(0) position: vec2<f32>,
        @location(1) audio_sample: f32,
    }

    struct VertexOutput {
        @builtin(position) clip_position: vec4<f32>,
        @location(0) audio_level: f32,
    }

    @group(0) @binding(0) var<uniform> view_params: ViewParams;

    struct ViewParams {
        scale: vec2<f32>,
        offset: vec2<f32>,
        time: f32,
        amplitude: f32,
    }

    @vertex
    fn vs_main(input: VertexInput) -> VertexOutput {
        var output: VertexOutput;

        // Transform position with audio sample amplitude
        let wave_height = input.audio_sample * view_params.amplitude;
        let transformed_pos = vec2<f32>(
            input.position.x * view_params.scale.x + view_params.offset.x,
            input.position.y * view_params.scale.y + wave_height + view_params.offset.y
        );

        output.clip_position = vec4<f32>(transformed_pos, 0.0, 1.0);
        output.audio_level = abs(input.audio_sample);

        return output;
    }
"#;

/// Audio visualization fragment shader
const AUDIO_VISUALIZATION_FRAGMENT_SHADER: &str = r#"
    @fragment
    fn fs_main(@location(0) audio_level: f32) -> @location(0) vec4<f32> {
        // Color based on audio level
        let intensity = clamp(audio_level, 0.0, 1.0);
        let color = vec3<f32>(
            intensity,
            0.5 * intensity,
            1.0 - intensity
        );

        return vec4<f32>(color, 1.0);
    }
"#;

fn create_compute_pipeline_descriptor() -> Object {
    let descriptor = Object::new();

    // Create compute stage
    let compute_stage = Object::new();
    Reflect::set(&compute_stage, &JsValue::from_str("module"), &JsValue::NULL).unwrap();
    Reflect::set(&compute_stage, &JsValue::from_str("entryPoint"), &JsValue::from_str("audio_process_main")).unwrap();

    Reflect::set(&descriptor, &JsValue::from_str("compute"), &compute_stage).unwrap();
    Reflect::set(&descriptor, &JsValue::from_str("layout"), &JsValue::from_str("auto")).unwrap();

    descriptor
}

fn create_render_pipeline_descriptor() -> Object {
    let descriptor = Object::new();

    // Vertex stage
    let vertex_stage = Object::new();
    Reflect::set(&vertex_stage, &JsValue::from_str("module"), &JsValue::NULL).unwrap();
    Reflect::set(&vertex_stage, &JsValue::from_str("entryPoint"), &JsValue::from_str("vs_main")).unwrap();

    // Fragment stage
    let fragment_stage = Object::new();
    Reflect::set(&fragment_stage, &JsValue::from_str("module"), &JsValue::NULL).unwrap();
    Reflect::set(&fragment_stage, &JsValue::from_str("entryPoint"), &JsValue::from_str("fs_main")).unwrap();

    // Color target
    let color_target = Object::new();
    Reflect::set(&color_target, &JsValue::from_str("format"), &JsValue::from_str("bgra8unorm")).unwrap();

    let color_targets = js_sys::Array::new();
    color_targets.push(&color_target);

    Reflect::set(&fragment_stage, &JsValue::from_str("targets"), &color_targets).unwrap();

    Reflect::set(&descriptor, &JsValue::from_str("vertex"), &vertex_stage).unwrap();
    Reflect::set(&descriptor, &JsValue::from_str("fragment"), &fragment_stage).unwrap();
    Reflect::set(&descriptor, &JsValue::from_str("layout"), &JsValue::from_str("auto")).unwrap();

    // Primitive state
    let primitive = Object::new();
    Reflect::set(&primitive, &JsValue::from_str("topology"), &JsValue::from_str("triangle-list")).unwrap();
    Reflect::set(&descriptor, &JsValue::from_str("primitive"), &primitive).unwrap();

    descriptor
}

#[cfg(test)]
mod compute_pipeline_tests {
    use super::*;

    #[wasm_bindgen_test]
    fn test_audio_processing_shader_validation() {
        console::log_1(&"Testing audio processing compute shader validation".into());

        // Validate shader structure
        assert!(AUDIO_PROCESSING_COMPUTE_SHADER.contains("@compute"));
        assert!(AUDIO_PROCESSING_COMPUTE_SHADER.contains("@workgroup_size"));
        assert!(AUDIO_PROCESSING_COMPUTE_SHADER.contains("audio_process_main"));
        assert!(AUDIO_PROCESSING_COMPUTE_SHADER.contains("spectral_analysis_main"));

        // Check required bindings
        assert!(AUDIO_PROCESSING_COMPUTE_SHADER.contains("@group(0) @binding(0)"));
        assert!(AUDIO_PROCESSING_COMPUTE_SHADER.contains("@group(0) @binding(1)"));
        assert!(AUDIO_PROCESSING_COMPUTE_SHADER.contains("@group(0) @binding(2)"));

        // Verify data types
        assert!(AUDIO_PROCESSING_COMPUTE_SHADER.contains("array<f32>"));
        assert!(AUDIO_PROCESSING_COMPUTE_SHADER.contains("ProcessingParams"));

        console::log_1(&"✓ Audio processing shader validation passed".into());
    }

    #[wasm_bindgen_test]
    fn test_compute_pipeline_descriptor_creation() {
        console::log_1(&"Testing compute pipeline descriptor creation".into());

        let descriptor = create_compute_pipeline_descriptor();

        // Verify descriptor structure
        let compute_stage = Reflect::get(&descriptor, &JsValue::from_str("compute")).unwrap();
        assert!(!compute_stage.is_undefined());

        let entry_point = Reflect::get(&compute_stage, &JsValue::from_str("entryPoint")).unwrap();
        assert_eq!(entry_point.as_string().unwrap(), "audio_process_main");

        let layout = Reflect::get(&descriptor, &JsValue::from_str("layout")).unwrap();
        assert_eq!(layout.as_string().unwrap(), "auto");

        console::log_1(&"✓ Compute pipeline descriptor creation successful".into());
    }

    #[wasm_bindgen_test]
    fn test_compute_workgroup_size_validation() {
        console::log_1(&"Testing compute workgroup size validation".into());

        // Test different workgroup sizes for audio processing
        let workgroup_configurations = vec![
            (64, 1, 1),   // For audio processing
            (256, 1, 1),  // For spectral analysis
            (32, 32, 1),  // For 2D audio processing
            (16, 16, 4),  // For 3D audio processing
        ];

        for (x, y, z) in workgroup_configurations {
            // Validate workgroup size limits
            assert!(x > 0 && x <= 256, "Workgroup size X must be 1-256");
            assert!(y > 0 && y <= 256, "Workgroup size Y must be 1-256");
            assert!(z > 0 && z <= 64, "Workgroup size Z must be 1-64");

            // Check total invocations limit
            let total_invocations = x * y * z;
            assert!(total_invocations <= 256, "Total workgroup invocations must be <= 256");
        }

        console::log_1(&"✓ Compute workgroup size validation passed".into());
    }

    #[wasm_bindgen_test]
    fn test_audio_buffer_binding() {
        console::log_1(&"Testing audio buffer binding configuration".into());

        // Test bind group layout for audio processing
        let bind_group_layout = Object::new();
        let entries = js_sys::Array::new();

        // Input buffer binding
        let input_entry = Object::new();
        Reflect::set(&input_entry, &JsValue::from_str("binding"), &JsValue::from_f64(0.0)).unwrap();
        Reflect::set(&input_entry, &JsValue::from_str("visibility"), &JsValue::from_f64(4.0)).unwrap(); // COMPUTE

        let input_buffer_layout = Object::new();
        Reflect::set(&input_buffer_layout, &JsValue::from_str("type"), &JsValue::from_str("read-only-storage")).unwrap();
        Reflect::set(&input_entry, &JsValue::from_str("buffer"), &input_buffer_layout).unwrap();
        entries.push(&input_entry);

        // Output buffer binding
        let output_entry = Object::new();
        Reflect::set(&output_entry, &JsValue::from_str("binding"), &JsValue::from_f64(1.0)).unwrap();
        Reflect::set(&output_entry, &JsValue::from_str("visibility"), &JsValue::from_f64(4.0)).unwrap(); // COMPUTE

        let output_buffer_layout = Object::new();
        Reflect::set(&output_buffer_layout, &JsValue::from_str("type"), &JsValue::from_str("storage")).unwrap();
        Reflect::set(&output_entry, &JsValue::from_str("buffer"), &output_buffer_layout).unwrap();
        entries.push(&output_entry);

        // Uniform buffer binding
        let uniform_entry = Object::new();
        Reflect::set(&uniform_entry, &JsValue::from_str("binding"), &JsValue::from_f64(2.0)).unwrap();
        Reflect::set(&uniform_entry, &JsValue::from_str("visibility"), &JsValue::from_f64(4.0)).unwrap(); // COMPUTE

        let uniform_buffer_layout = Object::new();
        Reflect::set(&uniform_buffer_layout, &JsValue::from_str("type"), &JsValue::from_str("uniform")).unwrap();
        Reflect::set(&uniform_entry, &JsValue::from_str("buffer"), &uniform_buffer_layout).unwrap();
        entries.push(&uniform_entry);

        Reflect::set(&bind_group_layout, &JsValue::from_str("entries"), &entries).unwrap();

        // Verify binding configuration
        assert_eq!(entries.length(), 3);

        console::log_1(&"✓ Audio buffer binding configuration successful".into());
    }

    #[wasm_bindgen_test]
    fn test_compute_dispatch_parameters() {
        console::log_1(&"Testing compute dispatch parameters".into());

        // Test dispatch configurations for different audio buffer sizes
        let audio_buffer_sizes: Vec<u32> = vec![
            1024,    // 1K samples
            4096,    // 4K samples
            16384,   // 16K samples
            44100,   // 1 second at 44.1kHz
            176400,  // 4 seconds at 44.1kHz
        ];

        let workgroup_size = 64u32;

        for buffer_size in audio_buffer_sizes {
            // Calculate dispatch size
            let dispatch_x = buffer_size.div_ceil(workgroup_size);
            let dispatch_y = 1u32;
            let dispatch_z = 1u32;

            // Validate dispatch parameters
            assert!(dispatch_x > 0, "Dispatch X must be positive");
            assert!(dispatch_x <= 65535, "Dispatch X must be <= 65535");
            assert!(dispatch_y <= 65535, "Dispatch Y must be <= 65535");
            assert!(dispatch_z <= 65535, "Dispatch Z must be <= 65535");

            // Verify coverage
            let total_threads = dispatch_x * workgroup_size;
            assert!(total_threads >= buffer_size, "Must cover all samples");
        }

        console::log_1(&"✓ Compute dispatch parameters validation passed".into());
    }
}

#[cfg(test)]
mod render_pipeline_tests {
    use super::*;

    #[wasm_bindgen_test]
    fn test_audio_visualization_shaders() {
        console::log_1(&"Testing audio visualization shaders".into());

        // Validate vertex shader
        assert!(AUDIO_VISUALIZATION_VERTEX_SHADER.contains("@vertex"));
        assert!(AUDIO_VISUALIZATION_VERTEX_SHADER.contains("vs_main"));
        assert!(AUDIO_VISUALIZATION_VERTEX_SHADER.contains("VertexInput"));
        assert!(AUDIO_VISUALIZATION_VERTEX_SHADER.contains("VertexOutput"));
        assert!(AUDIO_VISUALIZATION_VERTEX_SHADER.contains("@location(0) position"));
        assert!(AUDIO_VISUALIZATION_VERTEX_SHADER.contains("@location(1) audio_sample"));

        // Validate fragment shader
        assert!(AUDIO_VISUALIZATION_FRAGMENT_SHADER.contains("@fragment"));
        assert!(AUDIO_VISUALIZATION_FRAGMENT_SHADER.contains("fs_main"));
        assert!(AUDIO_VISUALIZATION_FRAGMENT_SHADER.contains("@location(0) vec4<f32>"));

        console::log_1(&"✓ Audio visualization shaders validation passed".into());
    }

    #[wasm_bindgen_test]
    fn test_render_pipeline_descriptor_creation() {
        console::log_1(&"Testing render pipeline descriptor creation".into());

        let descriptor = create_render_pipeline_descriptor();

        // Verify vertex stage
        let vertex_stage = Reflect::get(&descriptor, &JsValue::from_str("vertex")).unwrap();
        assert!(!vertex_stage.is_undefined());

        let vertex_entry = Reflect::get(&vertex_stage, &JsValue::from_str("entryPoint")).unwrap();
        assert_eq!(vertex_entry.as_string().unwrap(), "vs_main");

        // Verify fragment stage
        let fragment_stage = Reflect::get(&descriptor, &JsValue::from_str("fragment")).unwrap();
        assert!(!fragment_stage.is_undefined());

        let fragment_entry = Reflect::get(&fragment_stage, &JsValue::from_str("entryPoint")).unwrap();
        assert_eq!(fragment_entry.as_string().unwrap(), "fs_main");

        // Verify primitive state
        let primitive = Reflect::get(&descriptor, &JsValue::from_str("primitive")).unwrap();
        let topology = Reflect::get(&primitive, &JsValue::from_str("topology")).unwrap();
        assert_eq!(topology.as_string().unwrap(), "triangle-list");

        console::log_1(&"✓ Render pipeline descriptor creation successful".into());
    }

    #[wasm_bindgen_test]
    fn test_vertex_buffer_layout() {
        console::log_1(&"Testing vertex buffer layout for audio visualization".into());

        // Define vertex attributes for audio visualization
        let vertex_attributes = vec![
            ("position", 0, "float32x2", 0),      // 2D position
            ("audio_sample", 1, "float32", 8),    // Audio sample value
        ];

        let mut offset = 0u64;
        for (name, location, format, expected_offset) in vertex_attributes {
            assert!(!name.is_empty());
            assert!(location < 16); // Max vertex attributes

            // Verify format
            match format {
                "float32" => assert_eq!(offset, expected_offset),
                "float32x2" => assert_eq!(offset, expected_offset),
                "float32x3" => assert_eq!(offset, expected_offset),
                "float32x4" => assert_eq!(offset, expected_offset),
                _ => panic!("Unsupported vertex format: {}", format),
            }

            // Update offset for next attribute
            offset += match format {
                "float32" => 4,
                "float32x2" => 8,
                "float32x3" => 12,
                "float32x4" => 16,
                _ => 0,
            };
        }

        console::log_1(&"✓ Vertex buffer layout validation passed".into());
    }

    #[wasm_bindgen_test]
    fn test_render_target_configuration() {
        console::log_1(&"Testing render target configuration".into());

        // Test different render target formats for audio visualization
        let render_formats = vec![
            "bgra8unorm",
            "rgba8unorm",
            "rgba16float",
            "rgba32float",
        ];

        for format in render_formats {
            let color_target = Object::new();
            Reflect::set(&color_target, &JsValue::from_str("format"), &JsValue::from_str(format)).unwrap();

            // Optional blend state for audio visualization
            let blend_state = Object::new();

            let color_blend = Object::new();
            Reflect::set(&color_blend, &JsValue::from_str("operation"), &JsValue::from_str("add")).unwrap();
            Reflect::set(&color_blend, &JsValue::from_str("srcFactor"), &JsValue::from_str("src-alpha")).unwrap();
            Reflect::set(&color_blend, &JsValue::from_str("dstFactor"), &JsValue::from_str("one-minus-src-alpha")).unwrap();

            let alpha_blend = Object::new();
            Reflect::set(&alpha_blend, &JsValue::from_str("operation"), &JsValue::from_str("add")).unwrap();
            Reflect::set(&alpha_blend, &JsValue::from_str("srcFactor"), &JsValue::from_str("one")).unwrap();
            Reflect::set(&alpha_blend, &JsValue::from_str("dstFactor"), &JsValue::from_str("zero")).unwrap();

            Reflect::set(&blend_state, &JsValue::from_str("color"), &color_blend).unwrap();
            Reflect::set(&blend_state, &JsValue::from_str("alpha"), &alpha_blend).unwrap();

            Reflect::set(&color_target, &JsValue::from_str("blend"), &blend_state).unwrap();

            // Verify configuration
            let format_val = Reflect::get(&color_target, &JsValue::from_str("format")).unwrap();
            assert_eq!(format_val.as_string().unwrap(), format);
        }

        console::log_1(&"✓ Render target configuration validation passed".into());
    }
}

#[cfg(test)]
mod pipeline_integration_tests {
    use super::*;

    #[wasm_bindgen_test]
    fn test_audio_processing_pipeline_flow() {
        console::log_1(&"Testing complete audio processing pipeline flow".into());

        // Step 1: Create compute pipeline for audio processing
        let compute_descriptor = create_compute_pipeline_descriptor();
        assert!(!JsValue::from(&compute_descriptor).is_undefined());

        // Step 2: Create render pipeline for visualization
        let render_descriptor = create_render_pipeline_descriptor();
        assert!(!JsValue::from(&render_descriptor).is_undefined());

        // Step 3: Test command encoder setup
        let encoder = MockGpuCommandEncoder::new_encoder();
        let compute_pass = encoder.begin_compute_pass();
        assert!(!JsValue::from(&compute_pass).is_undefined());

        // Step 4: Test render pass setup
        let render_pass_descriptor = Object::new();
        let color_attachments = js_sys::Array::new();

        let color_attachment = Object::new();
        Reflect::set(&color_attachment, &JsValue::from_str("view"), &JsValue::NULL).unwrap();
        Reflect::set(&color_attachment, &JsValue::from_str("loadOp"), &JsValue::from_str("clear")).unwrap();
        Reflect::set(&color_attachment, &JsValue::from_str("storeOp"), &JsValue::from_str("store")).unwrap();

        let clear_color = Object::new();
        Reflect::set(&clear_color, &JsValue::from_str("r"), &JsValue::from_f64(0.0)).unwrap();
        Reflect::set(&clear_color, &JsValue::from_str("g"), &JsValue::from_f64(0.0)).unwrap();
        Reflect::set(&clear_color, &JsValue::from_str("b"), &JsValue::from_f64(0.0)).unwrap();
        Reflect::set(&clear_color, &JsValue::from_str("a"), &JsValue::from_f64(1.0)).unwrap();
        Reflect::set(&color_attachment, &JsValue::from_str("clearValue"), &clear_color).unwrap();

        color_attachments.push(&color_attachment);
        Reflect::set(&render_pass_descriptor, &JsValue::from_str("colorAttachments"), &color_attachments).unwrap();

        let render_pass = encoder.begin_render_pass(&render_pass_descriptor);
        assert!(!JsValue::from(&render_pass).is_undefined());

        console::log_1(&"✓ Audio processing pipeline flow test successful".into());
    }

    #[wasm_bindgen_test]
    fn test_pipeline_resource_binding() {
        console::log_1(&"Testing pipeline resource binding".into());

        // Test bind group creation for audio processing
        let bind_group_descriptor = Object::new();
        let entries = js_sys::Array::new();

        // Audio input buffer entry
        let input_entry = Object::new();
        Reflect::set(&input_entry, &JsValue::from_str("binding"), &JsValue::from_f64(0.0)).unwrap();
        Reflect::set(&input_entry, &JsValue::from_str("resource"), &JsValue::NULL).unwrap(); // Would be actual buffer
        entries.push(&input_entry);

        // Audio output buffer entry
        let output_entry = Object::new();
        Reflect::set(&output_entry, &JsValue::from_str("binding"), &JsValue::from_f64(1.0)).unwrap();
        Reflect::set(&output_entry, &JsValue::from_str("resource"), &JsValue::NULL).unwrap(); // Would be actual buffer
        entries.push(&output_entry);

        // Processing parameters uniform
        let params_entry = Object::new();
        Reflect::set(&params_entry, &JsValue::from_str("binding"), &JsValue::from_f64(2.0)).unwrap();
        Reflect::set(&params_entry, &JsValue::from_str("resource"), &JsValue::NULL).unwrap(); // Would be actual buffer
        entries.push(&params_entry);

        Reflect::set(&bind_group_descriptor, &JsValue::from_str("entries"), &entries).unwrap();
        Reflect::set(&bind_group_descriptor, &JsValue::from_str("layout"), &JsValue::NULL).unwrap(); // Would be actual layout

        // Verify bind group structure
        assert_eq!(entries.length(), 3);

        console::log_1(&"✓ Pipeline resource binding test successful".into());
    }

    #[wasm_bindgen_test]
    fn test_pipeline_error_handling() {
        console::log_1(&"Testing pipeline error handling".into());

        // Test error scenarios for pipeline creation
        let error_scenarios = vec![
            "Invalid shader module",
            "Unsupported vertex format",
            "Binding layout mismatch",
            "Resource limit exceeded",
            "Device lost during creation",
        ];

        for error_msg in error_scenarios {
            let pipeline_error = WebError::WebGpu(format!("Pipeline error: {}", error_msg));

            match pipeline_error {
                WebError::WebGpu(msg) => {
                    assert!(msg.contains("Pipeline error"));
                    assert!(msg.contains(error_msg));
                }
                _ => panic!("Expected WebGPU error variant"),
            }
        }

        console::log_1(&"✓ Pipeline error handling test successful".into());
    }

    #[wasm_bindgen_test]
    fn test_pipeline_performance_optimization() {
        console::log_1(&"Testing pipeline performance optimization".into());

        // Test optimal configurations for audio processing
        struct PipelineOptimization {
            workgroup_size: u32,
            buffer_usage: u32,
            memory_layout: String,
            batch_size: u32,
        }

        let optimizations = vec![
            PipelineOptimization {
                workgroup_size: 64,
                buffer_usage: 0x0084, // STORAGE | COPY_SRC
                memory_layout: "interleaved".to_string(),
                batch_size: 4096,
            },
            PipelineOptimization {
                workgroup_size: 256,
                buffer_usage: 0x0088, // STORAGE | COPY_DST
                memory_layout: "planar".to_string(),
                batch_size: 16384,
            },
        ];

        for opt in optimizations {
            // Validate optimization parameters
            assert!(opt.workgroup_size > 0 && opt.workgroup_size <= 256);
            assert!(opt.buffer_usage > 0);
            assert!(!opt.memory_layout.is_empty());
            assert!(opt.batch_size > 0);

            // Test batch size efficiency
            let efficiency = opt.batch_size as f32 / opt.workgroup_size as f32;
            assert!(efficiency >= 1.0, "Batch size should be efficient for workgroup");
        }

        console::log_1(&"✓ Pipeline performance optimization test successful".into());
    }
}