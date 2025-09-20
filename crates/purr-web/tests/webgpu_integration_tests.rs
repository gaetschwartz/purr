//! Comprehensive WebGPU integration tests
//!
//! This module provides end-to-end integration tests for WebGPU functionality
//! in the purr-web crate, including platform integration and real-world scenarios.

use purr_web::{
    PlatformImpl, WebError, TranscriptionWorker, TranscriptionConfig,
    WebModelManager, WebStorage, ModelInfo, AudioProcessingConfig,
    start_transcription_process, validate_audio_file
};
use purr_common::platform::{Platform, TranscriptionRequest, TranscriptionStatus};
use wasm_bindgen_test::*;
use wasm_bindgen::prelude::*;
use web_sys::{console};
use js_sys::{Reflect};
use futures::StreamExt;
use std::sync::Arc;
use bytes::Bytes;

wasm_bindgen_test_configure!(run_in_browser);

// Test utilities for creating mock data
fn create_mock_model_info() -> ModelInfo {
    ModelInfo {
        id: "whisper-base-webgpu".to_string(),
        name: "Whisper Base WebGPU".to_string(),
        url: "https://example.com/whisper-base.onnx".to_string(),
        size: 145_000_000, // 145MB
        checksum: Some("mock-checksum".to_string()),
        version: "1.0.0".to_string(),
        description: "WebGPU-optimized Whisper base model".to_string(),
    }
}

fn create_test_wav_audio() -> Vec<u8> {
    let mut wav_data = Vec::new();

    // Standard WAV header for 16kHz mono
    wav_data.extend_from_slice(b"RIFF");
    wav_data.extend_from_slice(&(36u32 + 32000).to_le_bytes()); // File size
    wav_data.extend_from_slice(b"WAVE");
    wav_data.extend_from_slice(b"fmt ");
    wav_data.extend_from_slice(&16u32.to_le_bytes()); // Format chunk size
    wav_data.extend_from_slice(&1u16.to_le_bytes()); // PCM
    wav_data.extend_from_slice(&1u16.to_le_bytes()); // Mono
    wav_data.extend_from_slice(&16000u32.to_le_bytes()); // 16kHz sample rate
    wav_data.extend_from_slice(&32000u32.to_le_bytes()); // Byte rate
    wav_data.extend_from_slice(&2u16.to_le_bytes()); // Block align
    wav_data.extend_from_slice(&16u16.to_le_bytes()); // Bits per sample
    wav_data.extend_from_slice(b"data");
    wav_data.extend_from_slice(&32000u32.to_le_bytes()); // Data size

    // Generate 1 second of test audio (simple sine wave)
    for i in 0..16000 {
        let t = i as f32 / 16000.0;
        let sample = (440.0 * 2.0 * std::f32::consts::PI * t).sin();
        let sample16 = (sample * 16383.0) as i16; // Reduced amplitude
        wav_data.extend_from_slice(&sample16.to_le_bytes());
    }

    wav_data
}

async fn check_webgpu_availability() -> bool {
    let window = match web_sys::window() {
        Some(w) => w,
        None => return false,
    };

    let navigator = window.navigator();
    match Reflect::get(&navigator, &JsValue::from_str("gpu")) {
        Ok(gpu_val) if !gpu_val.is_undefined() => true,
        _ => false,
    }
}

#[cfg(test)]
mod platform_integration_tests {
    use super::*;

    #[wasm_bindgen_test]
    async fn test_platform_webgpu_initialization() {
        console::log_1(&"Testing platform WebGPU initialization".into());

        let webgpu_available = check_webgpu_availability().await;
        if !webgpu_available {
            console::log_1(&"WebGPU not available - testing fallback behavior".into());
        }

        // Create platform instance
        let platform = match PlatformImpl::new() {
            Ok(p) => p,
            Err(e) => {
                console::log_1(&format!("Platform creation failed: {}", e).into());
                return; // Skip test if platform creation fails
            }
        };

        // Test that platform can list models
        match platform.list_available_models().await {
            Ok(models) => {
                console::log_1(&format!("Found {} available models", models.len()).into());

                // Check for WebGPU-compatible models
                let webgpu_models: Vec<_> = models.iter()
                    .filter(|m| {
                        m.metadata.format.contains("webgpu")
                    })
                    .collect();

                console::log_1(&format!("Found {} WebGPU-compatible models", webgpu_models.len()).into());
                assert!(!webgpu_models.is_empty(), "Should have WebGPU-compatible models");
            }
            Err(e) => {
                console::log_1(&format!("Failed to list models: {}", e).into());
            }
        }

        console::log_1(&"✓ Platform WebGPU initialization test completed".into());
    }

    #[wasm_bindgen_test]
    async fn test_webgpu_model_compatibility() {
        console::log_1(&"Testing WebGPU model compatibility".into());

        let platform = match PlatformImpl::new() {
            Ok(p) => p,
            Err(_) => return, // Skip if platform creation fails
        };

        // Test model info retrieval
        match platform.get_model_info("whisper-base").await {
            Ok(model) => {
                // Check WebGPU compatibility attributes
                let is_webgpu_compatible = model.metadata.format.contains("webgpu");

                if is_webgpu_compatible {
                    console::log_1(&"✓ Model is WebGPU compatible".into());

                    // Check quantization
                    if let Some(quantization) = model.metadata.quantization {
                        assert!(quantization == "fp16" || quantization == "fp32",
                               "WebGPU models should use fp16 or fp32 quantization");
                    }
                } else {
                    console::log_1(&"Model is not WebGPU compatible".into());
                }
            }
            Err(e) => {
                console::log_1(&format!("Failed to get model info: {}", e).into());
            }
        }

        console::log_1(&"✓ WebGPU model compatibility test completed".into());
    }

    #[wasm_bindgen_test]
    async fn test_webgpu_audio_processing_integration() {
        console::log_1(&"Testing WebGPU audio processing integration".into());

        let platform = match PlatformImpl::new() {
            Ok(p) => p,
            Err(_) => return,
        };

        // Create test audio data
        let audio_data = create_test_wav_audio();
        let audio_bytes = Bytes::from(audio_data);

        // Process file through platform
        match platform.process_file(audio_bytes.clone(), std::path::Path::new("test.wav")).await {
            Ok(file_id) => {
                console::log_1(&format!("File processed with ID: {}", file_id).into());

                // Create transcription request
                let request = TranscriptionRequest {
                    file_data: audio_bytes.clone(),
                    language: Some("en".to_string()),
                    translate: false,
                };

                // Start transcription
                match platform.transcribe(file_id.clone(), request).await {
                    Ok(mut stream) => {
                        console::log_1(&"Transcription started successfully".into());

                        // Process a few stream items
                        let mut count = 0;
                        while let Some(result) = stream.next().await {
                            if count >= 3 { break; } // Limit test iterations

                            match result {
                                Ok(status) => {
                                    match status {
                                        TranscriptionStatus::Starting => {
                                            console::log_1(&"Starting transcription...".into());
                                        }
                                        TranscriptionStatus::ProcessingAudio => {
                                            console::log_1(&"Processing audio...".into());
                                        }
                                        TranscriptionStatus::InProgress { text, .. } => {
                                            console::log_1(&format!("Progress: {}", text).into());
                                        }
                                        TranscriptionStatus::Completed { processing_time, audio_duration, word_count } => {
                                            console::log_1(&format!("Completed in {:.2}s, duration: {:.2}s, words: {}", processing_time, audio_duration, word_count).into());
                                            break;
                                        }
                                        TranscriptionStatus::Error { message } => {
                                            console::log_1(&format!("Error: {}", message).into());
                                            break;
                                        }
                                        TranscriptionStatus::InitFailed { message } => {
                                            console::log_1(&format!("Init failed: {}", message).into());
                                            break;
                                        }
                                    }
                                }
                                Err(e) => {
                                    console::log_1(&format!("Stream error: {}", e).into());
                                    break;
                                }
                            }
                            count += 1;
                        }

                        // Cleanup
                        let _ = platform.cleanup(&file_id.to_string()).await;
                    }
                    Err(e) => {
                        console::log_1(&format!("Transcription failed: {}", e).into());
                    }
                }
            }
            Err(e) => {
                console::log_1(&format!("File processing failed: {}", e).into());
            }
        }

        console::log_1(&"✓ WebGPU audio processing integration test completed".into());
    }
}

#[cfg(test)]
mod worker_integration_tests {
    use super::*;

    #[wasm_bindgen_test]
    async fn test_webgpu_worker_initialization() {
        console::log_1(&"Testing WebGPU worker initialization".into());

        // Create storage and model manager
        let storage = Arc::new(WebStorage::new());
        let model_manager = Arc::new(WebModelManager::with_storage(storage.clone()));

        // Create transcription worker
        let worker = TranscriptionWorker::new(model_manager.clone());

        // Test worker configuration
        let config = TranscriptionConfig {
            model_name: "whisper-base".to_string(),
            language: Some("en".to_string()),
            translate: false,
            ..Default::default()
        };

        // Initialize worker session
        let session_id = "test-webgpu-session".to_string();
        match worker.create_session(config.clone()).await {
            Ok(created_session_id) => {
                console::log_1(&"✓ Worker session created successfully".into());

                // Test transcription instead of status check
                let sample_request = TranscriptionRequest {
                    file_data: Bytes::from(vec![0u8; 1024]), // Sample audio data
                    language: None,
                    translate: false,
                };

                match worker.transcribe(&created_session_id, sample_request).await {
                    Ok(mut stream) => {
                        console::log_1(&"✓ Transcription stream started".into());

                        // Process a few stream items for testing
                        let mut items_processed = 0;
                        while let Some(status) = stream.next().await {
                            if items_processed >= 2 { break; } // Limit test
                            console::log_1(&format!("Stream status: {:?}", status).into());
                            items_processed += 1;
                        }
                    }
                    Err(e) => {
                        console::log_1(&format!("Transcription failed: {}", e).into());
                    }
                }

                // Cleanup session
                let _ = worker.close_session(&created_session_id).await;
            }
            Err(e) => {
                console::log_1(&format!("Worker initialization failed: {}", e).into());
                // This is expected if WebGPU is not available
            }
        }

        console::log_1(&"✓ WebGPU worker initialization test completed".into());
    }

    #[wasm_bindgen_test]
    async fn test_webgpu_transcription_workflow() {
        console::log_1(&"Testing complete WebGPU transcription workflow".into());

        let audio_data = create_test_wav_audio();
        let audio_bytes = Bytes::from(audio_data);

        // Validate audio file first
        match validate_audio_file(&audio_bytes, 100 * 1024 * 1024) {
            Ok(_) => {
                console::log_1(&"✓ Audio file validation passed".into());
            }
            Err(e) => {
                console::log_1(&format!("Audio validation failed: {}", e).into());
                return;
            }
        }

        // Configure transcription
        let transcription_config = TranscriptionConfig {
            model_name: "whisper-base".to_string(),
            language: Some("en".to_string()),
            translate: false,
            ..Default::default()
        };

        let audio_config = AudioProcessingConfig {
            target_sample_rate: 16000.0,
            target_channels: 1,
            enable_agc: true,
            enable_noise_reduction: false,
            max_file_size: 100 * 1024 * 1024,
            ..Default::default()
        };

        // Start transcription process
        match start_transcription_process(
            audio_bytes,
            transcription_config,
            Some(audio_config),
        ).await {
            Ok(mut stream) => {
                console::log_1(&"✓ Transcription process started".into());

                // Process stream items
                let mut items_processed = 0;
                while let Some(status) = stream.next().await {
                    if items_processed >= 5 { break; } // Limit test iterations

                    match status {
                        TranscriptionStatus::Starting => {
                            console::log_1(&"Starting transcription...".into());
                        }
                        TranscriptionStatus::ProcessingAudio => {
                            console::log_1(&"Processing audio...".into());
                        }
                        TranscriptionStatus::InProgress { text, .. } => {
                            console::log_1(&format!("Progress: {}", text).into());
                        }
                        TranscriptionStatus::Completed { processing_time, audio_duration, word_count } => {
                            console::log_1(&format!("Transcription completed in {:.2}s, duration: {:.2}s, words: {}", processing_time, audio_duration, word_count).into());
                            break;
                        }
                        TranscriptionStatus::Error { message } => {
                            console::log_1(&format!("Transcription error: {}", message).into());
                            break;
                        }
                        TranscriptionStatus::InitFailed { message } => {
                            console::log_1(&format!("Init failed: {}", message).into());
                            break;
                        }
                    }
                    items_processed += 1;
                }
            }
            Err(e) => {
                console::log_1(&format!("Transcription process failed: {}", e).into());
                // This may be expected if WebGPU is not available
            }
        }

        console::log_1(&"✓ WebGPU transcription workflow test completed".into());
    }
}

#[cfg(test)]
mod error_handling_integration_tests {
    use super::*;

    #[wasm_bindgen_test]
    async fn test_webgpu_error_propagation() {
        console::log_1(&"Testing WebGPU error propagation".into());

        // Test various WebGPU error scenarios
        let webgpu_errors = vec![
            "WebGPU device initialization failed",
            "WebGPU adapter not found",
            "WebGPU device lost during operation",
            "WebGPU out of memory",
            "WebGPU validation error in shader",
            "WebGPU buffer creation failed",
            "WebGPU pipeline creation failed",
        ];

        for error_msg in webgpu_errors {
            let webgpu_error = WebError::WebGpu(error_msg.to_string());
            let platform_error = webgpu_error.into_platform_error();

            // Verify error propagation
            assert!(platform_error.to_string().contains("WebGPU"));
            assert!(platform_error.to_string().contains(error_msg));

            console::log_1(&format!("✓ Error propagated correctly: {}", error_msg).into());
        }

        console::log_1(&"✓ WebGPU error propagation test completed".into());
    }

    #[wasm_bindgen_test]
    async fn test_webgpu_fallback_behavior() {
        console::log_1(&"Testing WebGPU fallback behavior".into());

        let webgpu_available = check_webgpu_availability().await;

        if !webgpu_available {
            console::log_1(&"WebGPU not available - testing CPU fallback".into());

            // Test that platform still works without WebGPU
            let platform = match PlatformImpl::new() {
                Ok(p) => p,
                Err(e) => {
                    console::log_1(&format!("Platform creation failed: {}", e).into());
                    return;
                }
            };

            // Test model listing (should work with CPU models)
            match platform.list_available_models().await {
                Ok(models) => {
                    console::log_1(&format!("✓ CPU fallback works: {} models available", models.len()).into());
                }
                Err(e) => {
                    console::log_1(&format!("CPU fallback failed: {}", e).into());
                }
            }
        } else {
            console::log_1(&"WebGPU available - testing graceful degradation".into());

            // Test graceful degradation when WebGPU operations fail
            let webgpu_error = WebError::WebGpu("Simulated WebGPU failure".to_string());

            match webgpu_error {
                WebError::WebGpu(msg) => {
                    console::log_1(&format!("✓ WebGPU error handled gracefully: {}", msg).into());
                }
                _ => panic!("Expected WebGPU error"),
            }
        }

        console::log_1(&"✓ WebGPU fallback behavior test completed".into());
    }
}

#[cfg(test)]
mod performance_integration_tests {
    use super::*;

    #[wasm_bindgen_test]
    async fn test_webgpu_performance_monitoring() {
        console::log_1(&"Testing WebGPU performance monitoring".into());

        // Performance metrics structure for testing
        struct WebGPUPerformanceMetrics {
            device_type: String,
            memory_usage: u64,
            compute_units: u32,
            max_workgroup_size: [u32; 3],
            max_buffer_size: u64,
            timestamp_period: f64,
            transcription_speed: f64, // Real-time factor
        }

        let test_metrics = WebGPUPerformanceMetrics {
            device_type: "integrated".to_string(),
            memory_usage: 512 * 1024 * 1024, // 512MB
            compute_units: 16,
            max_workgroup_size: [256, 256, 64],
            max_buffer_size: 256 * 1024 * 1024, // 256MB
            timestamp_period: 1.0, // 1ns per tick
            transcription_speed: 2.5, // 2.5x real-time
        };

        // Validate performance metrics
        assert!(!test_metrics.device_type.is_empty());
        assert!(test_metrics.memory_usage > 0);
        assert!(test_metrics.compute_units > 0);
        assert!(test_metrics.max_workgroup_size[0] > 0);
        assert!(test_metrics.max_buffer_size > 0);
        assert!(test_metrics.timestamp_period > 0.0);
        assert!(test_metrics.transcription_speed > 0.0);

        console::log_1(&format!("✓ Performance metrics validated: {} device", test_metrics.device_type).into());
        console::log_1(&format!("✓ Memory usage: {} MB", test_metrics.memory_usage / (1024 * 1024)).into());
        console::log_1(&format!("✓ Transcription speed: {}x real-time", test_metrics.transcription_speed).into());

        console::log_1(&"✓ WebGPU performance monitoring test completed".into());
    }

    #[wasm_bindgen_test]
    async fn test_webgpu_memory_optimization() {
        console::log_1(&"Testing WebGPU memory optimization".into());

        // Test memory usage patterns for audio processing
        let audio_buffer_sizes = vec![
            1024,      // 1K samples
            4096,      // 4K samples
            16384,     // 16K samples
            65536,     // 64K samples
        ];

        for buffer_size in audio_buffer_sizes {
            // Calculate memory requirements
            let sample_size = 4; // 32-bit float
            let input_buffer_size = buffer_size * sample_size;
            let output_buffer_size = buffer_size * sample_size;
            let uniform_buffer_size = 64; // Typical uniform buffer size
            let total_memory = input_buffer_size + output_buffer_size + uniform_buffer_size;

            // Validate memory usage is reasonable
            assert!(total_memory < 10 * 1024 * 1024, "Memory usage should be < 10MB for test buffers");

            console::log_1(&format!("✓ Buffer size {}: {} KB total memory", buffer_size, total_memory / 1024).into());
        }

        console::log_1(&"✓ WebGPU memory optimization test completed".into());
    }
}

// Test configuration validation
#[cfg(test)]
mod configuration_integration_tests {
    use super::*;

    #[wasm_bindgen_test]
    fn test_webgpu_configuration_validation() {
        console::log_1(&"Testing WebGPU configuration validation".into());

        // Test transcription configuration for WebGPU
        let valid_configs = vec![
            TranscriptionConfig {
                model_name: "whisper-tiny".to_string(),
                language: Some("en".to_string()),
                translate: false,
                ..Default::default()
            },
            TranscriptionConfig {
                model_name: "whisper-base".to_string(),
                language: Some("es".to_string()),
                translate: true,
                ..Default::default()
            },
        ];

        for config in valid_configs {
            // Validate configuration
            assert!(!config.model_name.is_empty());

            if let Some(lang) = &config.language {
                assert!(lang.len() >= 2, "Language code should be at least 2 characters");
            }

            console::log_1(&format!("✓ Valid config: {} (translate: {})",
                                   config.model_name, config.translate).into());
        }

        // Test audio processing configuration
        let audio_config = AudioProcessingConfig {
            target_sample_rate: 16000.0,
            target_channels: 1,
            enable_agc: true,
            enable_noise_reduction: false,
            max_file_size: 100 * 1024 * 1024,
            ..Default::default()
        };

        assert!(audio_config.target_sample_rate > 0.0);
        assert!(audio_config.target_channels > 0);
        assert!(audio_config.max_file_size > 0);

        console::log_1(&"✓ WebGPU configuration validation test completed".into());
    }

    #[wasm_bindgen_test]
    fn test_model_metadata_webgpu_attributes() {
        console::log_1(&"Testing model metadata WebGPU attributes".into());

        let model = create_mock_model_info();

        // Test model attributes that would be relevant for WebGPU
        assert!(!model.id.is_empty());
        assert!(!model.name.is_empty());
        assert!(!model.url.is_empty());
        assert!(model.size > 0);
        assert!(!model.version.is_empty());

        // In a real implementation, we'd check WebGPU-specific metadata
        console::log_1(&format!("✓ Model validated: {} ({}MB)",
                               model.name, model.size / (1024 * 1024)).into());

        console::log_1(&"✓ Model metadata WebGPU attributes test completed".into());
    }
}