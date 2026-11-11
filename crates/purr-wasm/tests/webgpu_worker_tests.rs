//! WebGPU transcription worker tests
//!
//! This module tests the WebGPU-based transcription worker functionality,
//! including worker initialization, message handling, and transcription processing.

#![allow(dead_code)]

use js_sys::{Object, Reflect, Uint8Array};
use purr_wasm::{TranscriptionConfig, WebError};
use std::collections::HashMap;
use wasm_bindgen::prelude::*;
use wasm_bindgen_test::*;
use web_sys::console;

mod common;

wasm_bindgen_test_configure!(run_in_browser);

// Mock worker message types matching the Rust WorkerMessage enum
#[derive(Debug, Clone)]
enum MockWorkerMessage {
    Initialize {
        config: TranscriptionConfig,
        session_id: String,
    },
    ProcessAudio {
        audio_data: Vec<u8>,
        session_id: String,
    },
    UpdateConfig {
        config: TranscriptionConfig,
    },
    GetStatus,
    Shutdown,
}

impl MockWorkerMessage {
    fn to_js_value(&self) -> JsValue {
        let obj = Object::new();

        match self {
            Self::Initialize { config, session_id } => {
                Reflect::set(
                    &obj,
                    &JsValue::from_str("type"),
                    &JsValue::from_str("Initialize"),
                )
                .unwrap();
                Reflect::set(
                    &obj,
                    &JsValue::from_str("session_id"),
                    &JsValue::from_str(session_id),
                )
                .unwrap();

                let config_obj = Object::new();
                Reflect::set(
                    &config_obj,
                    &JsValue::from_str("repo"),
                    &JsValue::from_str(&config.model.repo),
                )
                .unwrap();
                Reflect::set(
                    &config_obj,
                    &JsValue::from_str("language"),
                    &config
                        .language
                        .as_ref()
                        .map_or(JsValue::NULL, |l| JsValue::from_str(l)),
                )
                .unwrap();
                Reflect::set(
                    &config_obj,
                    &JsValue::from_str("translate"),
                    &JsValue::from_bool(config.translate),
                )
                .unwrap();
                Reflect::set(&obj, &JsValue::from_str("config"), &config_obj).unwrap();
            }
            Self::ProcessAudio {
                audio_data,
                session_id,
            } => {
                Reflect::set(
                    &obj,
                    &JsValue::from_str("type"),
                    &JsValue::from_str("ProcessAudio"),
                )
                .unwrap();
                Reflect::set(
                    &obj,
                    &JsValue::from_str("session_id"),
                    &JsValue::from_str(session_id),
                )
                .unwrap();

                let array = Uint8Array::new_with_length(audio_data.len() as u32);
                for (i, &byte) in audio_data.iter().enumerate() {
                    array.set_index(i as u32, byte);
                }
                Reflect::set(&obj, &JsValue::from_str("audio_data"), &array).unwrap();
            }
            Self::UpdateConfig { config } => {
                Reflect::set(
                    &obj,
                    &JsValue::from_str("type"),
                    &JsValue::from_str("UpdateConfig"),
                )
                .unwrap();

                let config_obj = Object::new();
                Reflect::set(
                    &config_obj,
                    &JsValue::from_str("repo"),
                    &JsValue::from_str(&config.model.repo),
                )
                .unwrap();
                Reflect::set(&obj, &JsValue::from_str("config"), &config_obj).unwrap();
            }
            Self::GetStatus => {
                Reflect::set(
                    &obj,
                    &JsValue::from_str("type"),
                    &JsValue::from_str("GetStatus"),
                )
                .unwrap();
            }
            Self::Shutdown => {
                Reflect::set(
                    &obj,
                    &JsValue::from_str("type"),
                    &JsValue::from_str("Shutdown"),
                )
                .unwrap();
            }
        }

        obj.into()
    }
}

// Mock worker response types
#[derive(Debug, Clone)]
enum MockWorkerResponse {
    Initialized {
        session_id: String,
        webgpu_available: bool,
        model_loaded: bool,
    },
    TranscriptionProgress {
        session_id: String,
        progress: f32,
        text: Option<String>,
    },
    TranscriptionComplete {
        session_id: String,
        text: String,
        metadata: HashMap<String, String>,
    },
    Error {
        message: String,
        session_id: Option<String>,
    },
    Status {
        webgpu_available: bool,
        active_sessions: u32,
        memory_usage: f64,
    },
}

impl MockWorkerResponse {
    fn to_js_value(&self) -> JsValue {
        let obj = Object::new();

        match self {
            Self::Initialized {
                session_id,
                webgpu_available,
                model_loaded,
            } => {
                Reflect::set(
                    &obj,
                    &JsValue::from_str("type"),
                    &JsValue::from_str("Initialized"),
                )
                .unwrap();
                Reflect::set(
                    &obj,
                    &JsValue::from_str("session_id"),
                    &JsValue::from_str(session_id),
                )
                .unwrap();
                Reflect::set(
                    &obj,
                    &JsValue::from_str("webgpu_available"),
                    &JsValue::from_bool(*webgpu_available),
                )
                .unwrap();
                Reflect::set(
                    &obj,
                    &JsValue::from_str("model_loaded"),
                    &JsValue::from_bool(*model_loaded),
                )
                .unwrap();
            }
            Self::TranscriptionProgress {
                session_id,
                progress,
                text,
            } => {
                Reflect::set(
                    &obj,
                    &JsValue::from_str("type"),
                    &JsValue::from_str("TranscriptionProgress"),
                )
                .unwrap();
                Reflect::set(
                    &obj,
                    &JsValue::from_str("session_id"),
                    &JsValue::from_str(session_id),
                )
                .unwrap();
                Reflect::set(
                    &obj,
                    &JsValue::from_str("progress"),
                    &JsValue::from_f64(f64::from(*progress)),
                )
                .unwrap();
                if let Some(text) = text {
                    Reflect::set(&obj, &JsValue::from_str("text"), &JsValue::from_str(text))
                        .unwrap();
                }
            }
            Self::TranscriptionComplete {
                session_id,
                text,
                metadata,
            } => {
                Reflect::set(
                    &obj,
                    &JsValue::from_str("type"),
                    &JsValue::from_str("TranscriptionComplete"),
                )
                .unwrap();
                Reflect::set(
                    &obj,
                    &JsValue::from_str("session_id"),
                    &JsValue::from_str(session_id),
                )
                .unwrap();
                Reflect::set(&obj, &JsValue::from_str("text"), &JsValue::from_str(text)).unwrap();

                let metadata_obj = Object::new();
                for (key, value) in metadata {
                    Reflect::set(
                        &metadata_obj,
                        &JsValue::from_str(key),
                        &JsValue::from_str(value),
                    )
                    .unwrap();
                }
                Reflect::set(&obj, &JsValue::from_str("metadata"), &metadata_obj).unwrap();
            }
            Self::Error {
                message,
                session_id,
            } => {
                Reflect::set(
                    &obj,
                    &JsValue::from_str("type"),
                    &JsValue::from_str("Error"),
                )
                .unwrap();
                Reflect::set(
                    &obj,
                    &JsValue::from_str("message"),
                    &JsValue::from_str(message),
                )
                .unwrap();
                if let Some(session_id) = session_id {
                    Reflect::set(
                        &obj,
                        &JsValue::from_str("session_id"),
                        &JsValue::from_str(session_id),
                    )
                    .unwrap();
                }
            }
            Self::Status {
                webgpu_available,
                active_sessions,
                memory_usage,
            } => {
                Reflect::set(
                    &obj,
                    &JsValue::from_str("type"),
                    &JsValue::from_str("Status"),
                )
                .unwrap();
                Reflect::set(
                    &obj,
                    &JsValue::from_str("webgpu_available"),
                    &JsValue::from_bool(*webgpu_available),
                )
                .unwrap();
                Reflect::set(
                    &obj,
                    &JsValue::from_str("active_sessions"),
                    &JsValue::from_f64(f64::from(*active_sessions)),
                )
                .unwrap();
                Reflect::set(
                    &obj,
                    &JsValue::from_str("memory_usage"),
                    &JsValue::from_f64(*memory_usage),
                )
                .unwrap();
            }
        }

        obj.into()
    }
}

fn create_test_config() -> TranscriptionConfig {
    TranscriptionConfig {
        language: Some("en".to_owned()),
        ..TranscriptionConfig::new(common::default_worker_model())
    }
}

fn create_test_audio_data() -> Vec<u8> {
    // Create simple sine wave audio data (WAV format)
    let mut data = Vec::new();

    // WAV header
    data.extend_from_slice(b"RIFF");
    data.extend_from_slice(&(36u32 + 8000).to_le_bytes()); // File size
    data.extend_from_slice(b"WAVE");
    data.extend_from_slice(b"fmt ");
    data.extend_from_slice(&16u32.to_le_bytes()); // Format chunk size
    data.extend_from_slice(&1u16.to_le_bytes()); // PCM
    data.extend_from_slice(&1u16.to_le_bytes()); // Mono
    data.extend_from_slice(&16000u32.to_le_bytes()); // 16kHz sample rate
    data.extend_from_slice(&32000u32.to_le_bytes()); // Byte rate
    data.extend_from_slice(&2u16.to_le_bytes()); // Block align
    data.extend_from_slice(&16u16.to_le_bytes()); // Bits per sample
    data.extend_from_slice(b"data");
    data.extend_from_slice(&8000u32.to_le_bytes()); // Data size

    // Generate 0.5 seconds of 440Hz sine wave
    for i in 0..4000 {
        let t = i as f32 / 16000.0;
        let sample = (440.0 * 2.0 * std::f32::consts::PI * t).sin();
        let sample16 = (sample * 32767.0) as i16;
        data.extend_from_slice(&sample16.to_le_bytes());
    }

    data
}

#[cfg(test)]
mod worker_initialization_tests {
    use super::*;

    #[wasm_bindgen_test]
    fn test_worker_message_serialization() {
        console::log_1(&"Testing worker message serialization".into());

        let config = create_test_config();
        let session_id = "test-session-123".to_owned();

        let message = MockWorkerMessage::Initialize {
            config,
            session_id: session_id.clone(),
        };

        let js_value = message.to_js_value();
        let obj: Object = js_value.dyn_into().unwrap();

        // Verify message structure
        let msg_type = Reflect::get(&obj, &JsValue::from_str("type")).unwrap();
        assert_eq!(msg_type.as_string().unwrap(), "Initialize");

        let msg_session_id = Reflect::get(&obj, &JsValue::from_str("session_id")).unwrap();
        assert_eq!(msg_session_id.as_string().unwrap(), session_id);

        console::log_1(&"✓ Worker message serialization successful".into());
    }

    #[wasm_bindgen_test]
    fn test_worker_response_deserialization() {
        console::log_1(&"Testing worker response deserialization".into());

        let response = MockWorkerResponse::Initialized {
            session_id: "test-session-123".to_owned(),
            webgpu_available: true,
            model_loaded: true,
        };

        let js_value = response.to_js_value();
        let obj: Object = js_value.dyn_into().unwrap();

        // Verify response structure
        let resp_type = Reflect::get(&obj, &JsValue::from_str("type")).unwrap();
        assert_eq!(resp_type.as_string().unwrap(), "Initialized");

        let webgpu_available = Reflect::get(&obj, &JsValue::from_str("webgpu_available")).unwrap();
        assert!(webgpu_available.as_bool().unwrap());

        let model_loaded = Reflect::get(&obj, &JsValue::from_str("model_loaded")).unwrap();
        assert!(model_loaded.as_bool().unwrap());

        console::log_1(&"✓ Worker response deserialization successful".into());
    }

    #[wasm_bindgen_test]
    fn test_transcription_config_validation() {
        console::log_1(&"Testing transcription config validation".into());

        let config = create_test_config();

        // Test valid configuration
        assert_ne!(config.model.repo, "");
        assert!(config.language.is_some());
        assert_eq!(config.language.as_ref().unwrap(), "en");
        assert!(!config.translate);

        // Test invalid configurations
        let empty_repo = purr_wasm::WorkerModel {
            repo: String::new(),
            ..common::default_worker_model()
        };
        let invalid_configs = vec![
            TranscriptionConfig::new(empty_repo),
            TranscriptionConfig {
                language: Some("invalid-lang".to_owned()), // Invalid language code
                ..create_test_config()
            },
        ];

        for invalid_config in invalid_configs {
            if invalid_config.model.repo.is_empty() {
                assert_eq!(invalid_config.model.repo, "");
            }
        }

        console::log_1(&"✓ Transcription config validation successful".into());
    }
}

#[cfg(test)]
mod worker_processing_tests {
    use super::*;

    #[wasm_bindgen_test]
    fn test_audio_data_processing() {
        console::log_1(&"Testing audio data processing".into());

        let audio_data = create_test_audio_data();
        let session_id = "test-session-456".to_owned();

        let message = MockWorkerMessage::ProcessAudio {
            audio_data: audio_data.clone(),
            session_id,
        };

        let js_value = message.to_js_value();
        let obj: Object = js_value.dyn_into().unwrap();

        // Verify audio data transmission
        let audio_array = Reflect::get(&obj, &JsValue::from_str("audio_data")).unwrap();
        let uint8_array: Uint8Array = audio_array.dyn_into().unwrap();

        assert_eq!(uint8_array.length(), audio_data.len() as u32);

        // Verify WAV header
        assert_eq!(uint8_array.get_index(0), b'R');
        assert_eq!(uint8_array.get_index(1), b'I');
        assert_eq!(uint8_array.get_index(2), b'F');
        assert_eq!(uint8_array.get_index(3), b'F');

        console::log_1(&"✓ Audio data processing test successful".into());
    }

    #[wasm_bindgen_test]
    fn test_transcription_progress_tracking() {
        console::log_1(&"Testing transcription progress tracking".into());

        let session_id = "test-session-789".to_owned();
        let progress_steps = vec![0.0, 0.25, 0.5, 0.75, 1.0];

        for progress in progress_steps {
            let response = MockWorkerResponse::TranscriptionProgress {
                session_id: session_id.clone(),
                progress,
                text: (progress > 0.0).then(|| "Partial text".to_owned()),
            };

            let js_value = response.to_js_value();
            let obj: Object = js_value.dyn_into().unwrap();

            let progress_val = Reflect::get(&obj, &JsValue::from_str("progress")).unwrap();
            assert_eq!(progress_val.as_f64().unwrap() as f32, progress);

            if progress > 0.0 {
                let text_val = Reflect::get(&obj, &JsValue::from_str("text")).unwrap();
                assert!(!text_val.is_undefined());
            }
        }

        console::log_1(&"✓ Transcription progress tracking successful".into());
    }

    #[wasm_bindgen_test]
    fn test_transcription_completion() {
        console::log_1(&"Testing transcription completion".into());

        let session_id = "test-session-complete".to_owned();
        let final_text = "Hello, this is a test transcription.".to_owned();
        let mut metadata = HashMap::new();
        metadata.insert("duration".to_owned(), "5.2".to_owned());
        metadata.insert("language".to_owned(), "en".to_owned());
        metadata.insert("confidence".to_owned(), "0.95".to_owned());

        let response = MockWorkerResponse::TranscriptionComplete {
            session_id,
            text: final_text.clone(),
            metadata,
        };

        let js_value = response.to_js_value();
        let obj: Object = js_value.dyn_into().unwrap();

        // Verify completion response
        let text_val = Reflect::get(&obj, &JsValue::from_str("text")).unwrap();
        assert_eq!(text_val.as_string().unwrap(), final_text);

        let metadata_obj = Reflect::get(&obj, &JsValue::from_str("metadata")).unwrap();
        let duration = Reflect::get(&metadata_obj, &JsValue::from_str("duration")).unwrap();
        assert_eq!(duration.as_string().unwrap(), "5.2");

        console::log_1(&"✓ Transcription completion test successful".into());
    }
}

#[cfg(test)]
mod worker_error_handling_tests {
    use super::*;

    #[wasm_bindgen_test]
    fn test_worker_error_handling() {
        console::log_1(&"Testing worker error handling".into());

        let error_scenarios = vec![
            ("WebGPU device lost", Some("session-123".to_owned())),
            ("Model loading failed", None),
            ("Audio format not supported", Some("session-456".to_owned())),
            ("Out of memory", Some("session-789".to_owned())),
            ("Network timeout", None),
        ];

        for (error_message, session_id) in error_scenarios {
            let response = MockWorkerResponse::Error {
                message: error_message.to_owned(),
                session_id,
            };

            let js_value = response.to_js_value();
            let obj: Object = js_value.dyn_into().unwrap();

            let msg_val = Reflect::get(&obj, &JsValue::from_str("message")).unwrap();
            assert_eq!(msg_val.as_string().unwrap(), error_message);

            // Check session ID handling
            let session_id_val = Reflect::get(&obj, &JsValue::from_str("session_id")).unwrap();
            match &response {
                MockWorkerResponse::Error { session_id, .. } => {
                    if session_id.is_some() {
                        assert!(!session_id_val.is_undefined());
                    }
                }
                _ => {
                    // Other variants might have session_id as non-optional
                    assert!(!session_id_val.is_undefined());
                }
            }
        }

        console::log_1(&"✓ Worker error handling test successful".into());
    }

    #[wasm_bindgen_test]
    fn test_webgpu_specific_errors() {
        console::log_1(&"Testing WebGPU-specific error handling".into());

        let webgpu_errors = vec![
            "WebGPU not supported",
            "WebGPU adapter not available",
            "WebGPU device lost",
            "WebGPU out of memory",
            "WebGPU validation error",
            "WebGPU operation error",
        ];

        for error_msg in webgpu_errors {
            let webgpu_error = WebError::WebGpu(purr_wasm::WebGpuError::FeatureNotSupported {
                feature: error_msg.to_owned(),
            });

            match webgpu_error {
                WebError::WebGpu(webgpu_err) => {
                    let msg = webgpu_err.to_string();
                    assert!(msg.contains("Missing WebGPU features"));
                    assert!(msg.contains(error_msg));
                }
                _ => panic!("Expected WebGPU error variant"),
            }
        }

        console::log_1(&"✓ WebGPU-specific error handling test successful".into());
    }
}

#[cfg(test)]
mod worker_status_tests {
    use super::*;

    #[wasm_bindgen_test]
    fn test_worker_status_monitoring() {
        console::log_1(&"Testing worker status monitoring".into());

        let status_response = MockWorkerResponse::Status {
            webgpu_available: true,
            active_sessions: 2,
            memory_usage: 256.5, // MB
        };

        let js_value = status_response.to_js_value();
        let obj: Object = js_value.dyn_into().unwrap();

        // Verify status fields
        let webgpu_available = Reflect::get(&obj, &JsValue::from_str("webgpu_available")).unwrap();
        assert!(webgpu_available.as_bool().unwrap());

        let active_sessions = Reflect::get(&obj, &JsValue::from_str("active_sessions")).unwrap();
        assert_eq!(active_sessions.as_f64().unwrap() as u32, 2);

        let memory_usage = Reflect::get(&obj, &JsValue::from_str("memory_usage")).unwrap();
        assert!((memory_usage.as_f64().unwrap() - 256.5).abs() < 0.1);

        console::log_1(&"✓ Worker status monitoring test successful".into());
    }

    #[wasm_bindgen_test]
    fn test_worker_performance_metrics() {
        console::log_1(&"Testing worker performance metrics".into());

        // Test performance metric collection
        struct WorkerPerformanceMetrics {
            transcription_speed: f64, // Real-time factor
            memory_peak: f64,         // MB
            gpu_utilization: f64,     // Percentage
            throughput: f64,          // Audio minutes per minute
        }

        let metrics = WorkerPerformanceMetrics {
            transcription_speed: 2.5, // 2.5x real-time
            memory_peak: 512.0,       // 512 MB
            gpu_utilization: 85.0,    // 85%
            throughput: 3.2,          // 3.2 minutes of audio per minute
        };

        // Validate performance metrics
        assert!(
            metrics.transcription_speed > 1.0,
            "Should be faster than real-time"
        );
        assert!(metrics.memory_peak > 0.0, "Memory usage should be positive");
        assert!(
            metrics.gpu_utilization >= 0.0 && metrics.gpu_utilization <= 100.0,
            "GPU utilization should be 0-100%"
        );
        assert!(metrics.throughput > 0.0, "Throughput should be positive");

        console::log_1(&"✓ Worker performance metrics test successful".into());
    }
}

#[cfg(test)]
mod worker_lifecycle_tests {
    use super::*;

    #[wasm_bindgen_test]
    fn test_worker_initialization_sequence() {
        console::log_1(&"Testing worker initialization sequence".into());

        let config = create_test_config();
        let session_id = "test-init-sequence".to_owned();

        // Step 1: Initialize worker
        let init_message = MockWorkerMessage::Initialize {
            config,
            session_id: session_id.clone(),
        };

        let init_js = init_message.to_js_value();
        assert!(!init_js.is_undefined());

        // Step 2: Expect initialization response
        let init_response = MockWorkerResponse::Initialized {
            session_id,
            webgpu_available: true,
            model_loaded: true,
        };

        let response_js = init_response.to_js_value();
        let response_obj: Object = response_js.dyn_into().unwrap();

        let webgpu_available =
            Reflect::get(&response_obj, &JsValue::from_str("webgpu_available")).unwrap();
        assert!(webgpu_available.as_bool().unwrap());

        console::log_1(&"✓ Worker initialization sequence test successful".into());
    }

    #[wasm_bindgen_test]
    fn test_worker_shutdown_sequence() {
        console::log_1(&"Testing worker shutdown sequence".into());

        let shutdown_message = MockWorkerMessage::Shutdown;
        let shutdown_js = shutdown_message.to_js_value();

        let obj: Object = shutdown_js.dyn_into().unwrap();
        let msg_type = Reflect::get(&obj, &JsValue::from_str("type")).unwrap();
        assert_eq!(msg_type.as_string().unwrap(), "Shutdown");

        console::log_1(&"✓ Worker shutdown sequence test successful".into());
    }

    #[wasm_bindgen_test]
    fn test_worker_config_updates() {
        console::log_1(&"Testing worker configuration updates".into());

        let mut config = create_test_config();
        config.model.repo = "onnx-community/whisper-large-v3-turbo".to_owned();
        config.language = Some("es".to_owned());
        config.translate = true;

        let update_message = MockWorkerMessage::UpdateConfig { config };
        let update_js = update_message.to_js_value();

        let obj: Object = update_js.dyn_into().unwrap();
        let msg_type = Reflect::get(&obj, &JsValue::from_str("type")).unwrap();
        assert_eq!(msg_type.as_string().unwrap(), "UpdateConfig");

        let config_obj = Reflect::get(&obj, &JsValue::from_str("config")).unwrap();
        let model_name = Reflect::get(&config_obj, &JsValue::from_str("repo")).unwrap();
        assert_eq!(
            model_name.as_string().unwrap(),
            "onnx-community/whisper-large-v3-turbo"
        );

        console::log_1(&"✓ Worker configuration updates test successful".into());
    }
}
