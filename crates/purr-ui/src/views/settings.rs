use crate::components::{Card, NumberInput, Section, Select, SelectOption, Slider, Toggle};
use dioxus::prelude::*;

/// Application settings state structure
#[derive(Clone, PartialEq)]
pub struct AppSettings {
    // Model Settings
    pub model_name: String,
    pub temperature: f64,
    pub max_tokens: u32,

    // Audio Settings
    pub sample_rate: String,
    pub audio_quality: String,
    pub file_format: String,

    // UI Settings
    pub theme: String,
    pub compact_mode: bool,
    pub language: String,

    // Performance Settings
    pub gpu_acceleration: bool,
    pub concurrent_processing: bool,
    pub max_concurrent_jobs: u32,

    // Logging Settings
    pub log_level: String,
    pub enable_file_logging: bool,
}

impl Default for AppSettings {
    fn default() -> Self {
        Self {
            // Model Settings
            model_name: "whisper-1".to_string(),
            temperature: 0.0,
            max_tokens: 4096,

            // Audio Settings
            sample_rate: "16000".to_string(),
            audio_quality: "high".to_string(),
            file_format: "mp3".to_string(),

            // UI Settings
            theme: "light".to_string(),
            compact_mode: false,
            language: "en".to_string(),

            // Performance Settings
            gpu_acceleration: false,
            concurrent_processing: true,
            max_concurrent_jobs: 2,

            // Logging Settings
            log_level: "info".to_string(),
            enable_file_logging: true,
        }
    }
}

/// The Settings page component with comprehensive configuration options
#[component]
pub fn Settings() -> Element {
    // Settings state
    let mut settings = use_signal(AppSettings::default);
    let mut has_changes = use_signal(|| false);
    let mut save_status = use_signal(|| "");

    // Mark as changed when any setting is modified
    let mut mark_changed = move || {
        has_changes.set(true);
        save_status.set("");
    };

    // Save settings function
    let save_settings = move |_| {
        // TODO: Implement actual persistence (localStorage, file, etc.)
        save_status.set("Settings saved successfully!");
        has_changes.set(false);
    };

    // Reset settings function
    let reset_settings = move |_| {
        settings.set(AppSettings::default());
        has_changes.set(false);
        save_status.set("Settings reset to defaults");
    };

    rsx! {
        div { class: "main-content",
            div { class: "content-container",
                // Page header
                div { class: "mb-8",
                    h1 { class: "text-3xl font-bold text-gradient mb-2", "Settings" }
                    p { class: "text-gray-600", "Configure your Purr transcription experience" }
                }

                // Settings form
                Card { class: "max-w-4xl mx-auto",
                    form { class: "space-y-8",
                        // Model Settings Section
                        Section {
                            title: "Model Settings",
                            description: "Configure the AI model and its behavior",

                            div { class: "grid grid-cols-1 md:grid-cols-2 gap-6",
                                Select {
                                    label: "Model",
                                    description: "Choose the transcription model to use",
                                    value: settings.read().model_name.clone(),
                                    options: vec![
                                        SelectOption { value: "whisper-1".to_string(), label: "Whisper v1".to_string() },
                                        SelectOption { value: "whisper-2".to_string(), label: "Whisper v2".to_string() },
                                        SelectOption { value: "whisper-large".to_string(), label: "Whisper Large".to_string() },
                                    ],
                                    onchange: move |value| {
                                        settings.with_mut(|s| s.model_name = value);
                                        mark_changed();
                                    }
                                }

                                NumberInput {
                                    label: "Max Tokens",
                                    description: "Maximum number of tokens to generate",
                                    value: settings.read().max_tokens,
                                    min: 256,
                                    max: 8192,
                                    onchange: move |value| {
                                        settings.with_mut(|s| s.max_tokens = value);
                                        mark_changed();
                                    }
                                }
                            }

                            Slider {
                                label: "Temperature",
                                description: "Controls randomness in output (0.0 = deterministic, 1.0 = creative)",
                                value: settings.read().temperature,
                                min: 0.0,
                                max: 1.0,
                                step: 0.01,
                                onchange: move |value| {
                                    settings.with_mut(|s| s.temperature = value);
                                    mark_changed();
                                }
                            }
                        }

                        // Audio Settings Section
                        Section {
                            title: "Audio Settings",
                            description: "Configure audio processing and quality",

                            div { class: "grid grid-cols-1 md:grid-cols-3 gap-6",
                                Select {
                                    label: "Sample Rate",
                                    description: "Audio sample rate in Hz",
                                    value: settings.read().sample_rate.clone(),
                                    options: vec![
                                        SelectOption { value: "8000".to_string(), label: "8 kHz".to_string() },
                                        SelectOption { value: "16000".to_string(), label: "16 kHz".to_string() },
                                        SelectOption { value: "22050".to_string(), label: "22.05 kHz".to_string() },
                                        SelectOption { value: "44100".to_string(), label: "44.1 kHz".to_string() },
                                        SelectOption { value: "48000".to_string(), label: "48 kHz".to_string() },
                                    ],
                                    onchange: move |value| {
                                        settings.with_mut(|s| s.sample_rate = value);
                                        mark_changed();
                                    }
                                }

                                Select {
                                    label: "Audio Quality",
                                    description: "Processing quality level",
                                    value: settings.read().audio_quality.clone(),
                                    options: vec![
                                        SelectOption { value: "low".to_string(), label: "Low (Fast)".to_string() },
                                        SelectOption { value: "medium".to_string(), label: "Medium".to_string() },
                                        SelectOption { value: "high".to_string(), label: "High (Slow)".to_string() },
                                        SelectOption { value: "ultra".to_string(), label: "Ultra (Slowest)".to_string() },
                                    ],
                                    onchange: move |value| {
                                        settings.with_mut(|s| s.audio_quality = value);
                                        mark_changed();
                                    }
                                }

                                Select {
                                    label: "File Format",
                                    description: "Preferred audio file format",
                                    value: settings.read().file_format.clone(),
                                    options: vec![
                                        SelectOption { value: "mp3".to_string(), label: "MP3".to_string() },
                                        SelectOption { value: "wav".to_string(), label: "WAV".to_string() },
                                        SelectOption { value: "flac".to_string(), label: "FLAC".to_string() },
                                        SelectOption { value: "ogg".to_string(), label: "OGG".to_string() },
                                    ],
                                    onchange: move |value| {
                                        settings.with_mut(|s| s.file_format = value);
                                        mark_changed();
                                    }
                                }
                            }
                        }

                        // UI Settings Section
                        Section {
                            title: "User Interface",
                            description: "Customize the appearance and behavior of the interface",

                            div { class: "grid grid-cols-1 md:grid-cols-2 gap-6",
                                Select {
                                    label: "Theme",
                                    description: "Choose your preferred color scheme",
                                    value: settings.read().theme.clone(),
                                    options: vec![
                                        SelectOption { value: "light".to_string(), label: "Light".to_string() },
                                        SelectOption { value: "dark".to_string(), label: "Dark".to_string() },
                                        SelectOption { value: "system".to_string(), label: "System".to_string() },
                                    ],
                                    onchange: move |value| {
                                        settings.with_mut(|s| s.theme = value);
                                        mark_changed();
                                    }
                                }

                                Select {
                                    label: "Language",
                                    description: "Interface language",
                                    value: settings.read().language.clone(),
                                    options: vec![
                                        SelectOption { value: "en".to_string(), label: "English".to_string() },
                                        SelectOption { value: "es".to_string(), label: "Spanish".to_string() },
                                        SelectOption { value: "fr".to_string(), label: "French".to_string() },
                                        SelectOption { value: "de".to_string(), label: "German".to_string() },
                                        SelectOption { value: "zh".to_string(), label: "Chinese".to_string() },
                                    ],
                                    onchange: move |value| {
                                        settings.with_mut(|s| s.language = value);
                                        mark_changed();
                                    }
                                }
                            }

                            Toggle {
                                label: "Compact Mode",
                                description: "Use a more condensed interface layout",
                                checked: settings.read().compact_mode,
                                onchange: move |value| {
                                    settings.with_mut(|s| s.compact_mode = value);
                                    mark_changed();
                                }
                            }
                        }

                        // Performance Settings Section
                        Section {
                            title: "Performance",
                            description: "Optimize performance for your system",

                            div { class: "space-y-4",
                                Toggle {
                                    label: "GPU Acceleration",
                                    description: "Use GPU for faster transcription (requires compatible hardware)",
                                    checked: settings.read().gpu_acceleration,
                                    onchange: move |value| {
                                        settings.with_mut(|s| s.gpu_acceleration = value);
                                        mark_changed();
                                    }
                                }

                                Toggle {
                                    label: "Concurrent Processing",
                                    description: "Process multiple audio files simultaneously",
                                    checked: settings.read().concurrent_processing,
                                    onchange: move |value| {
                                        settings.with_mut(|s| s.concurrent_processing = value);
                                        mark_changed();
                                    }
                                }

                                if settings.read().concurrent_processing {
                                    NumberInput {
                                        label: "Max Concurrent Jobs",
                                        description: "Maximum number of files to process simultaneously",
                                        value: settings.read().max_concurrent_jobs,
                                        min: 1,
                                        max: 8,
                                        onchange: move |value| {
                                            settings.with_mut(|s| s.max_concurrent_jobs = value);
                                            mark_changed();
                                        }
                                    }
                                }
                            }
                        }

                        // Logging Settings Section
                        Section {
                            title: "Logging",
                            description: "Configure application logging and debugging",

                            div { class: "space-y-4",
                                Select {
                                    label: "Log Level",
                                    description: "Minimum level of messages to log",
                                    value: settings.read().log_level.clone(),
                                    options: vec![
                                        SelectOption { value: "error".to_string(), label: "Error".to_string() },
                                        SelectOption { value: "warn".to_string(), label: "Warning".to_string() },
                                        SelectOption { value: "info".to_string(), label: "Info".to_string() },
                                        SelectOption { value: "debug".to_string(), label: "Debug".to_string() },
                                        SelectOption { value: "trace".to_string(), label: "Trace".to_string() },
                                    ],
                                    onchange: move |value| {
                                        settings.with_mut(|s| s.log_level = value);
                                        mark_changed();
                                    }
                                }

                                Toggle {
                                    label: "File Logging",
                                    description: "Save logs to file for debugging",
                                    checked: settings.read().enable_file_logging,
                                    onchange: move |value| {
                                        settings.with_mut(|s| s.enable_file_logging = value);
                                        mark_changed();
                                    }
                                }
                            }
                        }

                        // Action buttons
                        div { class: "flex items-center justify-between pt-6 border-t border-gray-200",
                            div { class: "flex items-center space-x-4",
                                if !save_status.read().is_empty() {
                                    span {
                                        class: if save_status.read().contains("successfully") {
                                            "text-green-600 text-sm"
                                        } else {
                                            "text-blue-600 text-sm"
                                        },
                                        "{save_status.read()}"
                                    }
                                }
                            }

                            div { class: "flex items-center space-x-3",
                                button {
                                    r#type: "button",
                                    class: "px-4 py-2 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-teal-500",
                                    onclick: reset_settings,
                                    "Reset to Defaults"
                                }

                                button {
                                    r#type: "button",
                                    class: format!(
                                        "px-4 py-2 rounded-md shadow-sm text-sm font-medium focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-teal-500 {}",
                                        if *has_changes.read() {
                                            "bg-teal-600 text-white hover:bg-teal-700"
                                        } else {
                                            "bg-gray-300 text-gray-500 cursor-not-allowed"
                                        }
                                    ),
                                    disabled: !*has_changes.read(),
                                    onclick: save_settings,
                                    "Save Changes"
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}