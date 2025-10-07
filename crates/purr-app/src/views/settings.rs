use crate::{
    components::{
        Card, Icon, IconType, NumberInput, Section, Select, SelectOption, Slider, Toggle,
    },
    main_app::Route,
    platform,
    views::NavBarState,
};
use dioxus::prelude::*;
use purr_common::platform::{ModelInfo, Platform};
use purr_common::settings::{
    AudioFormat, AudioQuality, LogLevel, SampleRate, Settings as SettingsConfig, Theme,
};

/// The Settings page component with comprehensive configuration options
#[component]
pub fn Settings() -> Element {
    // Settings state
    let mut settings = use_signal(SettingsConfig::default);
    let mut has_changes = use_signal(|| false);
    let mut save_status: Signal<Option<SaveStatus>> = use_signal(|| None);
    let mut is_loading = use_signal(|| true);
    let mut models = use_signal(|| Ok(Vec::<ModelInfo>::new()));
    let mut navbar_state = use_context::<NavBarState>();
    use_effect(move || {
        navbar_state.title.set("Settings".to_string());
        navbar_state.show_back.set(true);
        navbar_state.top_right.set(Some(
            rsx! {
                Link {
                    to: Route::Logs {},
                    class: "p-2 rounded-lg text-gray-600 hover:text-teal-600 hover:bg-gray-100 transition-colors",
                    title: "Logs",
                    onclick: move |_| {
                        tracing::info!("Navigating to Logs page");
                    },
                    Icon { icon_type: IconType::Logs }
                }
            }
        ));
    });

    // Load models list
    use_future(move || async move {
        models.set(platform::get_platform().list_installed_models().await);
    });

    // Load settings on mount
    use_future(move || async move {
        let platform = platform::get_platform();
        match platform.load_settings().await {
            Ok(loaded_settings) => {
                settings.set(loaded_settings);
                save_status.set(None);
            }
            Err(err) => {
                tracing::warn!("Failed to load settings, using defaults: {}", err);
                save_status.set(Some(SaveStatus::LoadError(format!(
                    "Failed to load settings: {}",
                    err
                ))));
            }
        }
        is_loading.set(false);
    });

    // Mark as changed when any setting is modified
    let mut mark_changed = move || {
        has_changes.set(true);
        save_status.set(None);
    };

    // Save settings function
    let save_settings = move |_| {
        let settings_copy = settings.read().clone();
        save_status.set(Some(SaveStatus::Saving));

        spawn(async move {
            let platform = platform::get_platform();
            match platform.save_settings(&settings_copy).await {
                Ok(()) => {
                    save_status.set(Some(SaveStatus::Success));
                    has_changes.set(false);
                    tracing::info!("Settings saved successfully");
                }
                Err(err) => {
                    tracing::error!("Failed to save settings: {}", err);
                    save_status.set(Some(SaveStatus::SaveError(format!(
                        "Failed to save settings: {}",
                        err
                    ))));
                }
            }
        });
    };

    // Reset settings function
    let reset_settings = move |_| {
        settings.set(SettingsConfig::default());
        has_changes.set(true); // Mark as changed so user can save the reset
        save_status.set(Some(SaveStatus::ResetToDefaults));
    };

    rsx! {
        div { class: "main-content",
            div { class: "content-container",
                // Page header
                div { class: "mb-8 ml-8",
                    h1 { class: "text-3xl font-bold text-gradient mb-2", "Settings" }
                    p { class: "text-gray-600", "Configure your Purr transcription experience" }
                }

                // Settings form
                Card { class: "max-w-4xl mx-auto",
                    form { class: "space-y-8",
                        // UI Settings Section
                        Section {
                            title: "User Interface",
                            description: "Customize the appearance and behavior of the interface",

                            div { class: "grid grid-cols-1 md:grid-cols-2 gap-6",
                                Select {
                                    label: "Theme",
                                    description: "Choose your preferred color scheme",
                                    value: settings.read().ui.theme.to_string().to_string(),
                                    options: vec![
                                        SelectOption {
                                            value: "light".to_string(),
                                            label: "Light".to_string(),
                                        },
                                        SelectOption {
                                            value: "dark".to_string(),
                                            label: "Dark".to_string(),
                                        },
                                        SelectOption {
                                            value: "system".to_string(),
                                            label: "System".to_string(),
                                        },
                                    ],
                                    onchange: move |value: String| {
                                        if let Some(theme) = Theme::from_string(&value) {
                                            settings.with_mut(|s| s.ui.theme = theme);
                                            mark_changed();
                                        }
                                    },
                                }

                                Select {
                                    label: "Language",
                                    description: "Interface language",
                                    value: settings.read().ui.language.clone(),
                                    options: purr_common::Language::VARIANTS
                                        .iter()
                                        .map(|lang| SelectOption {
                                            value: lang.code().to_string(),
                                            label: lang.name().to_string(),
                                        })
                                        .collect(),
                                    onchange: move |value: String| {
                                        settings.with_mut(|s| s.ui.language = value);
                                        mark_changed();
                                    },
                                }
                            }
                        }

                        // Model Settings Section
                        Section {
                            title: "Model Settings",
                            description: "Configure the AI model and its behavior",

                            div { class: "grid grid-cols-1 md:grid-cols-2 gap-6",
                                match &*models.read() {
                                    Ok(model_list) => {
                                        rsx! {
                                            Select {
                                                label: "Model",
                                                description: "Choose the transcription model to use",
                                                value: settings.read().model.model_name.clone(),
                                                options: model_list
                                                    .iter()
                                                    .map(|model| SelectOption {
                                                        value: model.name.clone(),
                                                        label: format!("{} ({})", model.name, ByteSize(model.size_bytes)),
                                                    })
                                                    .collect(),
                                                onchange: move |value: String| {
                                                    settings.with_mut(|s| s.model.model_name = value);
                                                    mark_changed();
                                                },
                                            }
                                        }
                                    }
                                    Err(err) => rsx! {
                                        div { class: "text-red-600", "Error loading models: {err}" }
                                    },
                                }

                                NumberInput {
                                    label: "Max Tokens",
                                    description: "Maximum number of tokens to generate",
                                    value: settings.read().model.max_tokens,
                                    min: 256,
                                    max: 8192,
                                    onchange: move |value| {
                                        settings.with_mut(|s| s.model.max_tokens = value);
                                        mark_changed();
                                    },
                                }
                            }

                            Slider {
                                label: "Temperature",
                                description: "Controls randomness in output (0.0 = deterministic, 1.0 = creative)",
                                value: settings.read().model.temperature,
                                min: 0.0,
                                max: 1.0,
                                step: 0.01,
                                onchange: move |value| {
                                    settings.with_mut(|s| s.model.temperature = value);
                                    mark_changed();
                                },
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
                                    value: settings.read().audio.sample_rate.to_hz_string().to_string(),
                                    options: vec![
                                        SelectOption {
                                            value: "8000".to_string(),
                                            label: "8 kHz".to_string(),
                                        },
                                        SelectOption {
                                            value: "16000".to_string(),
                                            label: "16 kHz".to_string(),
                                        },
                                        SelectOption {
                                            value: "22050".to_string(),
                                            label: "22.05 kHz".to_string(),
                                        },
                                        SelectOption {
                                            value: "44100".to_string(),
                                            label: "44.1 kHz".to_string(),
                                        },
                                        SelectOption {
                                            value: "48000".to_string(),
                                            label: "48 kHz".to_string(),
                                        },
                                    ],
                                    onchange: move |value: String| {
                                        if let Some(sample_rate) = SampleRate::from_hz_string(&value) {
                                            settings.with_mut(|s| s.audio.sample_rate = sample_rate);
                                            mark_changed();
                                        }
                                    },
                                }

                                Select {
                                    label: "Audio Quality",
                                    description: "Processing quality level",
                                    value: settings.read().audio.quality.to_string().to_string(),
                                    options: vec![
                                        SelectOption {
                                            value: "low".to_string(),
                                            label: "Low (Fast)".to_string(),
                                        },
                                        SelectOption {
                                            value: "medium".to_string(),
                                            label: "Medium".to_string(),
                                        },
                                        SelectOption {
                                            value: "high".to_string(),
                                            label: "High (Slow)".to_string(),
                                        },
                                        SelectOption {
                                            value: "ultra".to_string(),
                                            label: "Ultra (Slowest)".to_string(),
                                        },
                                    ],
                                    onchange: move |value: String| {
                                        if let Some(quality) = AudioQuality::from_string(&value) {
                                            settings.with_mut(|s| s.audio.quality = quality);
                                            mark_changed();
                                        }
                                    },
                                }

                                Select {
                                    label: "File Format",
                                    description: "Preferred audio file format",
                                    value: settings.read().audio.file_format.to_extension().to_string(),
                                    options: vec![
                                        SelectOption {
                                            value: "mp3".to_string(),
                                            label: "MP3".to_string(),
                                        },
                                        SelectOption {
                                            value: "wav".to_string(),
                                            label: "WAV".to_string(),
                                        },
                                        SelectOption {
                                            value: "flac".to_string(),
                                            label: "FLAC".to_string(),
                                        },
                                        SelectOption {
                                            value: "ogg".to_string(),
                                            label: "OGG".to_string(),
                                        },
                                    ],
                                    onchange: move |value: String| {
                                        if let Some(format) = AudioFormat::from_extension(&value) {
                                            settings.with_mut(|s| s.audio.file_format = format);
                                            mark_changed();
                                        }
                                    },
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
                                    checked: settings.read().performance.gpu_acceleration,
                                    onchange: move |value| {
                                        settings.with_mut(|s| s.performance.gpu_acceleration = value);
                                        mark_changed();
                                    },
                                }

                                Toggle {
                                    label: "Concurrent Processing",
                                    description: "Process multiple audio files simultaneously",
                                    checked: settings.read().performance.concurrent_processing,
                                    onchange: move |value| {
                                        settings.with_mut(|s| s.performance.concurrent_processing = value);
                                        mark_changed();
                                    },
                                }

                                if settings.read().performance.concurrent_processing {
                                    NumberInput {
                                        label: "Max Concurrent Jobs",
                                        description: "Maximum number of files to process simultaneously",
                                        value: settings.read().performance.max_concurrent_jobs,
                                        min: 1,
                                        max: 8,
                                        onchange: move |value| {
                                            settings.with_mut(|s| s.performance.max_concurrent_jobs = value);
                                            mark_changed();
                                        },
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
                                    value: settings.read().logging.level.to_string().to_string(),
                                    options: vec![
                                        SelectOption {
                                            value: "error".to_string(),
                                            label: "Error".to_string(),
                                        },
                                        SelectOption {
                                            value: "warn".to_string(),
                                            label: "Warning".to_string(),
                                        },
                                        SelectOption {
                                            value: "info".to_string(),
                                            label: "Info".to_string(),
                                        },
                                        SelectOption {
                                            value: "debug".to_string(),
                                            label: "Debug".to_string(),
                                        },
                                        SelectOption {
                                            value: "trace".to_string(),
                                            label: "Trace".to_string(),
                                        },
                                    ],
                                    onchange: move |value: String| {
                                        if let Some(level) = LogLevel::from_string(&value) {
                                            settings.with_mut(|s| s.logging.level = level);
                                            mark_changed();
                                        }
                                    },
                                }

                                Toggle {
                                    label: "File Logging",
                                    description: "Save logs to file for debugging",
                                    checked: settings.read().logging.enable_file_logging,
                                    onchange: move |value| {
                                        settings.with_mut(|s| s.logging.enable_file_logging = value);
                                        mark_changed();
                                    },
                                }
                            }
                        }

                        // Action buttons
                        div { class: "flex items-center justify-between pt-6 border-t border-gray-200",
                            div { class: "flex items-center space-x-4",
                                match &*save_status.read() {
                                    None => rsx! {},
                                    Some(status) => rsx! {
                                        span { class: "{status.text_color()} text-sm", "{status}" }
                                    },
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
                                        match (&*save_status.read(), *has_changes.read()) {
                                            (Some(SaveStatus::Saving), _) => "bg-teal-400 text-white cursor-wait",
                                            (_, true) => "bg-teal-600 text-white hover:bg-teal-700",
                                            (_, false) => "bg-gray-300 text-gray-500 cursor-not-allowed",
                                        },
                                    ),
                                    disabled: !*has_changes.read() || matches!(*save_status.read(), Some(SaveStatus::Saving)),
                                    onclick: save_settings,
                                    match *save_status.read() {
                                        Some(SaveStatus::Saving) => "Saving...",
                                        _ => "Save Changes",
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum SaveStatus {
    Saving,
    Success,
    ResetToDefaults,
    LoadError(String),
    SaveError(String),
}

impl std::fmt::Display for SaveStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SaveStatus::Saving => write!(f, "Saving settings..."),
            SaveStatus::Success => write!(f, "Settings saved successfully!"),
            SaveStatus::ResetToDefaults => write!(
                f,
                "Settings reset to defaults. Click Save to persist changes."
            ),
            SaveStatus::LoadError(err) => write!(f, "Failed to load settings: {}", err),
            SaveStatus::SaveError(err) => write!(f, "Failed to save settings: {}", err),
        }
    }
}

impl SaveStatus {
    pub const fn text_color(&self) -> &'static str {
        match self {
            SaveStatus::Saving => "text-blue-600",
            SaveStatus::Success => "text-green-600",
            SaveStatus::ResetToDefaults => "text-orange-600",
            SaveStatus::LoadError(_) => "text-red-600",
            SaveStatus::SaveError(_) => "text-red-600",
        }
    }
}

pub struct ByteSize(pub u64);

impl std::fmt::Display for ByteSize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
        let mut size = self.0 as f64;
        let mut unit = 0;

        while size >= 1024.0 && unit < UNITS.len() - 1 {
            size /= 1024.0;
            unit += 1;
        }

        write!(f, "{:.1} {}", size, UNITS[unit])
    }
}
