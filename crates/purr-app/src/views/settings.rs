use crate::{
    components::{
        Card, Icon, IconType, NumberInput, Section, Select, SelectOption, Slider, Toggle,
        model_picker::ModelPicker,
    },
    hooks::{
        use_models::{GpuSupport, use_models},
        use_theme::use_theme,
    },
    main_app::Route,
    platform::{self, IS_WEB, logging::apply_log_level},
    utils::language_label::capitalized,
    views::NavBarState,
};
use dioxus::prelude::*;
use purr_common::platform::{Platform as _, model_flag};
use purr_common::settings::{
    AudioFormat, AudioQuality, LogLevel, SampleRate, Settings as SettingsConfig, Theme,
};

/// The Settings page component with comprehensive configuration options
#[component]
pub fn Settings() -> Element {
    let mut settings = use_signal(SettingsConfig::default);
    let mut has_changes = use_signal(|| false);
    let mut save_status: Signal<Option<SaveStatus>> = use_signal(|| None);
    let mut is_loading = use_signal(|| true);
    let mut navbar_state = use_context::<NavBarState>();
    let theme = use_theme();
    let models = use_models();

    use_effect(move || {
        navbar_state.title.set("Settings".to_owned());
        navbar_state.show_back.set(true);
        navbar_state.top_right.set(Some(rsx! {
            Link {
                to: Route::Logs {},
                class: "p-2 rounded-lg text-gray-600 dark:text-gray-300 hover:text-teal-600 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors",
                title: "Logs",
                aria_label: "Logs",
                Icon { icon_type: IconType::Logs }
            }
        }));
    });

    use_future(move || async move {
        match platform::get_platform().load_settings().await {
            Ok(loaded) => {
                settings.set(loaded);
                save_status.set(None);
            }
            Err(err) => {
                tracing::warn!("Failed to load settings, showing defaults: {}", err);
                save_status.set(Some(SaveStatus::LoadError(err.to_string())));
            }
        }
        is_loading.set(false);
    });

    let mut mark_changed = move || {
        has_changes.set(true);
        save_status.set(None);
    };

    let save_settings = move |_| {
        let settings_copy = settings.read().clone();
        save_status.set(Some(SaveStatus::Saving));
        spawn(async move {
            match platform::get_platform().save_settings(&settings_copy).await {
                Ok(()) => {
                    save_status.set(Some(SaveStatus::Success));
                    has_changes.set(false);
                }
                Err(err) => {
                    tracing::error!("Failed to save settings: {}", err);
                    save_status.set(Some(SaveStatus::SaveError(err.to_string())));
                }
            }
        });
    };

    let reset_settings = move |_| {
        let defaults = SettingsConfig::default();
        theme.set_theme.call(defaults.ui.theme);
        if let Err(error) = apply_log_level(&defaults.logging.level) {
            tracing::warn!("Could not apply the default log level: {error}");
        }
        settings.set(defaults);
        has_changes.set(true);
        save_status.set(Some(SaveStatus::ResetToDefaults));
    };

    let selected_model = settings.read().model.model_name.clone();
    let english_only = models
        .find(&selected_model)
        .is_some_and(|model| model.metadata.has_flag(model_flag::ENGLISH_ONLY));
    let gpu = models.gpu.read().clone();
    let status_line = save_status.read().as_ref().map_or_else(
        || {
            rsx! {
                if has_changes() {
                    span { class: "text-sm text-orange-600 dark:text-orange-400", "You have unsaved changes" }
                }
            }
        },
        |status| {
            rsx! {
                span { class: "{status.text_color()} text-sm", "{status}" }
            }
        },
    );
    let language_hint = if english_only {
        "The selected model only transcribes English"
    } else {
        "Language spoken in the audio, or detect it automatically"
    };
    let language_options: Vec<SelectOption> = purr_common::Language::VARIANTS
        .iter()
        .map(|lang| {
            let label = if lang.code() == "auto" {
                "Detect automatically".to_owned()
            } else {
                capitalized(lang.name())
            };
            SelectOption::new(lang.code(), label)
        })
        .collect();

    rsx! {
        document::Title { "Purr - Settings" }
        div { class: "main-content",
            div { class: "content-container",
                div { class: "mb-8",
                    h1 { class: "text-3xl font-bold text-gradient mb-2", "Settings" }
                    p { class: "text-gray-600 dark:text-gray-300", "Configure your Purr transcription experience" }
                }

                if is_loading() {
                    p { class: "text-gray-600 dark:text-gray-300", role: "status", "Loading settings..." }
                }

                Card { class: "max-w-4xl mx-auto",
                    form {
                        class: "space-y-8",
                        onsubmit: move |event| event.prevent_default(),

                        if IS_WEB {
                            p { class: "text-sm text-gray-600 dark:text-gray-300",
                                "Settings are stored in this browser. Options that only apply to the desktop app are not shown."
                            }
                            match gpu {
                                GpuSupport::Checking => rsx! {},
                                GpuSupport::Unavailable => rsx! {
                                    p { class: "text-sm text-red-600 dark:text-red-400", role: "alert",
                                        "WebGPU is not available in this browser, so Purr cannot transcribe here. Use a current Chrome, Edge or another browser with WebGPU enabled."
                                    }
                                },
                                GpuSupport::Available { name, shader_f16 } => rsx! {
                                    p { class: "text-sm text-gray-600 dark:text-gray-300",
                                        "GPU: {name}"
                                        if !shader_f16 {
                                            " (without shader-f16: the Large v3 Turbo model is unavailable)"
                                        }
                                    }
                                },
                            }
                        }

                        Section {
                            title: "User Interface",
                            description: "Customize the appearance and the language of the transcript",

                            div { class: "grid grid-cols-1 md:grid-cols-2 gap-6",
                                Select {
                                    label: "Theme",
                                    description: "Choose your preferred color scheme",
                                    value: theme.theme.read().to_storage_string().to_owned(),
                                    options: vec![
                                        SelectOption::new("light", "Light"),
                                        SelectOption::new("dark", "Dark"),
                                        SelectOption::new("system", "System"),
                                    ],
                                    onchange: move |value: String| {
                                        let chosen = Theme::from_storage_string(&value);
                                        theme.set_theme.call(chosen);
                                        settings.with_mut(|s| s.ui.theme = chosen);
                                    },
                                }

                                Select {
                                    label: "Language",
                                    description: language_hint,
                                    value: settings.read().ui.language.clone(),
                                    disabled: english_only,
                                    options: language_options,
                                    onchange: move |value: String| {
                                        settings.with_mut(|s| s.ui.language = value);
                                        mark_changed();
                                    },
                                }
                            }
                        }

                        Section {
                            title: "Model Settings",
                            description: "Choose the transcription model",

                            div { class: "space-y-4",
                                ModelPicker {
                                    selected: selected_model,
                                    models,
                                    onselect: move |value: String| {
                                        settings.with_mut(|s| s.model.model_name = value);
                                        mark_changed();
                                    },
                                }

                                if !IS_WEB {
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
                            }
                        }

                        if !IS_WEB {
                            Section {
                                title: "Audio Settings",
                                description: "Configure audio processing and quality",

                                div { class: "grid grid-cols-1 md:grid-cols-3 gap-6",
                                    Select {
                                        label: "Sample Rate",
                                        description: "Audio sample rate in Hz",
                                        value: settings.read().audio.sample_rate.to_hz_string().to_owned(),
                                        options: vec![
                                            SelectOption::new("8000", "8 kHz"),
                                            SelectOption::new("16000", "16 kHz"),
                                            SelectOption::new("22050", "22.05 kHz"),
                                            SelectOption::new("44100", "44.1 kHz"),
                                            SelectOption::new("48000", "48 kHz"),
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
                                        value: settings.read().audio.quality.to_string().to_owned(),
                                        options: vec![
                                            SelectOption::new("low", "Low (Fast)"),
                                            SelectOption::new("medium", "Medium"),
                                            SelectOption::new("high", "High (Slow)"),
                                            SelectOption::new("ultra", "Ultra (Slowest)"),
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
                                        value: settings.read().audio.file_format.to_extension().to_owned(),
                                        options: vec![
                                            SelectOption::new("mp3", "MP3"),
                                            SelectOption::new("wav", "WAV"),
                                            SelectOption::new("flac", "FLAC"),
                                            SelectOption::new("ogg", "OGG"),
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
                        }

                        Section {
                            title: "Logging",
                            description: "Configure application logging and debugging",

                            div { class: "space-y-4",
                                Select {
                                    label: "Log Level",
                                    description: "Minimum level of messages shown in the Logs page and the console",
                                    value: settings.read().logging.level.to_string().to_owned(),
                                    options: vec![
                                        SelectOption::new("error", "Error"),
                                        SelectOption::new("warn", "Warning"),
                                        SelectOption::new("info", "Info"),
                                        SelectOption::new("debug", "Debug"),
                                        SelectOption::new("trace", "Trace"),
                                    ],
                                    onchange: move |value: String| {
                                        if let Some(level) = LogLevel::from_string(&value) {
                                            if let Err(error) = apply_log_level(&level) {
                                                tracing::warn!("Could not apply the log level: {error}");
                                            }
                                            settings.with_mut(|s| s.logging.level = level);
                                            mark_changed();
                                        }
                                    },
                                }

                                if !IS_WEB {
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
                        }

                        div { class: "flex flex-wrap items-center justify-between gap-4 pt-6 border-t border-gray-200 dark:border-gray-700",
                            div { class: "flex items-center space-x-4", role: "status",
                                {status_line}
                            }

                            div { class: "flex items-center space-x-3",
                                button {
                                    r#type: "button",
                                    class: "px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm text-sm font-medium text-gray-700 dark:text-gray-200 bg-white dark:bg-gray-800 hover:bg-gray-50 dark:hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-teal-500",
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
                                            (_, false) => "bg-gray-300 text-gray-500 cursor-not-allowed dark:bg-gray-700 dark:text-gray-400",
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

#[derive(Debug, Clone, PartialEq, Eq)]
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
            Self::Saving => write!(f, "Saving settings..."),
            Self::Success => write!(f, "Settings saved successfully!"),
            Self::ResetToDefaults => write!(
                f,
                "Settings reset to defaults. Click Save to persist changes."
            ),
            Self::LoadError(err) => write!(f, "Failed to load settings: {err}"),
            Self::SaveError(err) => write!(f, "Failed to save settings: {err}"),
        }
    }
}

impl SaveStatus {
    pub const fn text_color(&self) -> &'static str {
        match self {
            Self::Saving => "text-blue-600 dark:text-blue-400",
            Self::Success => "text-green-600 dark:text-green-400",
            Self::ResetToDefaults => "text-orange-600 dark:text-orange-400",
            Self::LoadError(_) | Self::SaveError(_) => "text-red-600 dark:text-red-400",
        }
    }
}
