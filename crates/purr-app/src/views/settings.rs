use crate::{
    appearance::{Appearance, use_appearance},
    components::{
        banner::{Banner, BannerTone},
        button::{Button, ButtonVariant},
        dialog::{Dialog, DialogTone},
        group::{Group, Item, ItemAction, TileColor},
        icon::IconName,
        model_picker::ModelPicker,
        page::Page,
        segmented::{Segment, Segmented},
        select::{Select, SelectOption},
        slider::Slider,
        stepper::Stepper,
        switch::Switch,
    },
    hooks::use_models::{GpuSupport, use_models},
    main_app::Route,
    platform::{self, IS_WEB, logging::apply_log_level},
    utils::language_label::capitalized,
    views::about::AboutSheet,
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
    let mut about_open = use_signal(|| false);
    let mut reset_open = use_signal(|| false);
    let appearance = use_appearance();
    let models = use_models();

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
        let mut settings_copy = settings.read().clone();
        let chosen = appearance.peek();
        settings_copy.ui.theme = chosen.theme;
        settings_copy.ui.reduce_transparency = chosen.reduce_transparency;
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

    let reset_settings = move |()| {
        reset_open.set(false);
        let defaults = SettingsConfig::default();
        appearance.set(Appearance {
            theme: defaults.ui.theme,
            reduce_transparency: defaults.ui.reduce_transparency,
        });
        if let Err(error) = apply_log_level(&defaults.logging.level) {
            tracing::warn!("Could not apply the default log level: {error}");
        }
        settings.set(defaults);
        has_changes.set(true);
        save_status.set(Some(SaveStatus::ResetToDefaults));
    };

    let chosen = appearance.get();
    let current = settings.read().clone();
    let selected_model = current.model.model_name.clone();
    let english_only = models
        .find(&selected_model)
        .is_some_and(|model| model.metadata.has_flag(model_flag::ENGLISH_ONLY));
    let gpu = models.gpu.read().clone();
    let language_hint = if english_only {
        "The selected model only transcribes English."
    } else {
        "Automatic works for most recordings."
    };
    let language_options: Vec<SelectOption<String>> = purr_common::Language::VARIANTS
        .iter()
        .map(|lang| {
            let label = if lang.code() == "auto" {
                "Automatic".to_owned()
            } else {
                capitalized(lang.name())
            };
            SelectOption::new(lang.code().to_owned(), label)
        })
        .collect();
    let saving = matches!(*save_status.read(), Some(SaveStatus::Saving));

    rsx! {
        Page {
            title: "Settings",
            subtitle: "Appearance changes apply and save at once. Press Save for the rest.",
            actions: rsx! {
                if let Some(status) = save_status() {
                    span { class: "foot", role: "status", "{status}" }
                } else if has_changes() {
                    span { class: "foot", role: "status", "Unsaved changes" }
                }
                Button {
                    variant: ButtonVariant::Primary,
                    icon: IconName::Check,
                    disabled: !has_changes() || saving,
                    onclick: save_settings,
                    if saving {
                        "Saving…"
                    } else {
                        "Save"
                    }
                }
            },

            if is_loading() {
                p { class: "muted", role: "status", "Loading settings…" }
            }

            if IS_WEB {
                Banner { tone: BannerTone::Info, icon: IconName::Window, title: "You're using Purr in the browser",
                    "Settings are stored in this browser. Options that only apply to the desktop app are not shown."
                }
                match gpu {
                    GpuSupport::Checking => rsx! {},
                    GpuSupport::Unavailable => rsx! {
                        Banner { tone: BannerTone::Bad, icon: IconName::Cpu, title: "WebGPU is not available", role: "alert",
                            "Purr cannot transcribe in this browser. Use a current Chrome, Edge or another browser with WebGPU enabled."
                        }
                    },
                    GpuSupport::Available { name, shader_f16 } => rsx! {
                        p { class: "foot", style: "margin:0",
                            "GPU: {name}"
                            if !shader_f16 {
                                " (without shader-f16: the Large v3 Turbo model is unavailable)"
                            }
                        }
                    },
                }
            }

            div { class: "settings",
                Group { header: "Appearance",
                    Item {
                        icon: IconName::Sun,
                        tile: TileColor::Blue,
                        title: "Theme",
                        stackable: true,
                        control: rsx! {
                            Segmented {
                                label: "Theme",
                                value: chosen.theme,
                                segments: vec![
                                    Segment::new(Theme::System, "Auto").with_icon(IconName::Auto),
                                    Segment::new(Theme::Light, "Light").with_icon(IconName::Sun),
                                    Segment::new(Theme::Dark, "Dark").with_icon(IconName::Moon),
                                ],
                                onchange: move |theme| appearance.set_theme(theme),
                            }
                        },
                    }
                    Item {
                        icon: IconName::Stack,
                        tile: TileColor::Gray,
                        title: "Reduce transparency",
                        subtitle: "Solid backgrounds instead of glass. Also follows the system setting.",
                        control: rsx! {
                            Switch {
                                label: "Reduce transparency",
                                checked: chosen.reduce_transparency,
                                onchange: move |reduce| appearance.set_reduce_transparency(reduce),
                            }
                        },
                    }
                }

                Group { header: "Transcription",
                    Item {
                        icon: IconName::Globe,
                        tile: TileColor::Indigo,
                        title: "Spoken language",
                        subtitle: language_hint,
                        control: rsx! {
                            Select {
                                label: "Spoken language",
                                value: current.ui.language.clone(),
                                disabled: english_only,
                                options: language_options,
                                onchange: move |value: String| {
                                    settings.with_mut(|s| s.ui.language = value);
                                    mark_changed();
                                },
                            }
                        },
                    }
                    div { class: "item col",
                        ModelPicker {
                            selected: selected_model,
                            models,
                            onselect: move |value: String| {
                                settings.with_mut(|s| s.model.model_name = value);
                                mark_changed();
                            },
                        }
                    }
                    if !IS_WEB {
                        Item {
                            icon: IconName::Hash,
                            tile: TileColor::Gray,
                            title: "Maximum tokens",
                            subtitle: "Maximum number of tokens to generate.",
                            control: rsx! {
                                Stepper {
                                    label: "Maximum tokens",
                                    value: current.model.max_tokens,
                                    min: 256,
                                    max: 8192,
                                    step: 256,
                                    onchange: move |value| {
                                        settings.with_mut(|s| s.model.max_tokens = value);
                                        mark_changed();
                                    },
                                }
                            },
                        }
                        Item {
                            icon: IconName::Thermometer,
                            tile: TileColor::Pink,
                            title: "Temperature",
                            subtitle: "Lower is more predictable. Raise it if the text repeats itself.",
                            value: format!("{:.2}", current.model.temperature),
                            Slider {
                                label: "Temperature",
                                value: current.model.temperature,
                                min: 0.0,
                                max: 1.0,
                                step: 0.01,
                                value_text: format!("{:.2}", current.model.temperature),
                                ends: ("0".to_owned(), "1".to_owned()),
                                onchange: move |value| {
                                    settings.with_mut(|s| s.model.temperature = value);
                                    mark_changed();
                                },
                            }
                        }
                    }
                }

                if !IS_WEB {
                    Group { header: "Audio",
                        Item {
                            icon: IconName::Speaker,
                            tile: TileColor::Red,
                            title: "Sample rate",
                            control: rsx! {
                                Select {
                                    label: "Sample rate",
                                    value: current.audio.sample_rate.clone(),
                                    options: vec![
                                        SelectOption::new(SampleRate::Hz8000, "8 kHz"),
                                        SelectOption::new(SampleRate::Hz16000, "16 kHz"),
                                        SelectOption::new(SampleRate::Hz22050, "22.05 kHz"),
                                        SelectOption::new(SampleRate::Hz44100, "44.1 kHz"),
                                        SelectOption::new(SampleRate::Hz48000, "48 kHz"),
                                    ],
                                    onchange: move |rate| {
                                        settings.with_mut(|s| s.audio.sample_rate = rate);
                                        mark_changed();
                                    },
                                }
                            },
                        }
                        Item {
                            icon: IconName::Wave,
                            tile: TileColor::Blue,
                            title: "Processing quality",
                            subtitle: "Higher is slower.",
                            stackable: true,
                            control: rsx! {
                                Segmented {
                                    label: "Processing quality",
                                    value: current.audio.quality.clone(),
                                    segments: vec![
                                        Segment::new(AudioQuality::Low, "Low"),
                                        Segment::new(AudioQuality::Medium, "Medium"),
                                        Segment::new(AudioQuality::High, "High"),
                                        Segment::new(AudioQuality::Ultra, "Ultra"),
                                    ],
                                    onchange: move |quality| {
                                        settings.with_mut(|s| s.audio.quality = quality);
                                        mark_changed();
                                    },
                                }
                            },
                        }
                        Item {
                            icon: IconName::Doc,
                            tile: TileColor::Teal,
                            title: "File format",
                            subtitle: "Preferred audio file format.",
                            control: rsx! {
                                Select {
                                    label: "File format",
                                    value: current.audio.file_format.clone(),
                                    options: vec![
                                        SelectOption::new(AudioFormat::Mp3, "MP3"),
                                        SelectOption::new(AudioFormat::Wav, "WAV"),
                                        SelectOption::new(AudioFormat::Flac, "FLAC"),
                                        SelectOption::new(AudioFormat::Ogg, "OGG"),
                                    ],
                                    onchange: move |format| {
                                        settings.with_mut(|s| s.audio.file_format = format);
                                        mark_changed();
                                    },
                                }
                            },
                        }
                    }

                    Group { header: "Performance",
                        Item {
                            icon: IconName::Bolt,
                            tile: TileColor::Orange,
                            title: "GPU acceleration",
                            subtitle: "Much faster on a supported graphics card.",
                            control: rsx! {
                                Switch {
                                    label: "GPU acceleration",
                                    checked: current.performance.gpu_acceleration,
                                    onchange: move |value| {
                                        settings.with_mut(|s| s.performance.gpu_acceleration = value);
                                        mark_changed();
                                    },
                                }
                            },
                        }
                        Item {
                            icon: IconName::Stack,
                            tile: TileColor::Purple,
                            title: "Process files concurrently",
                            subtitle: "Run more than one transcription at a time. Uses more memory.",
                            control: rsx! {
                                Switch {
                                    label: "Process files concurrently",
                                    checked: current.performance.concurrent_processing,
                                    onchange: move |value| {
                                        settings.with_mut(|s| s.performance.concurrent_processing = value);
                                        mark_changed();
                                    },
                                }
                            },
                        }
                        if current.performance.concurrent_processing {
                            Item {
                                icon: IconName::Hash,
                                tile: TileColor::Purple,
                                title: "Maximum concurrent jobs",
                                control: rsx! {
                                    Stepper {
                                        label: "Maximum concurrent jobs",
                                        value: current.performance.max_concurrent_jobs,
                                        min: 1,
                                        max: 8,
                                        onchange: move |value| {
                                            settings.with_mut(|s| s.performance.max_concurrent_jobs = value);
                                            mark_changed();
                                        },
                                    }
                                },
                            }
                        }
                    }
                }

                Group { header: "Logging",
                    Item {
                        icon: IconName::Logs,
                        tile: TileColor::Gray,
                        title: "Log level",
                        subtitle: "Debug and Trace help when reporting a problem.",
                        control: rsx! {
                            Select {
                                label: "Log level",
                                value: current.logging.level.clone(),
                                options: vec![
                                    SelectOption::new(LogLevel::Error, "Error"),
                                    SelectOption::new(LogLevel::Warn, "Warning"),
                                    SelectOption::new(LogLevel::Info, "Info"),
                                    SelectOption::new(LogLevel::Debug, "Debug"),
                                    SelectOption::new(LogLevel::Trace, "Trace"),
                                ],
                                onchange: move |level: LogLevel| {
                                    if let Err(error) = apply_log_level(&level) {
                                        tracing::warn!("Could not apply the log level: {error}");
                                    }
                                    settings.with_mut(|s| s.logging.level = level);
                                    mark_changed();
                                },
                            }
                        },
                    }
                    if !IS_WEB {
                        Item {
                            icon: IconName::File,
                            tile: TileColor::Blue,
                            title: "Write logs to a file",
                            control: rsx! {
                                Switch {
                                    label: "Write logs to a file",
                                    checked: current.logging.enable_file_logging,
                                    onchange: move |value| {
                                        settings.with_mut(|s| s.logging.enable_file_logging = value);
                                        mark_changed();
                                    },
                                }
                            },
                        }
                    }
                    Item {
                        icon: IconName::Logs,
                        tile: TileColor::Indigo,
                        title: "Open logs",
                        action: ItemAction::Navigate(Route::Logs {}),
                    }
                }

                Group { header: "About",
                    Item {
                        icon: IconName::Shield,
                        tile: TileColor::Green,
                        title: "Privacy",
                        subtitle: "What stays on this device, and the one thing that uses the network.",
                        action: ItemAction::Press(EventHandler::new(move |_| about_open.set(true))),
                    }
                    Item {
                        title: "Reset all settings",
                        subtitle: "Models are not affected.",
                        danger: true,
                        action: ItemAction::Press(EventHandler::new(move |_| reset_open.set(true))),
                    }
                }
            }
        }
        if about_open() {
            AboutSheet { on_close: move |()| about_open.set(false) }
        }
        if reset_open() {
            Dialog {
                title: "Reset all settings?",
                confirm_label: "Reset settings",
                tone: DialogTone::Destructive,
                on_confirm: reset_settings,
                on_cancel: move |()| reset_open.set(false),
                "Theme, language, model choice, performance and logging go back to their defaults. Press Save afterwards to keep them. Downloaded models stay."
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
            Self::Saving => write!(f, "Saving settings…"),
            Self::Success => write!(f, "Settings saved"),
            Self::ResetToDefaults => write!(f, "Defaults restored. Press Save to keep them."),
            Self::LoadError(err) => write!(f, "Failed to load settings: {err}"),
            Self::SaveError(err) => write!(f, "Failed to save settings: {err}"),
        }
    }
}
