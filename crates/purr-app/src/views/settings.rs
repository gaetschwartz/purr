use crate::{
    components::{
        banner::{Banner, BannerTone},
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
        toast::{Toast, ToastKind, use_toasts},
    },
    hooks::use_models::{GpuSupport, use_models},
    main_app::Route,
    platform::IS_WEB,
    settings_store::{
        status::{SaveStatus, UpdateError},
        use_settings,
    },
    utils::language_label::capitalized,
    views::about::AboutSheet,
};
use dioxus::prelude::*;
use purr_common::platform::model_flag;
use purr_common::settings::{AudioFormat, AudioQuality, LogLevel, SampleRate, Theme};

/// The Settings page component with comprehensive configuration options
#[component]
pub fn Settings() -> Element {
    let settings = use_settings();
    let toasts = use_toasts();
    let mut about_open = use_signal(|| false);
    let mut reset_open = use_signal(|| false);
    let models = use_models();

    let report = move |result: Result<(), UpdateError>| {
        if let Err(error) = result {
            toasts.error("That setting can't be used", error.to_string());
        }
    };

    let reset_settings = move |()| {
        reset_open.set(false);
        let previous = settings.reset_to_defaults();
        toasts.show(
            Toast::new(ToastKind::Info, "Settings reset").with_action("Undo", move || {
                if let Err(error) = settings.replace(previous.clone()) {
                    toasts.error("Couldn't undo the reset", error.to_string());
                }
            }),
        );
    };

    let current = settings.get();
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
    let status = save_status_text(&settings.status());

    rsx! {
        Page {
            title: "Settings",
            subtitle: "Changes are saved as you make them.",
            actions: rsx! {
                span { class: "foot", role: "status", {status} }
            },

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
                                value: current.ui.theme,
                                segments: vec![
                                    Segment::new(Theme::System, "Auto").with_icon(IconName::Auto),
                                    Segment::new(Theme::Light, "Light").with_icon(IconName::Sun),
                                    Segment::new(Theme::Dark, "Dark").with_icon(IconName::Moon),
                                ],
                                onchange: move |theme| report(settings.set_theme(theme)),
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
                                checked: current.ui.reduce_transparency,
                                onchange: move |reduce| report(settings.set_reduce_transparency(reduce)),
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
                                onchange: move |value: String| report(settings.set_language(value)),
                            }
                        },
                    }
                    div { class: "item col",
                        ModelPicker {
                            selected: selected_model,
                            models,
                            onselect: move |value: String| report(settings.set_model(value)),
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
                                    onchange: move |value| report(settings.update(|s| s.model.max_tokens = value)),
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
                                onchange: move |value| report(settings.update(|s| s.model.temperature = value)),
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
                                    onchange: move |rate| report(settings.update(|s| s.audio.sample_rate = rate)),
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
                                    onchange: move |quality| report(settings.update(|s| s.audio.quality = quality)),
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
                                    onchange: move |format| report(settings.update(|s| s.audio.file_format = format)),
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
                                    onchange: move |value| report(settings.update(|s| s.performance.gpu_acceleration = value)),
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
                                    onchange: move |value| report(settings.update(|s| s.performance.concurrent_processing = value)),
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
                                        onchange: move |value| report(settings.update(|s| s.performance.max_concurrent_jobs = value)),
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
                                onchange: move |level: LogLevel| report(settings.set_log_level(level)),
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
                                    onchange: move |value| report(settings.update(|s| s.logging.enable_file_logging = value)),
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
                "Theme, language, model choice, performance and logging go back to their defaults. Downloaded models stay."
            }
        }
    }
}

const fn save_status_text(status: &SaveStatus) -> &'static str {
    match status {
        SaveStatus::Idle => "",
        SaveStatus::Saving => "Saving…",
        SaveStatus::Saved { .. } => "Saved",
        SaveStatus::Failed { .. } => "Not saved",
        SaveStatus::Paused => "Not saved until the settings are reset",
    }
}
