use super::{changes::report, sections::SettingsSection};
use crate::{
    components::{
        group::{Group, Item, TileColor},
        icon::IconName,
        segmented::{Segment, Segmented},
        select::{Select, SelectOption},
        toast::use_toasts,
    },
    settings_store::use_settings,
};
use dioxus::prelude::*;
use purr_common::settings::{AudioFormat, AudioQuality, SampleRate};

/// Sample rate, processing quality and file format; desktop only.
#[component]
pub fn AudioSection() -> Element {
    let settings = use_settings();
    let toasts = use_toasts();
    let (sample_rate, quality, file_format) = settings.with(|settings| {
        (
            settings.audio.sample_rate.clone(),
            settings.audio.quality.clone(),
            settings.audio.file_format.clone(),
        )
    });

    rsx! {
        Group { id: SettingsSection::Audio.anchor(), header: SettingsSection::Audio.label(),
            Item {
                icon: IconName::Speaker,
                tile: TileColor::Red,
                title: "Sample rate",
                subtitle: "Whisper itself always listens at 16 kHz.",
                control: rsx! {
                    Select {
                        label: "Sample rate",
                        value: sample_rate,
                        options: vec![
                            SelectOption::new(SampleRate::Hz8000, "8 kHz"),
                            SelectOption::new(SampleRate::Hz16000, "16 kHz"),
                            SelectOption::new(SampleRate::Hz22050, "22.05 kHz"),
                            SelectOption::new(SampleRate::Hz44100, "44.1 kHz"),
                            SelectOption::new(SampleRate::Hz48000, "48 kHz"),
                        ],
                        onchange: move |rate| report(toasts, settings.update(|s| s.audio.sample_rate = rate)),
                    }
                },
            }
            Item {
                icon: IconName::Wave,
                tile: TileColor::Blue,
                title: "Processing quality",
                subtitle: "Resampling and noise handling. Higher is slower.",
                stackable: true,
                control: rsx! {
                    Segmented {
                        label: "Processing quality",
                        value: quality,
                        segments: vec![
                            Segment::new(AudioQuality::Low, "Low"),
                            Segment::new(AudioQuality::Medium, "Medium"),
                            Segment::new(AudioQuality::High, "High"),
                            Segment::new(AudioQuality::Ultra, "Ultra"),
                        ],
                        onchange: move |quality| report(toasts, settings.update(|s| s.audio.quality = quality)),
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
                        value: file_format,
                        options: vec![
                            SelectOption::new(AudioFormat::Mp3, "MP3"),
                            SelectOption::new(AudioFormat::Wav, "WAV"),
                            SelectOption::new(AudioFormat::Flac, "FLAC"),
                            SelectOption::new(AudioFormat::Ogg, "OGG"),
                        ],
                        onchange: move |format| report(toasts, settings.update(|s| s.audio.file_format = format)),
                    }
                },
            }
        }
    }
}
