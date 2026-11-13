//! Shared settings configuration for Purr applications
//!
//! This module provides a unified settings structure that can be used across
//! all Purr components (UI, core, CLI) with proper serialization support
//! and validation methods.

use crate::accel::{AccelBackend, AccelSelection};
use crate::devices::{DeviceDefaults, DeviceOrder, DevicePreference};
use crate::export_format::ExportFormat;
use crate::hugging_face_token::HuggingFaceToken;
use crate::transcript_view::TranscriptView;
use serde::{Deserialize, Serialize};

/// The version of the settings schema this build writes.
pub const CURRENT_VERSION: u32 = 3;

/// The most CPU threads a setting may ask for.
pub const MAX_THREADS: u32 = 256;

/// Application settings configuration
///
/// This structure contains all user-configurable settings for Purr applications.
/// It supports serialization/deserialization for persistence and includes
/// validation methods to ensure configuration integrity.
///
/// Files written by older builds also hold an `audio` table, a model `max_tokens`, the
/// concurrency options and the log file options; nothing read them, so loading ignores them.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Settings {
    /// Version of the settings schema for future migrations
    pub version: u32,

    /// Model configuration settings
    pub model: ModelSettings,

    /// User interface settings
    pub ui: UiSettings,

    /// Performance and optimization settings
    pub performance: PerformanceSettings,

    /// Logging configuration
    pub logging: LoggingSettings,
}

/// Model configuration settings
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModelSettings {
    /// Name/ID of the transcription model to use
    pub model_name: String,
    /// The model every launch starts with; `None` keeps the last one used
    #[serde(default)]
    pub default_model: Option<String>,
    /// Temperature for model randomness (0.0 = deterministic, 1.0 = creative)
    pub temperature: f64,
    /// Write English text whatever language is spoken
    #[serde(default)]
    pub translate: bool,
    /// Token that authenticates Hugging Face downloads
    #[serde(default, deserialize_with = "HuggingFaceToken::deserialize_optional")]
    pub hugging_face_token: Option<HuggingFaceToken>,
}

/// User interface settings
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct UiSettings {
    /// Application color theme
    pub theme: Theme,
    /// Use compact interface layout
    pub compact_mode: bool,
    /// Language spoken in the audio as a Whisper language code, or `auto` to detect it
    pub language: String,
    /// The language every launch starts with; `None` keeps the last one used
    #[serde(default)]
    pub default_language: Option<String>,
    /// Replace the translucent glass surfaces with solid ones
    #[serde(default)]
    pub reduce_transparency: bool,
    /// The format the Export menu offers first
    #[serde(default)]
    pub export_format: ExportFormat,
    /// How the transcript is shown
    #[serde(default)]
    pub transcript: TranscriptView,
}

/// Performance and optimization settings
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PerformanceSettings {
    /// Threads for the processor, `None` to let Whisper choose
    #[serde(default)]
    pub threads: Option<u32>,
    /// How devices are ordered until the user arranges them; set by migrating older settings
    #[serde(default, skip_serializing_if = "DeviceDefaults::is_default")]
    pub device_defaults: DeviceDefaults,
    /// The devices transcription tries, first to last; empty until the user arranges them
    #[serde(default)]
    pub devices: Vec<DevicePreference>,
    /// Version 2's switch for hardware acceleration, read only to migrate it
    #[serde(default, skip_serializing, rename = "gpu_acceleration")]
    legacy_gpu_acceleration: Option<bool>,
    /// Version 2's backend choice, read only to migrate it
    #[serde(default, skip_serializing, rename = "accel_backend")]
    legacy_accel_backend: Option<AccelSelection>,
}

/// Logging configuration settings
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LoggingSettings {
    /// Minimum log level to record
    pub level: LogLevel,
}

/// Application color themes
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Theme {
    /// Light color scheme
    Light,
    /// Dark color scheme
    Dark,
    /// Follow system preference
    System,
}

/// Logging levels
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LogLevel {
    /// Only log errors
    Error,
    /// Log warnings and errors
    Warn,
    /// Log info, warnings, and errors
    Info,
    /// Log debug info and above
    Debug,
    /// Log everything including trace info
    Trace,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            version: CURRENT_VERSION,
            model: ModelSettings::default(),
            ui: UiSettings::default(),
            performance: PerformanceSettings::default(),
            logging: LoggingSettings::default(),
        }
    }
}

impl Default for ModelSettings {
    fn default() -> Self {
        Self {
            model_name: "base".to_owned(),
            default_model: None,
            temperature: 0.0,
            translate: false,
            hugging_face_token: None,
        }
    }
}

impl Default for UiSettings {
    fn default() -> Self {
        Self {
            theme: Theme::System,
            compact_mode: false,
            language: "auto".to_owned(),
            default_language: None,
            reduce_transparency: false,
            export_format: ExportFormat::default(),
            transcript: TranscriptView::default(),
        }
    }
}

impl Default for PerformanceSettings {
    fn default() -> Self {
        Self {
            threads: None,
            device_defaults: DeviceDefaults::default(),
            devices: Vec::new(),
            legacy_gpu_acceleration: None,
            legacy_accel_backend: None,
        }
    }
}

impl Default for LoggingSettings {
    fn default() -> Self {
        Self {
            level: LogLevel::Info,
        }
    }
}

impl PerformanceSettings {
    /// The devices transcription is asked to try.
    #[must_use]
    pub fn device_order(&self) -> DeviceOrder {
        DeviceOrder::saved(self.devices.clone()).with_defaults(self.device_defaults.clone())
    }

    /// Saves `devices` as the user's arranged list.
    pub fn set_devices(&mut self, devices: Vec<DevicePreference>) {
        self.devices = devices;
        self.device_defaults = DeviceDefaults::default();
    }

    /// Version 2 chose between acceleration on or off and one backend; the devices it could
    /// not name yet are ordered by [`DeviceDefaults`] once they are detected.
    fn migrate_acceleration(&mut self) {
        let accelerate = self.legacy_gpu_acceleration.take().unwrap_or(true);
        let backend = self
            .legacy_accel_backend
            .take()
            .unwrap_or_default()
            .backend();
        if !self.devices.is_empty() {
            return;
        }
        self.device_defaults = match (accelerate, backend) {
            (false, _) => DeviceDefaults {
                gpus_enabled: false,
                backends_first: Vec::new(),
            },
            (true, Some(AccelBackend::Cpu)) => DeviceDefaults {
                gpus_enabled: false,
                backends_first: vec![AccelBackend::Cpu],
            },
            (true, Some(backend)) => DeviceDefaults {
                gpus_enabled: true,
                backends_first: vec![backend],
            },
            (true, None) => DeviceDefaults::default(),
        };
    }
}

impl LogLevel {
    /// Convert to string representation
    pub fn to_string(&self) -> &'static str {
        match self {
            Self::Error => "error",
            Self::Warn => "warn",
            Self::Info => "info",
            Self::Debug => "debug",
            Self::Trace => "trace",
        }
    }
}

// Validation methods
impl Settings {
    /// Validate all settings and return any validation errors
    pub fn validate(&self) -> Vec<SettingsError> {
        let mut errors = Vec::new();

        if self.model.temperature < 0.0 || self.model.temperature > 1.0 {
            errors.push(SettingsError::InvalidTemperature {
                value: self.model.temperature,
            });
        }

        if self.model.model_name.trim().is_empty() {
            errors.push(SettingsError::EmptyModelName);
        }

        if let Some(threads) = self.performance.threads
            && !(1..=MAX_THREADS).contains(&threads)
        {
            errors.push(SettingsError::InvalidThreads { value: threads });
        }

        if self.ui.language.trim().is_empty() {
            errors.push(SettingsError::EmptyLanguage);
        }

        if self
            .model
            .default_model
            .as_deref()
            .is_some_and(|name| name.trim().is_empty())
        {
            errors.push(SettingsError::EmptyModelName);
        }

        if self
            .ui
            .default_language
            .as_deref()
            .is_some_and(|code| code.trim().is_empty())
        {
            errors.push(SettingsError::EmptyLanguage);
        }

        if let Some(token) = &self.model.hugging_face_token
            && !token.is_header_safe()
        {
            errors.push(SettingsError::InvalidHuggingFaceToken);
        }

        errors
    }

    /// Brings settings read from schema version `from_version` up to [`CURRENT_VERSION`].
    ///
    /// Version 1 saved `gpu_acceleration` as off without anything reading it, so it moves to
    /// the new default instead of silently switching acceleration off. Version 2's acceleration
    /// switch and backend choice become the order of the device list. Its `ui.language` was the
    /// interface language, `en` by default; read as the spoken language it would force English
    /// on every recording, so `en` becomes detection.
    pub fn migrate_from_version(&mut self, from_version: u32) -> Result<(), SettingsError> {
        if from_version > CURRENT_VERSION {
            return Err(SettingsError::UnsupportedVersion {
                current: CURRENT_VERSION,
                target: from_version,
            });
        }
        for step in from_version..CURRENT_VERSION {
            match step {
                0 => {}
                1 => {
                    self.performance.legacy_gpu_acceleration = Some(true);
                    if self.ui.language == "en" {
                        "auto".clone_into(&mut self.ui.language);
                    }
                }
                2 => self.performance.migrate_acceleration(),
                other => {
                    return Err(SettingsError::UnsupportedMigration {
                        from: other,
                        to: other.saturating_add(1),
                    });
                }
            }
        }
        self.version = CURRENT_VERSION;
        Ok(())
    }
}

/// Settings validation errors
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum SettingsError {
    #[error("Invalid temperature value: {value} (must be between 0.0 and 1.0)")]
    InvalidTemperature { value: f64 },

    #[error("Model name cannot be empty")]
    EmptyModelName,

    #[error("Invalid thread count: {value} (must be between 1 and 256)")]
    InvalidThreads { value: u32 },

    #[error("Language code cannot be empty")]
    EmptyLanguage,

    #[error("The Hugging Face token can only contain letters, digits and punctuation")]
    InvalidHuggingFaceToken,

    #[error("Unsupported settings version: current={current}, target={target}")]
    UnsupportedVersion { current: u32, target: u32 },

    #[error("Unsupported migration: from version {from} to {to}")]
    UnsupportedMigration { from: u32, to: u32 },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::devices::DeviceKey;
    use crate::transcript_view::{TranscriptLayout, UnsureShare};

    #[test]
    fn default_settings_are_valid() {
        assert_eq!(Settings::default().validate(), []);
    }

    #[test]
    fn a_temperature_outside_the_range_is_invalid() {
        let mut settings = Settings::default();
        settings.model.temperature = 2.0;
        assert_eq!(
            settings.validate(),
            [SettingsError::InvalidTemperature { value: 2.0 }]
        );
    }

    #[test]
    fn ui_settings_saved_before_reduce_transparency_still_load() {
        let saved = r#"{"theme":"Dark","compact_mode":false,"language":"fr"}"#;
        let ui: UiSettings = serde_json::from_str(saved).unwrap();
        assert_eq!(
            ui,
            UiSettings {
                theme: Theme::Dark,
                compact_mode: false,
                language: "fr".to_owned(),
                default_language: None,
                reduce_transparency: false,
                export_format: ExportFormat::Txt,
                transcript: TranscriptView::default(),
            }
        );
    }

    #[test]
    fn a_saved_export_format_is_kept() {
        let saved =
            r#"{"theme":"Light","compact_mode":false,"language":"en","export_format":"Srt"}"#;
        let ui: UiSettings = serde_json::from_str(saved).unwrap();
        assert_eq!(ui.export_format, ExportFormat::Srt);
    }

    #[test]
    fn model_settings_saved_before_translate_still_load() {
        let saved = r#"{"model_name":"small","temperature":0.0}"#;
        let model: ModelSettings = serde_json::from_str(saved).unwrap();
        assert_eq!(
            model,
            ModelSettings {
                model_name: "small".to_owned(),
                translate: false,
                ..ModelSettings::default()
            }
        );
    }

    fn migrated(saved: &str) -> Settings {
        let mut settings: Settings = toml::from_str(saved).unwrap();
        let stored = settings.version;
        assert_eq!(settings.migrate_from_version(stored), Ok(()));
        assert_eq!(settings.version, CURRENT_VERSION);
        settings
    }

    fn performance_of(version: u32, performance: &str) -> PerformanceSettings {
        migrated(&format!(
            "version = {version}\n[model]\nmodel_name = \"base\"\ntemperature = 0.0\n\
             [ui]\ntheme = \"Dark\"\ncompact_mode = false\nlanguage = \"fr\"\n\
             [performance]\n{performance}\n[logging]\nlevel = \"Info\"\n"
        ))
        .performance
    }

    fn defaults(gpus_enabled: bool, backends_first: &[AccelBackend]) -> DeviceDefaults {
        DeviceDefaults {
            gpus_enabled,
            backends_first: backends_first.to_vec(),
        }
    }

    #[test]
    fn a_version_two_file_without_a_backend_choice_keeps_the_usual_order() {
        let performance = performance_of(2, "gpu_acceleration = true");
        assert_eq!(performance, PerformanceSettings::default());
    }

    #[test]
    fn version_two_with_acceleration_off_switches_every_gpu_off() {
        for backend in ["Auto", "Cuda", "Vulkan", "Cpu"] {
            let performance = performance_of(
                2,
                &format!(
                    "gpu_acceleration = false\naccel_backend = \"{backend}\"\naccel_fallback = true"
                ),
            );
            assert_eq!(performance.device_defaults, defaults(false, &[]));
            assert_eq!(performance.devices, []);
        }
    }

    #[test]
    fn version_two_with_a_forced_backend_puts_its_devices_first() {
        for fallback in [true, false] {
            let performance = performance_of(
                2,
                &format!(
                    "gpu_acceleration = true\naccel_backend = \"Vulkan\"\naccel_fallback = {fallback}"
                ),
            );
            assert_eq!(
                performance.device_defaults,
                defaults(true, &[AccelBackend::Vulkan])
            );
        }
        let rocm = performance_of(2, "gpu_acceleration = true\naccel_backend = \"Hip\"");
        assert_eq!(rocm.device_defaults, defaults(true, &[AccelBackend::Hip]));
    }

    #[test]
    fn version_two_with_automatic_selection_keeps_the_usual_order() {
        let performance = performance_of(
            2,
            "gpu_acceleration = true\naccel_backend = \"Auto\"\naccel_fallback = false",
        );
        assert_eq!(performance.device_defaults, DeviceDefaults::default());
    }

    #[test]
    fn version_two_forced_to_the_processor_puts_it_first_and_the_gpus_off() {
        let performance = performance_of(2, "gpu_acceleration = true\naccel_backend = \"Cpu\"");
        assert_eq!(
            performance.device_defaults,
            defaults(false, &[AccelBackend::Cpu])
        );
        assert!(performance.device_order().is_cpu_only());
    }

    #[test]
    fn threads_survive_the_acceleration_migration() {
        let performance = performance_of(2, "gpu_acceleration = false\nthreads = 6");
        assert_eq!(performance.threads, Some(6));
    }

    #[test]
    fn version_two_json_from_the_browser_migrates_too() {
        let saved = r#"{"version":2,"model":{"model_name":"base","temperature":0.0},
            "ui":{"theme":"Dark","compact_mode":false,"language":"auto"},
            "performance":{"gpu_acceleration":true,"accel_backend":"Cpu","accel_fallback":true},
            "logging":{"level":"Info"}}"#;
        let mut settings: Settings = serde_json::from_str(saved).unwrap();
        assert_eq!(settings.migrate_from_version(2), Ok(()));
        assert!(settings.performance.device_order().is_cpu_only());
    }

    #[test]
    fn a_saved_list_wins_over_leftover_version_two_fields() {
        let mut settings = Settings::default();
        settings.performance.legacy_gpu_acceleration = Some(false);
        settings.performance.devices = vec![DevicePreference::enabled(DeviceKey::cpu())];
        assert_eq!(settings.migrate_from_version(2), Ok(()));
        assert_eq!(
            settings.performance.device_defaults,
            DeviceDefaults::default()
        );
    }

    #[test]
    fn the_old_keys_are_never_written_again() {
        let performance = performance_of(2, "gpu_acceleration = false\naccel_backend = \"Cuda\"");
        let saved = toml::to_string_pretty(&Settings {
            performance,
            ..Settings::default()
        })
        .unwrap();
        let written: toml::Table = toml::from_str(&saved).unwrap();
        let keys: Vec<&str> = written["performance"]
            .as_table()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        assert_eq!(keys, ["device_defaults", "devices"]);
    }

    #[test]
    fn a_version_one_toml_file_with_the_old_layout_loads_and_migrates() {
        let saved = r#"
version = 1

[model]
model_name = "base"
temperature = 0.0
max_tokens = 4096

[audio]
sample_rate = "Hz16000"
quality = "High"
file_format = "Mp3"

[ui]
theme = "Light"
compact_mode = false
language = "en"

[performance]
gpu_acceleration = false
concurrent_processing = true
max_concurrent_jobs = 2

[logging]
level = "Info"
enable_file_logging = true
log_file_path = "C:/logs/purr.log"
"#;
        let settings = migrated(saved);
        assert_eq!(settings.performance, PerformanceSettings::default());
        assert_eq!(settings.ui.export_format, ExportFormat::Txt);
        assert_eq!(settings.ui.language, "auto");
        assert_eq!(settings.logging.level, LogLevel::Info);
        assert_eq!(settings.validate(), []);
    }

    #[test]
    fn a_spoken_language_chosen_in_version_one_survives_the_migration() {
        let mut settings = Settings {
            ui: UiSettings {
                language: "fr".to_owned(),
                ..UiSettings::default()
            },
            ..Settings::default()
        };
        assert_eq!(settings.migrate_from_version(1), Ok(()));
        assert_eq!(settings.ui.language, "fr");
    }

    #[test]
    fn migration_reaches_the_current_version_from_every_older_one() {
        for from in 0..CURRENT_VERSION {
            let mut settings = Settings::default();
            assert_eq!(settings.migrate_from_version(from), Ok(()));
            assert_eq!(settings.version, CURRENT_VERSION);
            assert_eq!(settings.performance, PerformanceSettings::default());
        }
    }

    #[test]
    fn current_settings_are_left_as_they_are() {
        let mut settings = Settings::default();
        settings
            .performance
            .set_devices(vec![DevicePreference::enabled(DeviceKey::cpu())]);
        let before = settings.clone();
        assert_eq!(settings.migrate_from_version(CURRENT_VERSION), Ok(()));
        assert_eq!(settings, before);
    }

    #[test]
    fn settings_from_a_newer_version_are_refused() {
        let mut settings = Settings::default();
        assert_eq!(
            settings.migrate_from_version(CURRENT_VERSION + 1),
            Err(SettingsError::UnsupportedVersion {
                current: CURRENT_VERSION,
                target: CURRENT_VERSION + 1
            })
        );
    }

    #[test]
    fn thread_counts_outside_the_limits_are_invalid() {
        let with_threads = |threads| {
            let mut settings = Settings::default();
            settings.performance.threads = threads;
            settings.validate()
        };
        assert_eq!(with_threads(None), []);
        assert_eq!(with_threads(Some(1)), []);
        assert_eq!(with_threads(Some(MAX_THREADS)), []);
        assert_eq!(
            with_threads(Some(0)),
            [SettingsError::InvalidThreads { value: 0 }]
        );
        assert_eq!(
            with_threads(Some(MAX_THREADS + 1)),
            [SettingsError::InvalidThreads {
                value: MAX_THREADS + 1
            }]
        );
    }

    #[test]
    fn the_defaults_leave_every_device_in_the_usual_order() {
        assert_eq!(
            Settings::default().performance.device_order(),
            DeviceOrder::automatic()
        );
    }

    #[test]
    fn saving_a_list_forgets_the_migrated_defaults() {
        let mut performance = PerformanceSettings {
            device_defaults: defaults(false, &[AccelBackend::Cpu]),
            ..PerformanceSettings::default()
        };
        performance.set_devices(vec![DevicePreference::enabled(DeviceKey::cpu())]);
        assert_eq!(performance.device_defaults, DeviceDefaults::default());
    }

    #[test]
    fn every_setting_survives_a_save_and_a_load() {
        let mut settings = Settings::default();
        settings.model.translate = true;
        settings.performance.device_defaults = defaults(true, &[AccelBackend::Vulkan]);
        settings.performance.threads = Some(6);
        settings.performance.devices = vec![
            DevicePreference {
                key: DeviceKey::new(AccelBackend::Vulkan, "AMD Radeon(TM) Graphics", 1),
                enabled: false,
            },
            DevicePreference::enabled(DeviceKey::cpu()),
        ];
        settings.ui.export_format = ExportFormat::Vtt;
        settings.ui.transcript.layout = TranscriptLayout::Reading;
        settings.ui.transcript.show_times = false;
        settings.ui.transcript.unsure_share = UnsureShare::new(25);
        let saved = toml::to_string_pretty(&settings).unwrap();
        assert_eq!(toml::from_str::<Settings>(&saved).unwrap(), settings);
    }

    #[test]
    fn a_saved_token_is_kept_and_needs_no_new_version() {
        let mut settings = Settings::default();
        settings.model.hugging_face_token = HuggingFaceToken::new("hf_abc");
        let saved = toml::to_string_pretty(&settings).unwrap();
        let mut loaded: Settings = toml::from_str(&saved).unwrap();
        assert_eq!(loaded, settings);
        assert_eq!(loaded.migrate_from_version(CURRENT_VERSION), Ok(()));
        assert_eq!(loaded, settings);
    }

    #[test]
    fn current_settings_saved_before_the_token_load_without_one() {
        let saved = r#"
version = 3

[model]
model_name = "small"
temperature = 0.0
translate = true

[ui]
theme = "Dark"
compact_mode = false
language = "fr"

[performance]
threads = 4

[logging]
level = "Info"
"#;
        let mut settings: Settings = toml::from_str(saved).unwrap();
        assert_eq!(settings.migrate_from_version(3), Ok(()));
        assert_eq!(settings.model.hugging_face_token, None);
        assert_eq!(settings.validate(), []);
    }

    #[test]
    fn the_default_settings_hold_no_token() {
        assert_eq!(Settings::default().model.hugging_face_token, None);
        let saved = toml::to_string_pretty(&Settings::default()).unwrap();
        assert_eq!(saved.matches("hugging_face_token").count(), 0);
    }

    #[test]
    fn a_blank_stored_token_loads_as_none() {
        let saved = toml::to_string_pretty(&Settings::default())
            .unwrap()
            .replace(
                "translate = false",
                "translate = false\nhugging_face_token = \"  \"",
            );
        let settings: Settings = toml::from_str(&saved).unwrap();
        assert_eq!(settings, Settings::default());
    }

    #[test]
    fn a_token_that_cannot_be_a_header_is_invalid() {
        let mut settings = Settings::default();
        settings.model.hugging_face_token = HuggingFaceToken::new("hf abc");
        assert_eq!(
            settings.validate(),
            [SettingsError::InvalidHuggingFaceToken]
        );
        settings.model.hugging_face_token = HuggingFaceToken::new("hf_abc");
        assert_eq!(settings.validate(), []);
    }

    #[test]
    fn the_debug_form_of_the_settings_hides_the_token() {
        let model = ModelSettings {
            hugging_face_token: HuggingFaceToken::new("hf_secret"),
            ..ModelSettings::default()
        };
        assert_eq!(
            format!("{model:?}"),
            "ModelSettings { model_name: \"base\", default_model: None, temperature: 0.0, translate: false, \
             hugging_face_token: Some(HuggingFaceToken(<redacted>)) }"
        );
    }

    #[test]
    fn json_round_trips_the_defaults() {
        let settings = Settings::default();
        let json = serde_json::to_string(&settings).unwrap();
        let deserialized: Settings = serde_json::from_str(&json).unwrap();
        assert_eq!(settings, deserialized);
    }
}
