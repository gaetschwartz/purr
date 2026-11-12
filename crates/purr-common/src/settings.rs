//! Shared settings configuration for Purr applications
//!
//! This module provides a unified settings structure that can be used across
//! all Purr components (UI, core, CLI) with proper serialization support
//! and validation methods.

use crate::accel::{AccelBackend, AccelPolicy, AccelSelection};
use crate::export_format::ExportFormat;
use crate::hugging_face_token::HuggingFaceToken;
use serde::{Deserialize, Serialize};

/// The version of the settings schema this build writes.
pub const CURRENT_VERSION: u32 = 2;

/// The most CPU threads a setting may ask for.
pub const MAX_THREADS: u32 = 256;

const fn enabled() -> bool {
    true
}

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
    /// Replace the translucent glass surfaces with solid ones
    #[serde(default)]
    pub reduce_transparency: bool,
    /// The format the Export menu offers first
    #[serde(default)]
    pub export_format: ExportFormat,
}

/// Performance and optimization settings
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PerformanceSettings {
    /// Allow hardware acceleration; off means the processor only, whatever `accel_backend` says
    pub gpu_acceleration: bool,
    /// The backend to use, or `Auto` to take the best one that works
    #[serde(default)]
    pub accel_backend: AccelSelection,
    /// Move on to the next best backend when the chosen one cannot run
    #[serde(default = "enabled")]
    pub accel_fallback: bool,
    /// Threads for the processor, `None` to let Whisper choose
    #[serde(default)]
    pub threads: Option<u32>,
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
            reduce_transparency: false,
            export_format: ExportFormat::default(),
        }
    }
}

impl Default for PerformanceSettings {
    fn default() -> Self {
        Self {
            gpu_acceleration: true,
            accel_backend: AccelSelection::Auto,
            accel_fallback: true,
            threads: None,
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
    /// How transcription picks its backend.
    ///
    /// Turning `gpu_acceleration` off wins over every other choice and means the processor only.
    /// With it on, `Auto` always falls back, and a named backend falls back only when
    /// `accel_fallback` allows it.
    #[must_use]
    pub const fn accel_policy(&self) -> AccelPolicy {
        if !self.gpu_acceleration {
            return AccelPolicy::CPU_ONLY;
        }
        match self.accel_backend.backend() {
            None => AccelPolicy::Auto,
            Some(AccelBackend::Cpu) => AccelPolicy::CPU_ONLY,
            Some(backend) => AccelPolicy::Forced {
                backend,
                allow_fallback: self.accel_fallback,
            },
        }
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
    /// the new default instead of silently switching acceleration off. Its `ui.language` was the
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
                    self.performance.gpu_acceleration = true;
                    if self.ui.language == "en" {
                        "auto".clone_into(&mut self.ui.language);
                    }
                }
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
                reduce_transparency: false,
                export_format: ExportFormat::Txt,
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

    #[test]
    fn performance_settings_saved_before_the_backend_choice_still_load() {
        let saved =
            r#"{"gpu_acceleration":true,"concurrent_processing":false,"max_concurrent_jobs":3}"#;
        let performance: PerformanceSettings = serde_json::from_str(saved).unwrap();
        assert_eq!(
            performance,
            PerformanceSettings {
                gpu_acceleration: true,
                accel_backend: AccelSelection::Auto,
                accel_fallback: true,
                threads: None,
            }
        );
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
        let mut settings: Settings = toml::from_str(saved).unwrap();
        let stored = settings.version;
        assert_eq!(stored, 1);
        assert_eq!(settings.migrate_from_version(stored), Ok(()));
        assert_eq!(settings.version, CURRENT_VERSION);
        assert!(settings.performance.gpu_acceleration);
        assert_eq!(settings.performance.accel_backend, AccelSelection::Auto);
        assert_eq!(settings.performance.threads, None);
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
            let mut settings = Settings {
                performance: PerformanceSettings {
                    gpu_acceleration: false,
                    ..PerformanceSettings::default()
                },
                ..Settings::default()
            };
            assert_eq!(settings.migrate_from_version(from), Ok(()));
            assert_eq!(settings.version, CURRENT_VERSION);
            assert!(settings.performance.gpu_acceleration);
        }
    }

    #[test]
    fn current_settings_are_left_as_they_are() {
        let mut settings = Settings {
            performance: PerformanceSettings {
                gpu_acceleration: false,
                ..PerformanceSettings::default()
            },
            ..Settings::default()
        };
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

    fn policy(gpu: bool, backend: AccelSelection, fallback: bool) -> AccelPolicy {
        PerformanceSettings {
            gpu_acceleration: gpu,
            accel_backend: backend,
            accel_fallback: fallback,
            ..PerformanceSettings::default()
        }
        .accel_policy()
    }

    #[test]
    fn turning_acceleration_off_beats_every_other_choice() {
        for backend in AccelSelection::ALL {
            for fallback in [true, false] {
                assert_eq!(policy(false, backend, fallback), AccelPolicy::CPU_ONLY);
            }
        }
    }

    #[test]
    fn automatic_always_falls_back() {
        assert_eq!(policy(true, AccelSelection::Auto, true), AccelPolicy::Auto);
        assert_eq!(policy(true, AccelSelection::Auto, false), AccelPolicy::Auto);
    }

    #[test]
    fn a_named_backend_falls_back_only_when_allowed() {
        assert_eq!(
            policy(true, AccelSelection::Vulkan, true),
            AccelPolicy::Forced {
                backend: AccelBackend::Vulkan,
                allow_fallback: true
            }
        );
        assert_eq!(
            policy(true, AccelSelection::Cuda, false),
            AccelPolicy::Forced {
                backend: AccelBackend::Cuda,
                allow_fallback: false
            }
        );
    }

    #[test]
    fn choosing_the_processor_never_falls_back_to_anything() {
        assert_eq!(
            policy(true, AccelSelection::Cpu, true),
            AccelPolicy::CPU_ONLY
        );
    }

    #[test]
    fn the_defaults_accelerate_automatically() {
        assert_eq!(
            Settings::default().performance.accel_policy(),
            AccelPolicy::Auto
        );
    }

    #[test]
    fn every_setting_survives_a_save_and_a_load() {
        let mut settings = Settings::default();
        settings.model.translate = true;
        settings.performance.accel_backend = AccelSelection::Vulkan;
        settings.performance.accel_fallback = false;
        settings.performance.threads = Some(6);
        settings.ui.export_format = ExportFormat::Vtt;
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
version = 2

[model]
model_name = "small"
temperature = 0.0
translate = true

[ui]
theme = "Dark"
compact_mode = false
language = "fr"

[performance]
gpu_acceleration = true

[logging]
level = "Info"
"#;
        let mut settings: Settings = toml::from_str(saved).unwrap();
        assert_eq!(settings.migrate_from_version(2), Ok(()));
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
            "ModelSettings { model_name: \"base\", temperature: 0.0, translate: false, \
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
