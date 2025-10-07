//! Shared settings configuration for Purr applications
//!
//! This module provides a unified settings structure that can be used across
//! all Purr components (UI, core, CLI) with proper serialization support
//! and validation methods.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Application settings configuration
///
/// This structure contains all user-configurable settings for Purr applications.
/// It supports serialization/deserialization for persistence and includes
/// validation methods to ensure configuration integrity.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Settings {
    /// Version of the settings schema for future migrations
    pub version: u32,

    /// Model configuration settings
    pub model: ModelSettings,

    /// Audio processing settings
    pub audio: AudioSettings,

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
    /// Maximum number of tokens to generate
    pub max_tokens: u32,
}

/// Audio processing settings
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AudioSettings {
    /// Audio sample rate in Hz
    pub sample_rate: SampleRate,
    /// Audio processing quality level
    pub quality: AudioQuality,
    /// Preferred audio file format
    pub file_format: AudioFormat,
}

/// User interface settings
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct UiSettings {
    /// Application color theme
    pub theme: Theme,
    /// Use compact interface layout
    pub compact_mode: bool,
    /// Interface language code
    pub language: String,
}

/// Performance and optimization settings
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PerformanceSettings {
    /// Enable GPU acceleration for transcription
    pub gpu_acceleration: bool,
    /// Enable concurrent processing of multiple files
    pub concurrent_processing: bool,
    /// Maximum number of concurrent transcription jobs
    pub max_concurrent_jobs: u32,
}

/// Logging configuration settings
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LoggingSettings {
    /// Minimum log level to record
    pub level: LogLevel,
    /// Enable writing logs to file
    pub enable_file_logging: bool,
    /// Optional custom log file path
    pub log_file_path: Option<PathBuf>,
}

/// Audio sample rate options
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum SampleRate {
    /// 8 kHz - Phone quality
    Hz8000,
    /// 16 kHz - Standard speech recognition
    Hz16000,
    /// 22.05 kHz - Near-CD quality
    Hz22050,
    /// 44.1 kHz - CD quality
    Hz44100,
    /// 48 kHz - Professional audio
    Hz48000,
}

/// Audio processing quality levels
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum AudioQuality {
    /// Low quality, fastest processing
    Low,
    /// Medium quality, balanced speed/quality
    Medium,
    /// High quality, slower processing
    High,
    /// Ultra quality, slowest processing
    Ultra,
}

/// Supported audio file formats
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum AudioFormat {
    /// MP3 format
    Mp3,
    /// WAV format
    Wav,
    /// FLAC format
    Flac,
    /// OGG format
    Ogg,
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
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
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
            version: 1,
            model: ModelSettings::default(),
            audio: AudioSettings::default(),
            ui: UiSettings::default(),
            performance: PerformanceSettings::default(),
            logging: LoggingSettings::default(),
        }
    }
}

impl Default for ModelSettings {
    fn default() -> Self {
        Self {
            model_name: "whisper-1".to_string(),
            temperature: 0.0,
            max_tokens: 4096,
        }
    }
}

impl Default for AudioSettings {
    fn default() -> Self {
        Self {
            sample_rate: SampleRate::Hz16000,
            quality: AudioQuality::High,
            file_format: AudioFormat::Mp3,
        }
    }
}

impl Default for UiSettings {
    fn default() -> Self {
        Self {
            theme: Theme::Light,
            compact_mode: false,
            language: "en".to_string(),
        }
    }
}

impl Default for PerformanceSettings {
    fn default() -> Self {
        Self {
            gpu_acceleration: false,
            concurrent_processing: true,
            max_concurrent_jobs: 2,
        }
    }
}

impl Default for LoggingSettings {
    fn default() -> Self {
        Self {
            level: LogLevel::Info,
            enable_file_logging: true,
            log_file_path: None,
        }
    }
}

// Enum conversion methods for compatibility with string-based interfaces
impl SampleRate {
    /// Convert to Hz value as a string
    pub fn to_hz_string(&self) -> &'static str {
        match self {
            SampleRate::Hz8000 => "8000",
            SampleRate::Hz16000 => "16000",
            SampleRate::Hz22050 => "22050",
            SampleRate::Hz44100 => "44100",
            SampleRate::Hz48000 => "48000",
        }
    }

    /// Convert to Hz value as a number
    pub fn to_hz(&self) -> u32 {
        match self {
            SampleRate::Hz8000 => 8000,
            SampleRate::Hz16000 => 16000,
            SampleRate::Hz22050 => 22050,
            SampleRate::Hz44100 => 44100,
            SampleRate::Hz48000 => 48000,
        }
    }

    /// Parse from Hz value string
    pub fn from_hz_string(hz: &str) -> Option<Self> {
        match hz {
            "8000" => Some(SampleRate::Hz8000),
            "16000" => Some(SampleRate::Hz16000),
            "22050" => Some(SampleRate::Hz22050),
            "44100" => Some(SampleRate::Hz44100),
            "48000" => Some(SampleRate::Hz48000),
            _ => None,
        }
    }
}

impl AudioQuality {
    /// Convert to string representation
    pub fn to_string(&self) -> &'static str {
        match self {
            AudioQuality::Low => "low",
            AudioQuality::Medium => "medium",
            AudioQuality::High => "high",
            AudioQuality::Ultra => "ultra",
        }
    }

    /// Parse from string representation
    pub fn from_string(quality: &str) -> Option<Self> {
        match quality {
            "low" => Some(AudioQuality::Low),
            "medium" => Some(AudioQuality::Medium),
            "high" => Some(AudioQuality::High),
            "ultra" => Some(AudioQuality::Ultra),
            _ => None,
        }
    }
}

impl AudioFormat {
    /// Convert to file extension string
    pub fn to_extension(&self) -> &'static str {
        match self {
            AudioFormat::Mp3 => "mp3",
            AudioFormat::Wav => "wav",
            AudioFormat::Flac => "flac",
            AudioFormat::Ogg => "ogg",
        }
    }

    /// Convert to MIME type string
    pub fn to_mime_type(&self) -> &'static str {
        match self {
            AudioFormat::Mp3 => "audio/mpeg",
            AudioFormat::Wav => "audio/wav",
            AudioFormat::Flac => "audio/flac",
            AudioFormat::Ogg => "audio/ogg",
        }
    }

    /// Parse from file extension
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_lowercase().as_str() {
            "mp3" => Some(AudioFormat::Mp3),
            "wav" => Some(AudioFormat::Wav),
            "flac" => Some(AudioFormat::Flac),
            "ogg" => Some(AudioFormat::Ogg),
            _ => None,
        }
    }
}

impl Theme {
    /// Detect the current system theme preference
    ///
    /// # Platform Support
    /// - **Windows**: Uses Windows Registry to check for dark mode
    /// - **macOS**: Uses NSUserDefaults to check appearance
    /// - **Linux**: Checks XDG desktop portal or GTK settings
    /// - **Web**: Uses prefers-color-scheme media query
    pub fn detect_system_preference() -> Self {
        #[cfg(target_os = "windows")]
        {
            Self::detect_windows_theme()
        }

        #[cfg(target_os = "macos")]
        {
            Self::detect_macos_theme()
        }

        #[cfg(target_os = "linux")]
        {
            Self::detect_linux_theme()
        }

        #[cfg(all(target_arch = "wasm32", feature = "web"))]
        {
            Self::detect_web_theme()
        }

        #[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux", all(target_arch = "wasm32", feature = "web"))))]
        {
            // Fallback for unsupported platforms
            Self::Light
        }
    }

    /// Resolve theme to concrete Light or Dark (handles System)
    pub fn resolve(&self) -> Self {
        match self {
            Self::Light => Self::Light,
            Self::Dark => Self::Dark,
            Self::System => Self::detect_system_preference(),
        }
    }

    /// Convert to storage string
    pub fn to_storage_string(&self) -> &'static str {
        match self {
            Self::Light => "light",
            Self::Dark => "dark",
            Self::System => "system",
        }
    }

    /// Parse from storage string
    pub fn from_storage_string(s: &str) -> Self {
        match s {
            "dark" => Self::Dark,
            "system" => Self::System,
            _ => Self::Light,
        }
    }

    // Platform-specific detection methods

    #[cfg(target_os = "windows")]
    fn detect_windows_theme() -> Self {
        use windows::Win32::System::Registry::{
            RegOpenKeyExW, RegQueryValueExW, HKEY, HKEY_CURRENT_USER, KEY_READ, REG_VALUE_TYPE,
        };
        use windows::core::w;

        unsafe {
            let subkey = w!("Software\\Microsoft\\Windows\\CurrentVersion\\Themes\\Personalize");
            let value_name = w!("AppsUseLightTheme");

            let mut hkey: HKEY = HKEY::default();

            // Open registry key
            if RegOpenKeyExW(HKEY_CURRENT_USER, subkey, 0, KEY_READ, &mut hkey).is_ok() {
                let mut buffer = [0u8; 4];
                let mut buffer_size = buffer.len() as u32;
                let mut value_type = REG_VALUE_TYPE::default();

                // Query value
                if RegQueryValueExW(
                    hkey,
                    value_name,
                    None,
                    Some(&mut value_type),
                    Some(buffer.as_mut_ptr()),
                    Some(&mut buffer_size),
                )
                .is_ok()
                {
                    let value = u32::from_le_bytes(buffer);
                    return if value == 0 { Self::Dark } else { Self::Light };
                }
            }
        }

        // Fallback to light if detection fails
        Self::Light
    }

    #[cfg(target_os = "macos")]
    fn detect_macos_theme() -> Self {
        use cocoa::appkit::NSAppearance;
        use cocoa::base::{id, nil};
        use cocoa::foundation::{NSAutoreleasePool, NSString};
        use objc::runtime::Object;
        use objc::{class, msg_send, sel, sel_impl};

        unsafe {
            let _pool = NSAutoreleasePool::new(nil);

            // Get effective appearance
            let appearance: id = msg_send![class!(NSAppearance), currentDrawingAppearance];
            if appearance != nil {
                let name: id = msg_send![appearance, name];
                let name_str = NSString::UTF8String(name);
                let name_cstr = std::ffi::CStr::from_ptr(name_str);

                if let Ok(name_string) = name_cstr.to_str() {
                    if name_string.contains("Dark") {
                        return Self::Dark;
                    }
                }
            }
        }

        Self::Light
    }

    #[cfg(target_os = "linux")]
    fn detect_linux_theme() -> Self {
        // Method 1: Try GTK settings
        if let Some(theme) = Self::detect_linux_gtk_theme() {
            return theme;
        }

        // Method 2: Try environment variables
        if let Some(theme) = Self::detect_linux_env_vars() {
            return theme;
        }

        // Fallback
        Self::Light
    }

    #[cfg(target_os = "linux")]
    fn detect_linux_gtk_theme() -> Option<Self> {
        use std::fs;

        // Try to read GTK settings file
        // ~/.config/gtk-3.0/settings.ini or ~/.config/gtk-4.0/settings.ini
        if let Some(config_dir) = dirs::config_dir() {
            for version in ["gtk-4.0", "gtk-3.0"] {
                let settings_path = config_dir.join(version).join("settings.ini");
                if let Ok(contents) = fs::read_to_string(&settings_path) {
                    // Look for gtk-application-prefer-dark-theme=1
                    for line in contents.lines() {
                        if line.trim().starts_with("gtk-application-prefer-dark-theme") {
                            if line.contains("=1") || line.contains("=true") {
                                return Some(Self::Dark);
                            }
                        }
                    }
                }
            }
        }

        None
    }

    #[cfg(target_os = "linux")]
    fn detect_linux_env_vars() -> Option<Self> {
        // Check GTK_THEME environment variable
        if let Ok(gtk_theme) = std::env::var("GTK_THEME") {
            if gtk_theme.to_lowercase().contains("dark") {
                return Some(Self::Dark);
            }
        }

        None
    }

    #[cfg(all(target_arch = "wasm32", feature = "web"))]
    fn detect_web_theme() -> Self {
        use web_sys::window;

        if let Some(window) = window() {
            if let Ok(Some(media_query)) = window.match_media("(prefers-color-scheme: dark)") {
                return if media_query.matches() {
                    Self::Dark
                } else {
                    Self::Light
                };
            }
        }

        Self::Light
    }

    /// Convert to string representation (deprecated, use to_storage_string)
    #[deprecated(since = "0.1.0", note = "Use to_storage_string instead")]
    pub fn to_string(&self) -> &'static str {
        self.to_storage_string()
    }

    /// Parse from string representation (deprecated, use from_storage_string)
    #[deprecated(since = "0.1.0", note = "Use from_storage_string instead")]
    pub fn from_string(theme: &str) -> Option<Self> {
        Some(Self::from_storage_string(theme))
    }
}

impl std::fmt::Display for Theme {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_storage_string())
    }
}

impl LogLevel {
    /// Convert to string representation
    pub fn to_string(&self) -> &'static str {
        match self {
            LogLevel::Error => "error",
            LogLevel::Warn => "warn",
            LogLevel::Info => "info",
            LogLevel::Debug => "debug",
            LogLevel::Trace => "trace",
        }
    }

    /// Parse from string representation
    pub fn from_string(level: &str) -> Option<Self> {
        match level {
            "error" => Some(LogLevel::Error),
            "warn" => Some(LogLevel::Warn),
            "info" => Some(LogLevel::Info),
            "debug" => Some(LogLevel::Debug),
            "trace" => Some(LogLevel::Trace),
            _ => None,
        }
    }
}

// Validation methods
impl Settings {
    /// Validate all settings and return any validation errors
    pub fn validate(&self) -> Vec<SettingsError> {
        let mut errors = Vec::new();

        // Validate model settings
        if self.model.temperature < 0.0 || self.model.temperature > 1.0 {
            errors.push(SettingsError::InvalidTemperature {
                value: self.model.temperature,
            });
        }

        if self.model.max_tokens == 0 || self.model.max_tokens > 32768 {
            errors.push(SettingsError::InvalidMaxTokens {
                value: self.model.max_tokens,
            });
        }

        if self.model.model_name.trim().is_empty() {
            errors.push(SettingsError::EmptyModelName);
        }

        // Validate performance settings
        if self.performance.max_concurrent_jobs == 0 || self.performance.max_concurrent_jobs > 16 {
            errors.push(SettingsError::InvalidConcurrentJobs {
                value: self.performance.max_concurrent_jobs,
            });
        }

        // Validate UI settings
        if self.ui.language.trim().is_empty() {
            errors.push(SettingsError::EmptyLanguage);
        }

        // Validate logging settings
        if let Some(ref path) = self.logging.log_file_path {
            if path.as_os_str().is_empty() {
                errors.push(SettingsError::InvalidLogPath);
            }
        }

        errors
    }

    /// Check if settings are valid (no validation errors)
    pub fn is_valid(&self) -> bool {
        self.validate().is_empty()
    }

    /// Get a builder for creating custom settings
    pub fn builder() -> SettingsBuilder {
        SettingsBuilder::new()
    }

    /// Migrate settings from an older version
    pub fn migrate_from_version(&mut self, from_version: u32) -> Result<(), SettingsError> {
        match from_version {
            0 => {
                // Migrate from version 0 to 1
                // Add any migration logic here
                self.version = 1;
                Ok(())
            }
            v if v == self.version => Ok(()), // No migration needed
            v if v > self.version => Err(SettingsError::UnsupportedVersion {
                current: self.version,
                target: v,
            }),
            _ => {
                // For now, we only support migrating from version 0
                Err(SettingsError::UnsupportedMigration {
                    from: from_version,
                    to: self.version,
                })
            }
        }
    }
}

/// Settings validation errors
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum SettingsError {
    #[error("Invalid temperature value: {value} (must be between 0.0 and 1.0)")]
    InvalidTemperature { value: f64 },

    #[error("Invalid max tokens value: {value} (must be between 1 and 32768)")]
    InvalidMaxTokens { value: u32 },

    #[error("Model name cannot be empty")]
    EmptyModelName,

    #[error("Invalid concurrent jobs value: {value} (must be between 1 and 16)")]
    InvalidConcurrentJobs { value: u32 },

    #[error("Language code cannot be empty")]
    EmptyLanguage,

    #[error("Invalid log file path")]
    InvalidLogPath,

    #[error("Unsupported settings version: current={current}, target={target}")]
    UnsupportedVersion { current: u32, target: u32 },

    #[error("Unsupported migration: from version {from} to {to}")]
    UnsupportedMigration { from: u32, to: u32 },
}

/// Builder for creating custom settings configurations
#[derive(Debug)]
pub struct SettingsBuilder {
    settings: Settings,
}

impl SettingsBuilder {
    /// Create a new settings builder with default values
    pub fn new() -> Self {
        Self {
            settings: Settings::default(),
        }
    }

    /// Set model settings
    pub fn model(mut self, model: ModelSettings) -> Self {
        self.settings.model = model;
        self
    }

    /// Set audio settings
    pub fn audio(mut self, audio: AudioSettings) -> Self {
        self.settings.audio = audio;
        self
    }

    /// Set UI settings
    pub fn ui(mut self, ui: UiSettings) -> Self {
        self.settings.ui = ui;
        self
    }

    /// Set performance settings
    pub fn performance(mut self, performance: PerformanceSettings) -> Self {
        self.settings.performance = performance;
        self
    }

    /// Set logging settings
    pub fn logging(mut self, logging: LoggingSettings) -> Self {
        self.settings.logging = logging;
        self
    }

    /// Build the settings, validating them first
    pub fn build(self) -> Result<Settings, Vec<SettingsError>> {
        let errors = self.settings.validate();
        if errors.is_empty() {
            Ok(self.settings)
        } else {
            Err(errors)
        }
    }

    /// Build the settings without validation
    pub fn build_unchecked(self) -> Settings {
        self.settings
    }
}

impl Default for SettingsBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_settings_are_valid() {
        let settings = Settings::default();
        assert!(settings.is_valid());
    }

    #[test]
    fn test_settings_validation() {
        let mut settings = Settings::default();

        // Test invalid temperature
        settings.model.temperature = 2.0;
        let errors = settings.validate();
        assert!(errors.iter().any(|e| matches!(e, SettingsError::InvalidTemperature { .. })));

        // Fix temperature and test invalid max_tokens
        settings.model.temperature = 0.5;
        settings.model.max_tokens = 0;
        let errors = settings.validate();
        assert!(errors.iter().any(|e| matches!(e, SettingsError::InvalidMaxTokens { .. })));
    }

    #[test]
    fn test_enum_conversions() {
        // Test SampleRate
        assert_eq!(SampleRate::Hz16000.to_hz_string(), "16000");
        assert_eq!(SampleRate::Hz16000.to_hz(), 16000);
        assert_eq!(SampleRate::from_hz_string("16000"), Some(SampleRate::Hz16000));

        // Test AudioQuality
        assert_eq!(AudioQuality::High.to_string(), "high");
        assert_eq!(AudioQuality::from_string("high"), Some(AudioQuality::High));

        // Test Theme
        assert_eq!(Theme::Dark.to_storage_string(), "dark");
        assert_eq!(Theme::from_storage_string("dark"), Theme::Dark);

        // Test LogLevel
        assert_eq!(LogLevel::Info.to_string(), "info");
        assert_eq!(LogLevel::from_string("info"), Some(LogLevel::Info));
    }

    #[test]
    fn test_settings_builder() {
        let settings = SettingsBuilder::new()
            .model(ModelSettings {
                model_name: "whisper-large".to_string(),
                temperature: 0.2,
                max_tokens: 8192,
            })
            .build()
            .unwrap();

        assert_eq!(settings.model.model_name, "whisper-large");
        assert_eq!(settings.model.temperature, 0.2);
        assert_eq!(settings.model.max_tokens, 8192);
    }

    #[test]
    fn test_serialization() {
        let settings = Settings::default();
        let json = serde_json::to_string(&settings).unwrap();
        let deserialized: Settings = serde_json::from_str(&json).unwrap();
        assert_eq!(settings, deserialized);
    }
}