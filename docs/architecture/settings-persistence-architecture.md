# Settings Persistence Architecture

## Executive Summary

This document outlines the architecture for implementing settings persistence in the purr-ui application. The design provides a shared Settings structure in `purr-common` that can be serialized/deserialized and persisted across both desktop and web platforms through the Platform trait abstraction.

## Goals

1. **Unified Settings Management**: Single source of truth for application settings
2. **Cross-Platform Persistence**: Desktop (TOML files) and Web (localStorage) support
3. **Type Safety**: Strongly typed settings with serde serialization
4. **Application Integration**: Settings actually influence application behavior
5. **Migration Support**: Handle settings schema evolution gracefully
6. **Default Values**: Sensible defaults with easy reset capability

## Architecture Overview

```
┌─────────────────┐
│   purr-ui       │
│   Settings      │
│   Component     │
└────────┬────────┘
         │ Uses
         ▼
┌─────────────────┐
│  purr-common    │
│ Settings Struct │
│  (Shared Core)  │
└────────┬────────┘
         │ Implements
         ▼
┌─────────────────┐
│ Platform Trait  │
│  load_settings  │
│  save_settings  │
└────────┬────────┘
         │
    ┌────┴─────┐
    │          │
    ▼          ▼
┌─────────┐ ┌──────────┐
│Desktop  │ │   WASM   │
│ (TOML)  │ │(localStorage)│
└─────────┘ └──────────┘
```

## Core Components

### 1. Settings Structure in `purr-common`

```rust
// crates/purr-common/src/settings.rs

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Application-wide settings that persist across sessions
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct Settings {
    /// Settings schema version for migration support
    #[serde(default = "default_version")]
    pub version: u32,

    /// Model configuration
    pub model: ModelSettings,

    /// Audio processing configuration
    pub audio: AudioSettings,

    /// User interface preferences
    pub ui: UiSettings,

    /// Performance tuning
    pub performance: PerformanceSettings,

    /// Logging and debugging
    pub logging: LoggingSettings,

    /// Advanced/experimental features
    pub advanced: AdvancedSettings,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct ModelSettings {
    /// Selected model name/ID
    pub selected_model: String,

    /// Temperature for transcription (0.0 - 1.0)
    #[serde(deserialize_with = "deserialize_bounded_f64::<0, 100>")]
    pub temperature: f64,

    /// Maximum tokens to generate
    #[serde(deserialize_with = "deserialize_bounded_u32::<256, 8192>")]
    pub max_tokens: u32,

    /// Preferred language for transcription (None = auto-detect)
    pub preferred_language: Option<String>,

    /// Automatically translate to English
    pub auto_translate: bool,

    /// Model download directory (desktop only)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_directory: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct AudioSettings {
    /// Sample rate in Hz
    pub sample_rate: SampleRate,

    /// Audio quality preset
    pub quality: AudioQuality,

    /// Preferred output format
    pub output_format: AudioFormat,

    /// Enable noise reduction preprocessing
    pub noise_reduction: bool,

    /// Enable voice activity detection
    pub vad_enabled: bool,

    /// VAD sensitivity (0.0 = most sensitive, 1.0 = least sensitive)
    #[serde(deserialize_with = "deserialize_bounded_f64::<0, 100>")]
    pub vad_sensitivity: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct UiSettings {
    /// Color theme
    pub theme: Theme,

    /// Compact UI mode
    pub compact_mode: bool,

    /// Interface language
    pub language: String,

    /// Show advanced options
    pub show_advanced: bool,

    /// Enable animations
    pub animations_enabled: bool,

    /// Font size scaling factor
    #[serde(deserialize_with = "deserialize_bounded_f64::<50, 200>")]
    pub font_scale: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct PerformanceSettings {
    /// Enable GPU acceleration when available
    pub gpu_acceleration: bool,

    /// Selected device ID for processing
    pub device_id: Option<i32>,

    /// Enable concurrent file processing
    pub concurrent_processing: bool,

    /// Maximum concurrent jobs
    #[serde(deserialize_with = "deserialize_bounded_u32::<1, 8>")]
    pub max_concurrent_jobs: u32,

    /// Memory limit in MB (0 = unlimited)
    pub memory_limit_mb: u32,

    /// Batch size for processing
    #[serde(deserialize_with = "deserialize_bounded_u32::<1, 64>")]
    pub batch_size: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct LoggingSettings {
    /// Minimum log level
    pub log_level: LogLevel,

    /// Enable file logging
    pub file_logging: bool,

    /// Log file directory (desktop only)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub log_directory: Option<PathBuf>,

    /// Maximum log file size in MB
    pub max_log_size_mb: u32,

    /// Keep log files for N days
    pub log_retention_days: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct AdvancedSettings {
    /// Enable experimental features
    pub experimental_features: bool,

    /// Send anonymous usage statistics
    pub telemetry_enabled: bool,

    /// Check for updates automatically
    pub auto_update_check: bool,

    /// Custom API endpoint (for self-hosted backends)
    pub custom_api_endpoint: Option<String>,

    /// Developer mode
    pub developer_mode: bool,
}

// Enums for specific settings

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum Theme {
    Light,
    Dark,
    System,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum SampleRate {
    #[serde(rename = "8000")]
    Hz8000,
    #[serde(rename = "16000")]
    Hz16000,
    #[serde(rename = "22050")]
    Hz22050,
    #[serde(rename = "44100")]
    Hz44100,
    #[serde(rename = "48000")]
    Hz48000,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum AudioQuality {
    Low,
    Medium,
    High,
    Ultra,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum AudioFormat {
    Mp3,
    Wav,
    Flac,
    Ogg,
    Webm,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum LogLevel {
    Error,
    Warn,
    Info,
    Debug,
    Trace,
}

// Default implementations

impl Default for Settings {
    fn default() -> Self {
        Self {
            version: default_version(),
            model: ModelSettings::default(),
            audio: AudioSettings::default(),
            ui: UiSettings::default(),
            performance: PerformanceSettings::default(),
            logging: LoggingSettings::default(),
            advanced: AdvancedSettings::default(),
        }
    }
}

impl Default for ModelSettings {
    fn default() -> Self {
        Self {
            selected_model: "whisper-1".to_string(),
            temperature: 0.0,
            max_tokens: 4096,
            preferred_language: None,
            auto_translate: false,
            model_directory: None,
        }
    }
}

impl Default for AudioSettings {
    fn default() -> Self {
        Self {
            sample_rate: SampleRate::Hz16000,
            quality: AudioQuality::High,
            output_format: AudioFormat::Mp3,
            noise_reduction: false,
            vad_enabled: false,
            vad_sensitivity: 0.5,
        }
    }
}

impl Default for UiSettings {
    fn default() -> Self {
        Self {
            theme: Theme::System,
            compact_mode: false,
            language: "en".to_string(),
            show_advanced: false,
            animations_enabled: true,
            font_scale: 100.0,
        }
    }
}

impl Default for PerformanceSettings {
    fn default() -> Self {
        Self {
            gpu_acceleration: false,
            device_id: None,
            concurrent_processing: true,
            max_concurrent_jobs: 2,
            memory_limit_mb: 0,
            batch_size: 1,
        }
    }
}

impl Default for LoggingSettings {
    fn default() -> Self {
        Self {
            log_level: LogLevel::Info,
            file_logging: true,
            log_directory: None,
            max_log_size_mb: 100,
            log_retention_days: 7,
        }
    }
}

impl Default for AdvancedSettings {
    fn default() -> Self {
        Self {
            experimental_features: false,
            telemetry_enabled: false,
            auto_update_check: true,
            custom_api_endpoint: None,
            developer_mode: false,
        }
    }
}

fn default_version() -> u32 {
    1
}

// Custom deserializers for bounded values
fn deserialize_bounded_f64<'de, const MIN: u8, const MAX: u8, D>(
    deserializer: D,
) -> Result<f64, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = f64::deserialize(deserializer)?;
    let min = MIN as f64 / 100.0;
    let max = MAX as f64 / 100.0;
    Ok(value.clamp(min, max))
}

fn deserialize_bounded_u32<'de, const MIN: u32, const MAX: u32, D>(
    deserializer: D,
) -> Result<u32, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = u32::deserialize(deserializer)?;
    Ok(value.clamp(MIN, MAX))
}

// Conversion helpers for UI
impl SampleRate {
    pub fn as_hz(&self) -> u32 {
        match self {
            SampleRate::Hz8000 => 8000,
            SampleRate::Hz16000 => 16000,
            SampleRate::Hz22050 => 22050,
            SampleRate::Hz44100 => 44100,
            SampleRate::Hz48000 => 48000,
        }
    }
}

impl AudioFormat {
    pub fn extension(&self) -> &'static str {
        match self {
            AudioFormat::Mp3 => "mp3",
            AudioFormat::Wav => "wav",
            AudioFormat::Flac => "flac",
            AudioFormat::Ogg => "ogg",
            AudioFormat::Webm => "webm",
        }
    }
}

impl LogLevel {
    pub fn as_str(&self) -> &'static str {
        match self {
            LogLevel::Error => "error",
            LogLevel::Warn => "warn",
            LogLevel::Info => "info",
            LogLevel::Debug => "debug",
            LogLevel::Trace => "trace",
        }
    }
}
```

### 2. Platform Trait Extensions

Add these methods to the `Platform` trait in `purr-common/src/platform.rs`:

```rust
pub trait Platform: Send + Sync + 'static {
    // ... existing methods ...

    // ========================================
    // Settings Management Operations
    // ========================================

    /// Load settings from persistent storage
    ///
    /// # Returns
    /// Loaded settings or default if none exist/corrupted
    ///
    /// # Platform Differences
    /// - **Desktop**: Reads from `~/.config/purr/settings.toml` using toml_edit
    /// - **Web**: Reads from localStorage with key "purr_settings"
    ///
    /// # Migration
    /// Implementations should handle version upgrades gracefully
    fn load_settings(
        &self,
    ) -> impl std::future::Future<Output = Result<Settings, PlatformError>> + Send;

    /// Save settings to persistent storage
    ///
    /// # Arguments
    /// * `settings` - Settings to persist
    ///
    /// # Platform Differences
    /// - **Desktop**: Writes to `~/.config/purr/settings.toml`, creates dir if needed
    /// - **Web**: Writes to localStorage, handles quota errors
    ///
    /// # Atomicity
    /// Implementations should attempt atomic writes where possible
    fn save_settings(
        &self,
        settings: &Settings,
    ) -> impl std::future::Future<Output = Result<(), PlatformError>> + Send;

    /// Get the settings storage location (for display/debugging)
    ///
    /// # Returns
    /// Human-readable description of where settings are stored
    ///
    /// # Examples
    /// - Desktop: "/home/user/.config/purr/settings.toml"
    /// - Web: "Browser Local Storage"
    fn get_settings_location(&self) -> String;

    /// Export settings to a string (for backup/sharing)
    ///
    /// # Arguments
    /// * `settings` - Settings to export
    ///
    /// # Returns
    /// TOML-formatted string representation
    fn export_settings(
        &self,
        settings: &Settings,
    ) -> impl std::future::Future<Output = Result<String, PlatformError>> + Send;

    /// Import settings from a string
    ///
    /// # Arguments
    /// * `data` - TOML-formatted settings string
    ///
    /// # Returns
    /// Parsed and validated settings
    fn import_settings(
        &self,
        data: &str,
    ) -> impl std::future::Future<Output = Result<Settings, PlatformError>> + Send;
}
```

### 3. Desktop Implementation

```rust
// crates/purr-ui/src/platform/desktop.rs (additions)

use directories::ProjectDirs;
use std::fs;
use std::path::PathBuf;
use toml_edit::{Document, Item};
use purr_common::settings::Settings;

impl Platform for PlatformImpl {
    async fn load_settings(&self) -> Result<Settings, PlatformError> {
        let path = self.get_settings_path()?;

        if !path.exists() {
            // Return defaults if file doesn't exist
            return Ok(Settings::default());
        }

        let contents = fs::read_to_string(&path)
            .map_err(|e| PlatformError::io(e))?;

        // Try to parse, fall back to defaults on error
        match toml_edit::de::from_str::<Settings>(&contents) {
            Ok(mut settings) => {
                // Handle migrations if needed
                settings = self.migrate_settings(settings)?;
                Ok(settings)
            }
            Err(e) => {
                tracing::warn!("Failed to parse settings, using defaults: {}", e);
                Ok(Settings::default())
            }
        }
    }

    async fn save_settings(&self, settings: &Settings) -> Result<(), PlatformError> {
        let path = self.get_settings_path()?;

        // Ensure parent directory exists
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| PlatformError::io(e))?;
        }

        // Serialize to TOML
        let toml_string = toml_edit::ser::to_string_pretty(settings)
            .map_err(|e| PlatformError::io(e))?;

        // Write atomically using temp file + rename
        let temp_path = path.with_extension("tmp");
        fs::write(&temp_path, toml_string)
            .map_err(|e| PlatformError::io(e))?;
        fs::rename(temp_path, path)
            .map_err(|e| PlatformError::io(e))?;

        Ok(())
    }

    fn get_settings_location(&self) -> String {
        self.get_settings_path()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|_| "Settings location unavailable".to_string())
    }

    async fn export_settings(&self, settings: &Settings) -> Result<String, PlatformError> {
        toml_edit::ser::to_string_pretty(settings)
            .map_err(|e| PlatformError::io(e))
    }

    async fn import_settings(&self, data: &str) -> Result<Settings, PlatformError> {
        toml_edit::de::from_str::<Settings>(data)
            .map_err(|e| PlatformError::io(e))
    }
}

impl PlatformImpl {
    fn get_settings_path(&self) -> Result<PathBuf, PlatformError> {
        ProjectDirs::from("", "", "purr")
            .ok_or_else(|| PlatformError::io("Failed to determine config directory"))
            .map(|dirs| dirs.config_dir().join("settings.toml"))
    }

    fn migrate_settings(&self, mut settings: Settings) -> Result<Settings, PlatformError> {
        // Example migration logic
        const CURRENT_VERSION: u32 = 1;

        if settings.version < CURRENT_VERSION {
            // Perform migrations based on version
            match settings.version {
                0 => {
                    // Migrate from v0 to v1
                    tracing::info!("Migrating settings from v0 to v1");
                    settings.version = 1;
                }
                _ => {}
            }
        }

        Ok(settings)
    }
}
```

### 4. WASM Implementation

```rust
// crates/purr-wasm/src/platform.rs (additions)

use wasm_bindgen::prelude::*;
use web_sys::{Storage, Window};
use purr_common::settings::Settings;

const SETTINGS_KEY: &str = "purr_settings";

impl Platform for PlatformImpl {
    async fn load_settings(&self) -> Result<Settings, PlatformError> {
        let storage = self.get_local_storage()?;

        match storage.get_item(SETTINGS_KEY) {
            Ok(Some(data)) => {
                // Try to parse stored JSON
                serde_json::from_str::<Settings>(&data)
                    .or_else(|_| {
                        // Try TOML format for imported settings
                        toml_edit::de::from_str::<Settings>(&data)
                    })
                    .unwrap_or_else(|e| {
                        tracing::warn!("Failed to parse settings: {}", e);
                        Settings::default()
                    })
            }
            Ok(None) => Ok(Settings::default()),
            Err(e) => {
                tracing::error!("Failed to access localStorage: {:?}", e);
                Ok(Settings::default())
            }
        }
    }

    async fn save_settings(&self, settings: &Settings) -> Result<(), PlatformError> {
        let storage = self.get_local_storage()?;

        // Serialize to JSON for web storage
        let json_string = serde_json::to_string(settings)
            .map_err(|e| PlatformError::io(e))?;

        storage.set_item(SETTINGS_KEY, &json_string)
            .map_err(|e| {
                PlatformError::io(format!("Failed to save to localStorage: {:?}", e))
            })?;

        Ok(())
    }

    fn get_settings_location(&self) -> String {
        "Browser Local Storage".to_string()
    }

    async fn export_settings(&self, settings: &Settings) -> Result<String, PlatformError> {
        // Export as TOML for compatibility
        toml_edit::ser::to_string_pretty(settings)
            .map_err(|e| PlatformError::io(e))
    }

    async fn import_settings(&self, data: &str) -> Result<Settings, PlatformError> {
        // Try TOML first, then JSON
        toml_edit::de::from_str::<Settings>(data)
            .or_else(|_| serde_json::from_str::<Settings>(data))
            .map_err(|e| PlatformError::io(format!("Failed to parse settings: {}", e)))
    }
}

impl PlatformImpl {
    fn get_local_storage(&self) -> Result<Storage, PlatformError> {
        web_sys::window()
            .ok_or_else(|| PlatformError::io("No window object"))?
            .local_storage()
            .map_err(|e| PlatformError::io(format!("localStorage error: {:?}", e)))?
            .ok_or_else(|| PlatformError::io("localStorage not available"))
    }
}
```

### 5. UI Integration Updates

Update the Settings component to use the new persistence layer:

```rust
// crates/purr-ui/src/views/settings.rs (key changes)

use purr_common::settings::{Settings, Theme, SampleRate, AudioQuality, AudioFormat, LogLevel};

#[component]
pub fn Settings() -> Element {
    let mut settings = use_signal(|| Settings::default());
    let mut loading = use_signal(|| true);
    let mut has_changes = use_signal(|| false);
    let mut save_status = use_signal(|| None);

    // Load settings on mount
    use_effect(move || {
        spawn(async move {
            if let Ok(platform) = platform::get_platform().await {
                match platform.load_settings().await {
                    Ok(loaded_settings) => {
                        settings.set(loaded_settings);
                        loading.set(false);
                    }
                    Err(e) => {
                        tracing::error!("Failed to load settings: {}", e);
                        loading.set(false);
                    }
                }
            }
        });
    });

    // Save settings function
    let save_settings = move |_| {
        spawn(async move {
            if let Ok(platform) = platform::get_platform().await {
                match platform.save_settings(&*settings.read()).await {
                    Ok(()) => {
                        save_status.set(Some(SaveStatus::Success));
                        has_changes.set(false);

                        // Apply settings to the application
                        apply_settings(&*settings.read()).await;
                    }
                    Err(e) => {
                        save_status.set(Some(SaveStatus::Error(e.to_string())));
                    }
                }
            }
        });
    };

    // ... rest of UI implementation
}

async fn apply_settings(settings: &Settings) {
    // Apply theme
    apply_theme(settings.ui.theme);

    // Configure logging
    configure_logging(&settings.logging);

    // Update transcription parameters
    update_transcription_config(settings);

    // Apply performance settings
    configure_performance(&settings.performance);
}
```

## Application Integration Points

### 1. Transcription Integration

```rust
// crates/purr-ui/src/views/transcription.rs

async fn start_transcription_process(
    file_id: FileId,
    mut status_signal: Signal<Option<TranscriptionStatus>>,
    mut text_signal: Signal<String>,
) {
    let platform = platform::get_platform().await?;
    let settings = platform.load_settings().await?;

    let request = TranscriptionRequest {
        file: FileSource::Uploaded(file_id),
        language: settings.model.preferred_language.clone(),
        translate: settings.model.auto_translate,
    };

    // Configure model with settings
    platform.configure_model(ModelConfig {
        model_id: &settings.model.selected_model,
        temperature: settings.model.temperature,
        max_tokens: settings.model.max_tokens,
        device_id: settings.performance.device_id,
        gpu_enabled: settings.performance.gpu_acceleration,
    }).await?;

    // Start transcription with configured parameters
    let stream = platform.transcribe(request).await?;
    // ... handle stream
}
```

### 2. Theme Application

```rust
// crates/purr-ui/src/utils/theme.rs

pub fn apply_theme(theme: Theme) {
    let document = web_sys::window()
        .and_then(|w| w.document());

    if let Some(doc) = document {
        let html = doc.document_element();
        if let Some(html) = html {
            match theme {
                Theme::Light => {
                    html.class_list().remove_1("dark");
                }
                Theme::Dark => {
                    html.class_list().add_1("dark");
                }
                Theme::System => {
                    // Check system preference
                    let prefers_dark = web_sys::window()
                        .and_then(|w| w.match_media("(prefers-color-scheme: dark)").ok())
                        .and_then(|m| m)
                        .map(|m| m.matches())
                        .unwrap_or(false);

                    if prefers_dark {
                        html.class_list().add_1("dark");
                    } else {
                        html.class_list().remove_1("dark");
                    }
                }
            }
        }
    }
}
```

### 3. Logging Configuration

```rust
// crates/purr-ui/src/platform/logging.rs

pub fn configure_logging(settings: &LoggingSettings) {
    use tracing_subscriber::EnvFilter;

    let filter = EnvFilter::new(settings.log_level.as_str());

    if settings.file_logging {
        // Configure file appender for desktop
        #[cfg(feature = "desktop")]
        {
            if let Some(log_dir) = &settings.log_directory {
                // Set up rolling file appender
                configure_file_logging(log_dir, settings);
            }
        }
    }

    // Update global subscriber
    tracing::subscriber::set_global_default(
        tracing_subscriber::fmt()
            .with_env_filter(filter)
            .finish()
    );
}
```

## Migration Strategy

### Schema Evolution

1. **Version Field**: Each settings struct includes a version field
2. **Forward Compatibility**: New fields use `#[serde(default)]`
3. **Backward Compatibility**: Old fields marked `#[serde(skip_serializing_if = "Option::is_none")]`
4. **Migration Functions**: Platform implementations handle version upgrades

### Migration Example

```rust
fn migrate_v1_to_v2(mut settings: Settings) -> Settings {
    // Example: Split a combined field into separate fields
    if settings.version == 1 {
        // Perform migration
        settings.version = 2;
        // ... migrate fields
    }
    settings
}
```

## Error Handling

### Corruption Recovery

1. **Parse Failure**: Fall back to defaults
2. **Partial Corruption**: Merge with defaults
3. **Version Mismatch**: Attempt migration, fall back if failed
4. **Storage Quota**: Web platform handles quota exceeded gracefully

### User Notifications

```rust
enum SettingsError {
    LoadFailed { fallback_used: bool },
    SaveFailed { reason: String },
    ImportFailed { line: Option<usize> },
    QuotaExceeded { available: usize, required: usize },
}
```

## Performance Considerations

### Caching

- Settings cached in memory after first load
- Write debouncing to prevent excessive disk I/O
- Batch updates when multiple settings change

### Web Storage Optimization

- Compress settings JSON before storing (if needed)
- Monitor storage quota usage
- Implement LRU cache for model preferences

## Security Considerations

### Desktop

- Settings file permissions: 0600 (user read/write only)
- No sensitive data in settings (API keys stored separately)
- Validate imported settings thoroughly

### Web

- localStorage isolated per origin
- No sensitive data in settings
- Content Security Policy prevents XSS access

## Testing Strategy

### Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_settings_default() {
        let settings = Settings::default();
        assert_eq!(settings.version, 1);
    }

    #[test]
    fn test_settings_serialization() {
        let settings = Settings::default();
        let toml = toml_edit::ser::to_string(&settings).unwrap();
        let deserialized: Settings = toml_edit::de::from_str(&toml).unwrap();
        assert_eq!(settings, deserialized);
    }

    #[test]
    fn test_bounded_values() {
        let json = r#"{"model": {"temperature": 2.0}}"#;
        let settings: Settings = serde_json::from_str(json).unwrap();
        assert_eq!(settings.model.temperature, 1.0); // Clamped to max
    }
}
```

### Integration Tests

- Test settings persistence across app restarts
- Test migration from older versions
- Test import/export functionality
- Test platform-specific storage mechanisms

## Implementation Checklist

- [ ] Create `settings.rs` in `purr-common`
- [ ] Add Platform trait methods
- [ ] Implement desktop persistence with toml_edit
- [ ] Implement WASM persistence with localStorage
- [ ] Update Settings UI component
- [ ] Integrate with transcription workflow
- [ ] Implement theme switching
- [ ] Configure logging based on settings
- [ ] Add import/export functionality
- [ ] Write comprehensive tests
- [ ] Document settings schema
- [ ] Add migration support

## Future Enhancements

1. **Cloud Sync**: Sync settings across devices
2. **Profiles**: Multiple settings profiles
3. **Presets**: Pre-configured settings for different use cases
4. **A/B Testing**: Experiment with different defaults
5. **Analytics**: Track which settings are most changed
6. **Keyboard Shortcuts**: Customizable shortcuts
7. **Plugin Settings**: Extensible settings for plugins

## Conclusion

This architecture provides a robust, type-safe, and cross-platform solution for settings persistence. The design prioritizes:

- **User Experience**: Settings persist and actually affect application behavior
- **Developer Experience**: Strongly typed, easy to extend
- **Maintainability**: Clear separation of concerns, comprehensive testing
- **Cross-Platform**: Single codebase, platform-specific optimizations

The implementation leverages Rust's type system and serde for safety while providing flexibility for future enhancements.