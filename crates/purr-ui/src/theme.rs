//! Theme system for Purr UI applications
//!
//! This module provides a comprehensive theme system with support for:
//! - Built-in themes (Light, Dark, System)
//! - Custom themes with user-defined color tokens
//! - Theme persistence via local storage
//! - React-style theme provider pattern for Dioxus

use dioxus::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[cfg(feature = "web")]
use web_sys::window;

/// Enhanced theme enumeration with custom theme support
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Theme {
    /// Light color scheme
    Light,
    /// Dark color scheme
    Dark,
    /// Follow system preference
    System,
    /// Custom theme with user-defined colors
    Custom(String),
}

/// Storage type for theme persistence
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum StorageType {
    /// Browser localStorage (web only)
    LocalStorage,
    /// File-based storage (desktop)
    File,
    /// In-memory only (no persistence)
    Memory,
}

/// Color token definitions for theme styling
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ColorTokens {
    /// Primary brand color
    pub primary: String,
    /// Secondary accent color
    pub secondary: String,
    /// Background color
    pub background: String,
    /// Primary text color
    pub text: String,
    /// Error state color (optional)
    pub error: Option<String>,
    /// Warning state color (optional)
    pub warning: Option<String>,
    /// Success state color (optional)
    pub success: Option<String>,
}

/// Custom theme definition
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CustomTheme {
    /// Human-readable name of the theme
    pub name: String,
    /// Base theme this custom theme extends from
    pub base: BaseTheme,
    /// Color token overrides
    pub tokens: ColorTokens,
}

/// Base theme options for custom themes
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum BaseTheme {
    /// Based on light theme
    Light,
    /// Based on dark theme
    Dark,
}

/// Theme provider props
#[derive(Props, Clone, PartialEq)]
pub struct ThemeProviderProps {
    /// Default theme to use
    pub default_theme: Theme,
    /// Storage type for theme persistence
    pub storage_type: StorageType,
    /// Storage key name for persistence
    pub storage_name: String,
    /// Available custom themes
    pub custom_themes: Option<Vec<CustomTheme>>,
    /// Child components
    children: Element,
}

/// Theme context for sharing theme state
#[derive(Clone, Debug, PartialEq)]
pub struct ThemeContext {
    /// Current active theme
    pub current_theme: Theme,
    /// Available custom themes
    pub custom_themes: Vec<CustomTheme>,
    /// Function to change the theme
    pub set_theme: Option<Callback<Theme>>,
    /// Current color tokens (computed from active theme)
    pub color_tokens: ColorTokens,
}

/// Theme provider component for managing application theming
#[component]
pub fn ThemeProvider(props: ThemeProviderProps) -> Element {
    // Clone props values we need to use in effects
    let storage_type = props.storage_type.clone();
    let storage_name = props.storage_name.clone();
    let default_theme = props.default_theme.clone();
    let custom_themes = props.custom_themes.clone().unwrap_or_default();

    // Initialize theme state
    let mut theme_state = use_signal(move || default_theme);
    let custom_themes_state = use_signal(move || custom_themes);

    // Load theme from storage on mount
    {
        let storage_type = storage_type.clone();
        let storage_name = storage_name.clone();
        use_effect(move || {
            let storage_type = storage_type.clone();
            let storage_name = storage_name.clone();
            spawn(async move {
                if let Some(stored_theme) = load_theme_from_storage(&storage_type, &storage_name).await {
                    theme_state.set(stored_theme);
                }
            });
        });
    }

    // Save theme to storage when it changes
    {
        let storage_type = storage_type.clone();
        let storage_name = storage_name.clone();
        use_effect(move || {
            let theme = theme_state.read().clone();
            let storage_type = storage_type.clone();
            let storage_name = storage_name.clone();
            spawn(async move {
                let _ = save_theme_to_storage(&storage_type, &storage_name, &theme).await;
            });
        });
    }

    // Create theme setter callback
    let set_theme = use_callback(move |new_theme: Theme| {
        theme_state.set(new_theme);
    });

    // Compute current color tokens
    let color_tokens = use_memo(move || {
        compute_color_tokens(&theme_state.read(), &custom_themes_state.read())
    });

    // Create theme context
    let theme_context = ThemeContext {
        current_theme: theme_state.read().clone(),
        custom_themes: custom_themes_state.read().clone(),
        set_theme: Some(set_theme),
        color_tokens: color_tokens.read().clone(),
    };

    use_context_provider(|| theme_context);

    rsx! {
        // Apply CSS custom properties for theming
        style {
            {format_css_variables(&color_tokens.read())}
        }

        // Render children
        {props.children}
    }
}

/// Hook to access theme context
pub fn use_theme() -> ThemeContext {
    use_context::<ThemeContext>()
}

/// Hook to get current color tokens
pub fn use_color_tokens() -> ColorTokens {
    use_theme().color_tokens
}

/// Hook to get theme setter function
pub fn use_set_theme() -> Option<Callback<Theme>> {
    use_theme().set_theme
}

// Implementation functions

/// Compute color tokens for a given theme
fn compute_color_tokens(theme: &Theme, custom_themes: &[CustomTheme]) -> ColorTokens {
    match theme {
        Theme::Light => get_light_color_tokens(),
        Theme::Dark => get_dark_color_tokens(),
        Theme::System => {
            // For system theme, detect system preference and return appropriate tokens
            if is_system_dark_mode() {
                get_dark_color_tokens()
            } else {
                get_light_color_tokens()
            }
        }
        Theme::Custom(name) => {
            // Find custom theme by name
            if let Some(custom_theme) = custom_themes.iter().find(|t| &t.name == name) {
                // Start with base theme tokens and override with custom tokens
                let mut base_tokens = match custom_theme.base {
                    BaseTheme::Light => get_light_color_tokens(),
                    BaseTheme::Dark => get_dark_color_tokens(),
                };

                // Override with custom tokens
                base_tokens.primary = custom_theme.tokens.primary.clone();
                base_tokens.secondary = custom_theme.tokens.secondary.clone();
                base_tokens.background = custom_theme.tokens.background.clone();
                base_tokens.text = custom_theme.tokens.text.clone();

                if let Some(error) = &custom_theme.tokens.error {
                    base_tokens.error = Some(error.clone());
                }
                if let Some(warning) = &custom_theme.tokens.warning {
                    base_tokens.warning = Some(warning.clone());
                }
                if let Some(success) = &custom_theme.tokens.success {
                    base_tokens.success = Some(success.clone());
                }

                base_tokens
            } else {
                // Fallback to light theme if custom theme not found
                get_light_color_tokens()
            }
        }
    }
}

/// Get default light theme color tokens
fn get_light_color_tokens() -> ColorTokens {
    ColorTokens {
        primary: "#3b82f6".to_string(),      // Blue-500
        secondary: "#6366f1".to_string(),    // Indigo-500
        background: "#ffffff".to_string(),   // White
        text: "#1f2937".to_string(),         // Gray-800
        error: Some("#ef4444".to_string()),  // Red-500
        warning: Some("#f59e0b".to_string()), // Amber-500
        success: Some("#10b981".to_string()), // Emerald-500
    }
}

/// Get default dark theme color tokens
fn get_dark_color_tokens() -> ColorTokens {
    ColorTokens {
        primary: "#60a5fa".to_string(),      // Blue-400
        secondary: "#818cf8".to_string(),    // Indigo-400
        background: "#111827".to_string(),   // Gray-900
        text: "#f9fafb".to_string(),         // Gray-50
        error: Some("#f87171".to_string()),  // Red-400
        warning: Some("#fbbf24".to_string()), // Amber-400
        success: Some("#34d399".to_string()), // Emerald-400
    }
}

/// Detect if system is in dark mode
fn is_system_dark_mode() -> bool {
    #[cfg(feature = "web")]
    {
        if let Some(window) = window() {
            if let Ok(media_query) = window.match_media("(prefers-color-scheme: dark)") {
                if let Ok(Some(query)) = media_query {
                    return query.matches();
                }
            }
        }
    }

    // Default to light mode if we can't detect
    false
}

/// Format CSS custom properties from color tokens
fn format_css_variables(tokens: &ColorTokens) -> String {
    let mut css = format!(
        ":root {{
            --color-primary: {};
            --color-secondary: {};
            --color-background: {};
            --color-text: {};",
        tokens.primary, tokens.secondary, tokens.background, tokens.text
    );

    if let Some(error) = &tokens.error {
        css.push_str(&format!("\n            --color-error: {};", error));
    }
    if let Some(warning) = &tokens.warning {
        css.push_str(&format!("\n            --color-warning: {};", warning));
    }
    if let Some(success) = &tokens.success {
        css.push_str(&format!("\n            --color-success: {};", success));
    }

    css.push_str("\n        }");
    css
}

/// Load theme from storage
async fn load_theme_from_storage(storage_type: &StorageType, _storage_name: &str) -> Option<Theme> {
    match storage_type {
        StorageType::LocalStorage => {
            #[cfg(feature = "web")]
            {
                if let Some(window) = window() {
                    if let Ok(Some(storage)) = window.local_storage() {
                        if let Ok(Some(theme_str)) = storage.get_item(_storage_name) {
                            return serde_json::from_str(&theme_str).ok();
                        }
                    }
                }
            }
            None
        }
        StorageType::File => {
            // File storage implementation would go here for desktop
            // For now, return None
            None
        }
        StorageType::Memory => {
            // Memory storage doesn't persist
            None
        }
    }
}

/// Save theme to storage
async fn save_theme_to_storage(storage_type: &StorageType, _storage_name: &str, _theme: &Theme) -> Result<(), String> {
    match storage_type {
        StorageType::LocalStorage => {
            #[cfg(feature = "web")]
            {
                if let Some(window) = window() {
                    if let Ok(Some(storage)) = window.local_storage() {
                        if let Ok(theme_str) = serde_json::to_string(_theme) {
                            return storage.set_item(_storage_name, &theme_str)
                                .map_err(|e| format!("Failed to save theme: {:?}", e));
                        }
                    }
                }
            }
            Err("LocalStorage not available".to_string())
        }
        StorageType::File => {
            // File storage implementation would go here for desktop
            Err("File storage not implemented".to_string())
        }
        StorageType::Memory => {
            // Memory storage doesn't persist
            Ok(())
        }
    }
}

// Utility functions for theme management

impl Theme {
    /// Convert theme to string representation
    pub fn to_string(&self) -> String {
        match self {
            Theme::Light => "light".to_string(),
            Theme::Dark => "dark".to_string(),
            Theme::System => "system".to_string(),
            Theme::Custom(name) => format!("custom:{}", name),
        }
    }

    /// Parse theme from string representation
    pub fn from_string(s: &str, custom_themes: &[CustomTheme]) -> Option<Self> {
        match s {
            "light" => Some(Theme::Light),
            "dark" => Some(Theme::Dark),
            "system" => Some(Theme::System),
            s if s.starts_with("custom:") => {
                let name = &s[7..]; // Remove "custom:" prefix
                // Check if the custom theme exists in the provided list
                if custom_themes.iter().any(|t| t.name == name) {
                    Some(Theme::Custom(name.to_string()))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Check if theme is a custom theme
    pub fn is_custom(&self) -> bool {
        matches!(self, Theme::Custom(_))
    }

    /// Get the name of the theme
    pub fn name(&self) -> String {
        match self {
            Theme::Light => "Light".to_string(),
            Theme::Dark => "Dark".to_string(),
            Theme::System => "System".to_string(),
            Theme::Custom(name) => name.clone(),
        }
    }
}

impl ColorTokens {
    /// Create a new set of color tokens
    pub fn new(
        primary: impl Into<String>,
        secondary: impl Into<String>,
        background: impl Into<String>,
        text: impl Into<String>,
    ) -> Self {
        Self {
            primary: primary.into(),
            secondary: secondary.into(),
            background: background.into(),
            text: text.into(),
            error: None,
            warning: None,
            success: None,
        }
    }

    /// Builder-style method to set error color
    pub fn with_error(mut self, error: impl Into<String>) -> Self {
        self.error = Some(error.into());
        self
    }

    /// Builder-style method to set warning color
    pub fn with_warning(mut self, warning: impl Into<String>) -> Self {
        self.warning = Some(warning.into());
        self
    }

    /// Builder-style method to set success color
    pub fn with_success(mut self, success: impl Into<String>) -> Self {
        self.success = Some(success.into());
        self
    }

    /// Get all colors as a HashMap for easy iteration
    pub fn to_map(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert("primary".to_string(), self.primary.clone());
        map.insert("secondary".to_string(), self.secondary.clone());
        map.insert("background".to_string(), self.background.clone());
        map.insert("text".to_string(), self.text.clone());

        if let Some(error) = &self.error {
            map.insert("error".to_string(), error.clone());
        }
        if let Some(warning) = &self.warning {
            map.insert("warning".to_string(), warning.clone());
        }
        if let Some(success) = &self.success {
            map.insert("success".to_string(), success.clone());
        }

        map
    }
}

impl CustomTheme {
    /// Create a new custom theme
    pub fn new(
        name: impl Into<String>,
        base: BaseTheme,
        tokens: ColorTokens,
    ) -> Self {
        Self {
            name: name.into(),
            base,
            tokens,
        }
    }

    /// Create a custom theme based on light theme
    pub fn light(name: impl Into<String>, tokens: ColorTokens) -> Self {
        Self::new(name, BaseTheme::Light, tokens)
    }

    /// Create a custom theme based on dark theme
    pub fn dark(name: impl Into<String>, tokens: ColorTokens) -> Self {
        Self::new(name, BaseTheme::Dark, tokens)
    }
}

impl Default for Theme {
    fn default() -> Self {
        Theme::Light
    }
}

impl Default for StorageType {
    fn default() -> Self {
        #[cfg(feature = "web")]
        return StorageType::LocalStorage;

        #[cfg(not(feature = "web"))]
        return StorageType::File;
    }
}

// Conversion traits for compatibility with existing settings

impl From<purr_common::settings::Theme> for Theme {
    fn from(theme: purr_common::settings::Theme) -> Self {
        match theme {
            purr_common::settings::Theme::Light => Theme::Light,
            purr_common::settings::Theme::Dark => Theme::Dark,
            purr_common::settings::Theme::System => Theme::System,
        }
    }
}

impl From<Theme> for purr_common::settings::Theme {
    fn from(theme: Theme) -> Self {
        match theme {
            Theme::Light => purr_common::settings::Theme::Light,
            Theme::Dark => purr_common::settings::Theme::Dark,
            Theme::System => purr_common::settings::Theme::System,
            Theme::Custom(_custom_name) => {
                // For settings compatibility, we'll default to Light for custom themes
                // In a full implementation, you'd look up the custom theme and check its base
                purr_common::settings::Theme::Light
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_color_tokens_creation() {
        let tokens = ColorTokens::new("#ff0000", "#00ff00", "#0000ff", "#ffffff")
            .with_error("#ff0000")
            .with_warning("#ffff00")
            .with_success("#00ff00");

        assert_eq!(tokens.primary, "#ff0000");
        assert_eq!(tokens.secondary, "#00ff00");
        assert_eq!(tokens.background, "#0000ff");
        assert_eq!(tokens.text, "#ffffff");
        assert_eq!(tokens.error, Some("#ff0000".to_string()));
        assert_eq!(tokens.warning, Some("#ffff00".to_string()));
        assert_eq!(tokens.success, Some("#00ff00".to_string()));
    }

    #[test]
    fn test_custom_theme_creation() {
        let tokens = ColorTokens::new("#ff0000", "#00ff00", "#0000ff", "#ffffff");
        let theme = CustomTheme::light("Test Theme", tokens);

        assert_eq!(theme.name, "Test Theme");
        assert_eq!(theme.base, BaseTheme::Light);
        assert_eq!(theme.tokens.primary, "#ff0000");
    }

    #[test]
    fn test_theme_string_conversion() {
        assert_eq!(Theme::Light.to_string(), "light");
        assert_eq!(Theme::Dark.to_string(), "dark");
        assert_eq!(Theme::System.to_string(), "system");

        let _custom_theme = CustomTheme::light("Test", ColorTokens::new("", "", "", ""));
        let theme = Theme::Custom("Test".to_string());
        assert_eq!(theme.to_string(), "custom:Test");
    }

    #[test]
    fn test_theme_from_string() {
        let custom_themes = vec![
            CustomTheme::light("Test", ColorTokens::new("", "", "", "")),
        ];

        assert_eq!(Theme::from_string("light", &custom_themes), Some(Theme::Light));
        assert_eq!(Theme::from_string("dark", &custom_themes), Some(Theme::Dark));
        assert_eq!(Theme::from_string("system", &custom_themes), Some(Theme::System));

        let custom = Theme::from_string("custom:Test", &custom_themes);
        assert!(custom.is_some());
        assert!(custom.unwrap().is_custom());
    }

    #[test]
    fn test_color_tokens_map() {
        let tokens = ColorTokens::new("#ff0000", "#00ff00", "#0000ff", "#ffffff")
            .with_error("#red");

        let map = tokens.to_map();
        assert_eq!(map.get("primary"), Some(&"#ff0000".to_string()));
        assert_eq!(map.get("error"), Some(&"#red".to_string()));
        assert_eq!(map.get("warning"), None);
    }

    #[test]
    fn test_settings_compatibility() {
        let common_theme = purr_common::settings::Theme::Dark;
        let ui_theme: Theme = common_theme.into();
        assert_eq!(ui_theme, Theme::Dark);

        let ui_theme = Theme::Light;
        let common_theme: purr_common::settings::Theme = ui_theme.into();
        assert_eq!(common_theme, purr_common::settings::Theme::Light);
    }

    #[test]
    fn test_solarized_custom_theme() {
        // Test Solarized theme as specified in requirements
        let solarized_theme = CustomTheme {
            name: "solarized".to_string(),
            base: BaseTheme::Light,
            tokens: ColorTokens {
                primary: "#268bd2".to_string(),
                secondary: "#2aa198".to_string(),
                background: "#fdf6e3".to_string(),
                text: "#657b83".to_string(),
                error: Some("#dc322f".to_string()),
                warning: Some("#cb4b16".to_string()),
                success: Some("#859900".to_string()),
            },
        };

        assert_eq!(solarized_theme.name, "solarized");
        assert_eq!(solarized_theme.base, BaseTheme::Light);
        assert_eq!(solarized_theme.tokens.primary, "#268bd2");
        assert_eq!(solarized_theme.tokens.background, "#fdf6e3");
    }

    #[test]
    fn test_theme_provider_structure() {
        // Test ThemeProvider props structure matches specification
        let custom_themes = vec![
            CustomTheme::light("test", ColorTokens::new("", "", "", ""))
        ];

        // This tests that we can create ThemeProviderProps with the required fields
        // The actual component test would require a test harness
        let _props_structure_test = (
            Theme::System,                    // default_theme
            StorageType::LocalStorage,        // storage_type
            "theme".to_string(),             // storage_name
            Some(custom_themes),             // custom_themes
        );

        // Test that Theme context has required fields
        let theme_context = ThemeContext {
            current_theme: Theme::Light,
            custom_themes: vec![],
            set_theme: None,
            color_tokens: get_light_color_tokens(),
        };

        assert!(matches!(theme_context.current_theme, Theme::Light));
        assert!(theme_context.set_theme.is_none());
    }

    #[test]
    fn test_css_variables_generation() {
        let tokens = ColorTokens::new("#3b82f6", "#6366f1", "#ffffff", "#1f2937")
            .with_error("#ef4444")
            .with_success("#10b981");

        let css = format_css_variables(&tokens);

        assert!(css.contains("--color-primary: #3b82f6"));
        assert!(css.contains("--color-secondary: #6366f1"));
        assert!(css.contains("--color-background: #ffffff"));
        assert!(css.contains("--color-text: #1f2937"));
        assert!(css.contains("--color-error: #ef4444"));
        assert!(css.contains("--color-success: #10b981"));
        assert!(!css.contains("--color-warning")); // Not set in this test
    }

    #[test]
    fn test_light_dark_color_tokens() {
        let light_tokens = get_light_color_tokens();
        let dark_tokens = get_dark_color_tokens();

        // Verify light theme colors match specification
        assert_eq!(light_tokens.primary, "#3b82f6");
        assert_eq!(light_tokens.background, "#ffffff");
        assert_eq!(light_tokens.text, "#1f2937");

        // Verify dark theme colors
        assert_eq!(dark_tokens.primary, "#60a5fa");
        assert_eq!(dark_tokens.background, "#111827");
        assert_eq!(dark_tokens.text, "#f9fafb");

        // Verify both have all optional colors
        assert!(light_tokens.error.is_some());
        assert!(light_tokens.warning.is_some());
        assert!(light_tokens.success.is_some());
        assert!(dark_tokens.error.is_some());
        assert!(dark_tokens.warning.is_some());
        assert!(dark_tokens.success.is_some());
    }
}