//! Theme system for Purr UI applications
//!
//! Heavily inspired by the theme crate (https://docs.rs/theme)
//! with Platform trait integration for storage

use dioxus::prelude::*;
use purr_common::platform::Platform;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, rc::Rc};

use crate::platform::get_platform;

// ============================================================================
// Re-export common Theme
// ============================================================================

pub use purr_common::settings::Theme;

/// Storage type for theme persistence
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum StorageType {
    /// Browser localStorage or desktop settings
    LocalStorage,
    /// Browser sessionStorage (web only)
    SessionStorage,
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
    /// Error state color
    pub error: Option<String>,
    /// Warning state color
    pub warning: Option<String>,
    /// Success state color
    pub success: Option<String>,
}

/// Custom theme definition with color tokens
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CustomTheme {
    /// Theme name identifier
    pub name: String,
    /// Base theme to extend (optional)
    pub base: Option<Theme>,
    /// Color tokens for this theme
    pub tokens: ColorTokens,
}

// ============================================================================
// ThemeProvider Component (adapted to use Platform)
// ============================================================================

/// Props for ThemeProvider component
#[derive(Props, Clone, PartialEq)]
pub struct ThemeProviderProps {
    /// Default theme to use
    pub default_theme: Theme,
    /// Storage type (LocalStorage or SessionStorage)
    pub storage_type: StorageType,
    /// Storage key name
    pub storage_name: String,
    /// Optional forced theme (locks theme to specific value)
    #[props(default)]
    pub forced_theme: Option<Theme>,
    /// Custom themes as Vec for easier API
    #[props(default)]
    pub custom_themes: Option<Vec<CustomTheme>>,
    /// Child components
    children: Element,
}

/// ThemeProvider component that wraps the app with theme context
#[component]
pub fn ThemeProvider(props: ThemeProviderProps) -> Element {
    let storage_name = props.storage_name.clone();
    let storage_name_load = storage_name.clone();
    let storage_name_save = storage_name.clone();
    let default_theme = props.default_theme.clone();
    let default_theme_load = default_theme.clone();
    let forced_theme = props.forced_theme.clone();
    let forced_theme_save = forced_theme.clone();
    let forced_theme_ctx = forced_theme.clone();

    // Convert Vec to HashMap for efficient lookup (unused but kept for future use)
    let _custom_themes: HashMap<String, Rc<CustomTheme>> = props
        .custom_themes
        .clone()
        .unwrap_or_default()
        .into_iter()
        .map(|theme| (theme.name.clone(), Rc::new(theme)))
        .collect();

    let mut theme_state = use_signal(|| default_theme.clone());

    // Load theme from Platform on mount
    use_effect(move || {
        let storage_name = storage_name_load.clone();
        let default_theme = default_theme_load;
        spawn(async move {
            let platform = get_platform();
            match platform.load_theme_setting(&storage_name).await {
                Ok(Some(stored)) => {
                    let theme = Theme::from_storage_string(&stored);
                    theme_state.set(theme);
                }
                Ok(None) => {
                    theme_state.set(default_theme);
                }
                Err(e) => {
                    tracing::warn!("Failed to load theme setting: {}", e);
                    theme_state.set(default_theme);
                }
            }
        });
    });

    // Save theme when it changes (unless forced)
    use_effect(move || {
        if forced_theme_save.is_some() {
            return;
        }

        let theme = *theme_state.read();
        let storage_name = storage_name_save.clone();
        spawn(async move {
            let platform = get_platform();
            let theme_str = theme.to_storage_string();
            if let Err(e) = platform.save_theme_setting(&storage_name, theme_str).await {
                tracing::error!("Failed to save theme setting: {}", e);
            }
        });
    });

    // Create context
    let current_theme = forced_theme_ctx
        .clone()
        .unwrap_or_else(|| theme_state.read().clone());

    let resolved_theme = use_memo(move || current_theme.resolve());

    let set_theme = use_callback(move |new_theme: Theme| {
        if forced_theme.is_none() {
            theme_state.set(new_theme);
        }
    });

    let themes = vec![Theme::Light, Theme::Dark, Theme::System];

    let ctx = UseThemeContext {
        resolved_theme: resolved_theme.read().clone(),
        set_theme,
        theme: current_theme.clone(),
        themes,
    };

    use_context_provider(|| ctx);

    // Apply theme class to root element
    let theme_class = match resolved_theme.read().clone() {
        Theme::Light => "light",
        Theme::Dark => "dark",
        Theme::System => unreachable!("System theme should be resolved"),
    };
    
    rsx! {
        div { class: theme_class,
            {props.children}
        }
    }
}

// ============================================================================
// use_theme Hook (copied from theme crate API)
// ============================================================================

/// Theme context returned by use_theme hook
#[derive(Clone)]
pub struct UseThemeContext {
    /// Currently resolved theme (Light or Dark, never System)
    pub resolved_theme: Theme,
    /// Callback to set the theme
    pub set_theme: Callback<Theme>,
    /// Current theme (may be System)
    pub theme: Theme,
    /// Available themes
    pub themes: Vec<Theme>,
}

/// Hook to access theme context
///
/// # Example
/// ```rust,no_run
/// use purr_app::theme::{use_theme, Theme};
/// 
/// // In a component:
/// let theme_ctx = use_theme();
/// let current = (theme_ctx.resolved_theme)();
/// 
/// // Change theme
/// theme_ctx.set_theme.call(Theme::Dark);
/// ```
pub fn use_theme() -> UseThemeContext {
    use_context::<UseThemeContext>()
}

// ============================================================================
// Default Implementations
// ============================================================================

impl Default for ColorTokens {
    fn default() -> Self {
        Self {
            primary: "#3b82f6".to_string(),
            secondary: "#6366f1".to_string(),
            background: "#ffffff".to_string(),
            text: "#1f2937".to_string(),
            error: Some("#ef4444".to_string()),
            warning: Some("#f59e0b".to_string()),
            success: Some("#10b981".to_string()),
        }
    }
}
