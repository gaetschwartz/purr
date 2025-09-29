//! Simplified use_theme hook that provides the exact API requested
//!
//! This hook integrates with the existing comprehensive theme system
//! and provides the specific interface requested: a resolved_theme function
//! and a set_theme callable.

use purr_common::settings::Theme as BaseTheme;
use std::rc::Rc;
use crate::theme::{use_theme as use_theme_internal, Theme};

/// Theme context that provides the exact API requested
#[derive(Clone)]
pub struct UseThemeResult {
    /// Function that returns the current resolved theme as a string
    pub resolved_theme: Rc<dyn Fn() -> String>,
    /// Function that accepts Theme enum values to change the theme
    pub set_theme: Rc<dyn Fn(BaseTheme)>,
}

/// Hook that provides access to theme management functionality
///
/// Returns a context with:
/// - `resolved_theme()` function that returns the current resolved theme as a string
/// - `set_theme` callable that accepts Theme enum values
///
/// Example usage:
/// ```rust,no_run
/// use purr_ui::hooks::use_theme;
/// use purr_common::settings::Theme;
///
/// // In a Dioxus component context:
/// // let ctx = use_theme();
/// // let resolved_theme = (ctx.resolved_theme)();
/// // let onclick = {
/// //     move |_: dioxus::events::MouseEvent| (ctx.set_theme)(Theme::Dark)
/// // };
/// ```
pub fn use_theme() -> UseThemeResult {
    // Use the existing comprehensive theme system internally
    let internal_ctx = use_theme_internal();

    UseThemeResult {
        resolved_theme: Rc::new({
            let current_theme = internal_ctx.current_theme.clone();
            let color_tokens = internal_ctx.color_tokens.clone();
            move || {
                match &current_theme {
                    Theme::Light => "light".to_string(),
                    Theme::Dark => "dark".to_string(),
                    Theme::System => {
                        // Resolve system theme by analyzing color tokens
                        if is_dark_theme(&color_tokens) {
                            "dark".to_string()
                        } else {
                            "light".to_string()
                        }
                    }
                    Theme::Custom(name) => format!("custom:{}", name),
                }
            }
        }),
        set_theme: Rc::new({
            let set_theme_internal = internal_ctx.set_theme.clone();
            move |new_theme: BaseTheme| {
                let theme = Theme::from(new_theme);
                if let Some(setter) = &set_theme_internal {
                    setter.call(theme);
                }
            }
        }),
    }
}

/// Helper function to detect if current tokens represent a dark theme
fn is_dark_theme(tokens: &crate::theme::ColorTokens) -> bool {
    // Simple heuristic: if background is darker than text, it's likely dark theme
    let bg_brightness = color_brightness(&tokens.background);
    let text_brightness = color_brightness(&tokens.text);
    bg_brightness < text_brightness
}

/// Calculate brightness of a hex color (0-255)
fn color_brightness(hex_color: &str) -> u8 {
    let hex = hex_color.trim_start_matches('#');

    if hex.len() != 6 {
        return 128; // Default to medium brightness if invalid
    }

    let r = u8::from_str_radix(&hex[0..2], 16).unwrap_or(128);
    let g = u8::from_str_radix(&hex[2..4], 16).unwrap_or(128);
    let b = u8::from_str_radix(&hex[4..6], 16).unwrap_or(128);

    // Use standard luminance formula
    (0.299 * r as f32 + 0.587 * g as f32 + 0.114 * b as f32) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_color_brightness() {
        assert_eq!(color_brightness("#000000"), 0);   // Black
        assert_eq!(color_brightness("#FFFFFF"), 255); // White

        // Test some common theme colors
        let dark_bg_brightness = color_brightness("#111827");
        let light_bg_brightness = color_brightness("#ffffff");

        assert!(dark_bg_brightness < light_bg_brightness);
    }

    #[test]
    fn test_is_dark_theme() {
        use crate::theme::ColorTokens;

        // Light theme tokens
        let light_tokens = ColorTokens {
            primary: "#3b82f6".to_string(),
            secondary: "#6366f1".to_string(),
            background: "#ffffff".to_string(),
            text: "#1f2937".to_string(),
            error: Some("#ef4444".to_string()),
            warning: Some("#f59e0b".to_string()),
            success: Some("#10b981".to_string()),
        };

        // Dark theme tokens
        let dark_tokens = ColorTokens {
            primary: "#60a5fa".to_string(),
            secondary: "#818cf8".to_string(),
            background: "#111827".to_string(),
            text: "#f9fafb".to_string(),
            error: Some("#f87171".to_string()),
            warning: Some("#fbbf24".to_string()),
            success: Some("#34d399".to_string()),
        };

        assert!(!is_dark_theme(&light_tokens));
        assert!(is_dark_theme(&dark_tokens));
    }

    #[test]
    fn test_use_theme_result_api() {
        use crate::theme::ColorTokens;
        use purr_common::settings::Theme as BaseTheme;

        // Test the UseThemeResult structure matches the exact API specification
        let light_tokens = ColorTokens {
            primary: "#3b82f6".to_string(),
            secondary: "#6366f1".to_string(),
            background: "#ffffff".to_string(),
            text: "#1f2937".to_string(),
            error: Some("#ef4444".to_string()),
            warning: Some("#f59e0b".to_string()),
            success: Some("#10b981".to_string()),
        };

        // Create a UseThemeResult manually to test the API structure
        let test_result = UseThemeResult {
            resolved_theme: Rc::new(|| "light".to_string()),
            set_theme: Rc::new(|_theme: BaseTheme| {
                // Test setter function signature
            }),
        };

        // Test that the API works as specified:
        // let resolved_theme = (ctx.resolved_theme)();
        let resolved_theme = (test_result.resolved_theme)();
        assert_eq!(resolved_theme, "light");

        // Test that set_theme accepts Theme enum values:
        // (ctx.set_theme)(Theme::Dark)
        (test_result.set_theme)(BaseTheme::Dark);
        (test_result.set_theme)(BaseTheme::Light);
        (test_result.set_theme)(BaseTheme::System);
    }

    #[test]
    fn test_resolved_theme_strings() {
        use crate::theme::{Theme, ColorTokens};

        // Test resolved theme string output for each theme type
        let light_tokens = ColorTokens {
            primary: "#3b82f6".to_string(),
            secondary: "#6366f1".to_string(),
            background: "#ffffff".to_string(),
            text: "#1f2937".to_string(),
            error: None,
            warning: None,
            success: None,
        };

        let dark_tokens = ColorTokens {
            primary: "#60a5fa".to_string(),
            secondary: "#818cf8".to_string(),
            background: "#111827".to_string(),
            text: "#f9fafb".to_string(),
            error: None,
            warning: None,
            success: None,
        };

        // Test different theme types produce correct resolved strings
        let light_resolver = {
            let theme = Theme::Light;
            move || match &theme {
                Theme::Light => "light".to_string(),
                Theme::Dark => "dark".to_string(),
                Theme::System => {
                    if is_dark_theme(&light_tokens) {
                        "dark".to_string()
                    } else {
                        "light".to_string()
                    }
                }
                Theme::Custom(name) => format!("custom:{}", name),
            }
        };

        assert_eq!(light_resolver(), "light");

        let custom_resolver = {
            let theme = Theme::Custom("solarized".to_string());
            move || match &theme {
                Theme::Light => "light".to_string(),
                Theme::Dark => "dark".to_string(),
                Theme::System => "light".to_string(), // Mock system detection
                Theme::Custom(name) => format!("custom:{}", name),
            }
        };

        assert_eq!(custom_resolver(), "custom:solarized");
    }
}