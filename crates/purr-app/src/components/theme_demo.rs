use crate::theme::{use_set_theme, use_theme, Theme};
use dioxus::prelude::*;

/// Theme component
#[component]
pub fn ThemeEditor() -> Element {
    let theme_ctx = use_theme();
    let set_theme = use_set_theme();

    // Find solarized theme from available custom themes
    let solarized_theme = theme_ctx
        .custom_themes
        .iter()
        .find(|t| t.name == "solarized")
        .map(|_| Theme::Custom("solarized".to_string()));

    rsx! {
        div { class: "p-6 max-w-4xl mx-auto",
            // Header
            div { class: "mb-8",
                h1 {
                    class: "text-3xl font-bold mb-2",
                    style: "color: var(--color-primary);",
                    "Theme System Demo"
                }
                p { class: "text-lg", style: "color: var(--color-text);",
                    "Explore different themes and see how colors change dynamically."
                }
            }

            // Theme Selector
            div {
                class: "mb-8 p-6 rounded-lg border",
                style: "background-color: var(--color-background); border-color: var(--color-secondary);",

                h2 {
                    class: "text-xl font-semibold mb-4",
                    style: "color: var(--color-text);",
                    "Theme Selector"
                }

                div { class: "grid grid-cols-2 md:grid-cols-4 gap-4",
                    // Built-in themes
                    ThemeButton {
                        theme: Theme::Light,
                        label: "Light",
                        current: theme_ctx.current_theme.clone(),
                        set_theme: set_theme.clone(),
                    }
                    ThemeButton {
                        theme: Theme::Dark,
                        label: "Dark",
                        current: theme_ctx.current_theme.clone(),
                        set_theme: set_theme.clone(),
                    }
                    ThemeButton {
                        theme: Theme::System,
                        label: "System",
                        current: theme_ctx.current_theme.clone(),
                        set_theme: set_theme.clone(),
                    }
                    if let Some(solarized) = solarized_theme {
                        ThemeButton {
                            theme: solarized,
                            label: "Solarized",
                            current: theme_ctx.current_theme.clone(),
                            set_theme: set_theme.clone(),
                        }
                    }
                }

                // Current theme info
                div {
                    class: "mt-4 p-4 rounded",
                    style: "background-color: var(--color-secondary); opacity: 0.1;",
                    p { style: "color: var(--color-text);",
                        "Current theme: "
                        strong { "{theme_ctx.current_theme.to_string()}" }
                    }
                }
            }

            // Color Tokens Display
            div {
                class: "mb-8 p-6 rounded-lg border",
                style: "background-color: var(--color-background); border-color: var(--color-secondary);",

                h2 {
                    class: "text-xl font-semibold mb-4",
                    style: "color: var(--color-text);",
                    "Color Tokens"
                }

                div { class: "grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4",
                    ColorToken {
                        name: "Primary",
                        color: theme_ctx.color_tokens.primary.clone(),
                        description: "Main brand color for buttons, links, and accents",
                    }
                    ColorToken {
                        name: "Secondary",
                        color: theme_ctx.color_tokens.secondary.clone(),
                        description: "Secondary color for less prominent elements",
                    }
                    ColorToken {
                        name: "Background",
                        color: theme_ctx.color_tokens.background.clone(),
                        description: "Main background color for pages and containers",
                    }
                    ColorToken {
                        name: "Text",
                        color: theme_ctx.color_tokens.text.clone(),
                        description: "Primary text color for readable content",
                    }

                    if let Some(ref error_color) = theme_ctx.color_tokens.error {
                        ColorToken {
                            name: "Error",
                            color: error_color.clone(),
                            description: "Color for error states and messages",
                        }
                    }

                    if let Some(ref warning_color) = theme_ctx.color_tokens.warning {
                        ColorToken {
                            name: "Warning",
                            color: warning_color.clone(),
                            description: "Color for warning states and messages",
                        }
                    }

                    if let Some(ref success_color) = theme_ctx.color_tokens.success {
                        ColorToken {
                            name: "Success",
                            color: success_color.clone(),
                            description: "Color for success states and messages",
                        }
                    }
                }
            }

            // Interactive Components Demo
            div {
                class: "p-6 rounded-lg border",
                style: "background-color: var(--color-background); border-color: var(--color-secondary);",

                h2 {
                    class: "text-xl font-semibold mb-4",
                    style: "color: var(--color-text);",
                    "Interactive Components"
                }

                div { class: "space-y-4",
                    // Buttons
                    div { class: "flex flex-wrap gap-4",
                        button {
                            class: "px-4 py-2 rounded font-medium transition-colors",
                            style: "background-color: var(--color-primary); color: white;",
                            "Primary Button"
                        }
                        button {
                            class: "px-4 py-2 rounded font-medium border transition-colors",
                            style: "border-color: var(--color-primary); color: var(--color-primary); background: transparent;",
                            "Secondary Button"
                        }
                        if let Some(ref error_color) = theme_ctx.color_tokens.error {
                            button {
                                class: "px-4 py-2 rounded font-medium transition-colors",
                                style: "background-color: {error_color}; color: white;",
                                "Error Button"
                            }
                        }
                    }

                    // Alert messages
                    div { class: "space-y-2",
                        if let Some(ref success_color) = theme_ctx.color_tokens.success {
                            div {
                                class: "p-3 rounded border-l-4",
                                style: "background-color: {success_color}20; border-left-color: {success_color}; color: var(--color-text);",
                                "Success: Theme system is working perfectly!"
                            }
                        }
                        if let Some(ref warning_color) = theme_ctx.color_tokens.warning {
                            div {
                                class: "p-3 rounded border-l-4",
                                style: "background-color: {warning_color}20; border-left-color: {warning_color}; color: var(--color-text);",
                                "Warning: This is a demonstration component."
                            }
                        }
                        if let Some(ref error_color) = theme_ctx.color_tokens.error {
                            div {
                                class: "p-3 rounded border-l-4",
                                style: "background-color: {error_color}20; border-left-color: {error_color}; color: var(--color-text);",
                                "Error: This is just for demo purposes."
                            }
                        }
                    }

                    // Form elements
                    div { class: "space-y-2",
                        label {
                            class: "block text-sm font-medium",
                            style: "color: var(--color-text);",
                            "Sample Input"
                        }
                        input {
                            class: "w-full px-3 py-2 rounded border",
                            style: "background-color: var(--color-background); border-color: var(--color-secondary); color: var(--color-text);",
                            r#type: "text",
                            placeholder: "Enter some text...",
                        }
                    }
                }
            }
        }
    }
}

/// Theme button component
#[component]
fn ThemeButton(
    theme: Theme,
    label: String,
    current: Theme,
    set_theme: Option<Callback<Theme>>,
) -> Element {
    let is_active = theme == current;

    let handle_click = move |_| {
        if let Some(setter) = &set_theme {
            setter.call(theme.clone());
        }
    };

    rsx! {
        button {
            class: format!(
                "px-4 py-2 rounded font-medium border-2 transition-all {}",
                if is_active { "ring-2 ring-offset-2" } else { "" },
            ),
            style: format!(
                "border-color: var(--color-primary); {} {}",
                if is_active {
                    "background-color: var(--color-primary); color: white;"
                } else {
                    "background-color: transparent; color: var(--color-primary);"
                },
                if is_active { "ring-color: var(--color-primary);" } else { "" },
            ),
            onclick: handle_click,
            "{label}"
        }
    }
}

/// Color token display component
#[component]
fn ColorToken(name: String, color: String, description: String) -> Element {
    // Calculate contrast color for text
    let text_color = if is_light_color(&color) {
        "#000000"
    } else {
        "#ffffff"
    };

    rsx! {
        div {
            class: "border rounded-lg overflow-hidden",
            style: "border-color: var(--theme-secondary);",

            // Color swatch
            div {
                class: "h-16 flex items-center justify-center",
                style: "background-color: {color}; color: {text_color};",
                span { class: "font-mono text-sm font-medium", "{color}" }
            }

            // Token info
            div {
                class: "p-3",
                style: "background-color: var(--theme-background);",
                h3 {
                    class: "font-semibold text-sm mb-1",
                    style: "color: var(--theme-text);",
                    "{name}"
                }
                p {
                    class: "text-xs opacity-75",
                    style: "color: var(--theme-text);",
                    "{description}"
                }
            }
        }
    }
}

/// Utility function to determine if a color is light or dark
fn is_light_color(hex: &str) -> bool {
    // Remove # if present
    let hex = hex.trim_start_matches('#');

    // Parse RGB values
    if hex.len() == 6 {
        if let (Ok(r), Ok(g), Ok(b)) = (
            u8::from_str_radix(&hex[0..2], 16),
            u8::from_str_radix(&hex[2..4], 16),
            u8::from_str_radix(&hex[4..6], 16),
        ) {
            // Calculate relative luminance
            let luminance = (0.299 * r as f32 + 0.587 * g as f32 + 0.114 * b as f32) / 255.0;
            return luminance > 0.5;
        }
    }

    // Default to dark if parsing fails
    false
}
