use crate::theme::{use_theme, Theme};
use dioxus::prelude::*;

/// Theme component
#[component]
pub fn ThemeEditor() -> Element {
    let theme_ctx = use_theme();
    let current_theme = theme_ctx.theme.clone();
    let resolved = theme_ctx.resolved_theme.clone();

    rsx! {
        div { class: "p-6 max-w-4xl mx-auto",
            // Header
            div { class: "mb-8",
                h1 {
                    class: "text-3xl font-bold mb-2",
                    "Theme System Demo"
                }
                p { class: "text-lg",
                    "Explore different themes and see how colors change dynamically."
                }
                p { class: "text-sm text-gray-500 mt-2",
                    "Current theme: {current_theme:?} (Resolved: {resolved:?})"
                }
            }

            // Theme Selector
            div {
                class: "mb-8 p-6 rounded-lg border",

                h2 {
                    class: "text-xl font-semibold mb-4",
                    "Theme Selector"
                }

                div { class: "grid grid-cols-2 md:grid-cols-3 gap-4",
                    // Built-in themes
                    ThemeButton {
                        theme: Theme::Light,
                        label: "Light",
                        current: current_theme.clone(),
                        on_select: theme_ctx.set_theme.clone(),
                    }
                    ThemeButton {
                        theme: Theme::Dark,
                        label: "Dark",
                        current: current_theme.clone(),
                        on_select: theme_ctx.set_theme.clone(),
                    }
                    ThemeButton {
                        theme: Theme::System,
                        label: "System",
                        current: current_theme.clone(),
                        on_select: theme_ctx.set_theme.clone(),
                    }
                }
            }

            // Interactive Components Demo
            div {
                class: "p-6 rounded-lg border",

                h2 {
                    class: "text-xl font-semibold mb-4",
                    "Interactive Components"
                }

                div { class: "space-y-4",
                    // Buttons
                    div { class: "flex flex-wrap gap-4",
                        button {
                            class: "px-4 py-2 rounded font-medium transition-colors bg-blue-500 text-white hover:bg-blue-600",
                            "Primary Button"
                        }
                        button {
                            class: "px-4 py-2 rounded font-medium border border-blue-500 text-blue-500 hover:bg-blue-50 transition-colors",
                            "Secondary Button"
                        }
                        button {
                            class: "px-4 py-2 rounded font-medium bg-red-500 text-white hover:bg-red-600 transition-colors",
                            "Error Button"
                        }
                    }

                    // Alert messages
                    div { class: "space-y-2",
                        div {
                            class: "p-3 rounded border-l-4 bg-green-50 border-green-500 text-gray-800",
                            "Success: Theme system is working perfectly!"
                        }
                        div {
                            class: "p-3 rounded border-l-4 bg-yellow-50 border-yellow-500 text-gray-800",
                            "Warning: This is a demonstration component."
                        }
                        div {
                            class: "p-3 rounded border-l-4 bg-red-50 border-red-500 text-gray-800",
                            "Error: This is just for demo purposes."
                        }
                    }

                    // Form elements
                    div { class: "space-y-2",
                        label {
                            class: "block text-sm font-medium",
                            "Sample Input"
                        }
                        input {
                            class: "w-full px-3 py-2 rounded border border-gray-300 focus:border-blue-500 focus:ring-1 focus:ring-blue-500",
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
    on_select: Callback<Theme>,
) -> Element {
    let is_active = theme == current;

    let handle_click = move |_| {
        on_select.call(theme.clone());
    };

    rsx! {
        button {
            class: format!(
                "px-4 py-2 rounded font-medium border-2 transition-all {}",
                if is_active { 
                    "bg-blue-500 border-blue-500 text-white ring-2 ring-offset-2 ring-blue-300" 
                } else { 
                    "bg-transparent border-blue-500 text-blue-500 hover:bg-blue-50" 
                },
            ),
            onclick: handle_click,
            "{label}"
        }
    }
}
