//! Example demonstrating how to use the use_theme hook
//!
//! This shows the exact API that was requested in the user's specification.

use dioxus::prelude::*;
use purr_common::settings::Theme;
use purr_ui::hooks::use_theme;

/// Example component demonstrating the use_theme hook
#[component]
pub fn ThemeUsageExample() -> Element {
    let ctx = use_theme();

    // Get the current resolved theme as a string
    let resolved_theme = (ctx.resolved_theme)();

    // Example onclick handlers for theme switching
    let onclick_dark = {
        let set_theme = ctx.set_theme.clone();
        move |_| (set_theme)(Theme::Dark)
    };

    let onclick_light = {
        let set_theme = ctx.set_theme.clone();
        move |_| (set_theme)(Theme::Light)
    };

    let onclick_system = {
        let set_theme = ctx.set_theme.clone();
        move |_| (set_theme)(Theme::System)
    };

    rsx! {
        div { class: "p-6 space-y-4",
            h2 { class: "text-2xl font-bold mb-4", "Theme Hook Example" }

            p {
                class: "text-lg",
                "Current resolved theme: "
                span { class: "font-semibold text-blue-600", "{resolved_theme}" }
            }

            div { class: "flex space-x-4",
                button {
                    class: "px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600",
                    onclick: onclick_light,
                    "Light Theme"
                }

                button {
                    class: "px-4 py-2 bg-gray-800 text-white rounded hover:bg-gray-900",
                    onclick: onclick_dark,
                    "Dark Theme"
                }

                button {
                    class: "px-4 py-2 bg-gray-500 text-white rounded hover:bg-gray-600",
                    onclick: onclick_system,
                    "System Theme"
                }
            }

            div { class: "mt-6 p-4 border rounded",
                h3 { class: "font-bold mb-2", "API Usage:" }
                pre { class: "bg-gray-100 p-2 rounded text-sm",
                    r#"
let ctx = use_theme();
let resolved_theme = (ctx.resolved_theme)();
let onclick = {
    move |_| (ctx.set_theme)(Theme::Dark)
};
"#
                }
            }
        }
    }
}