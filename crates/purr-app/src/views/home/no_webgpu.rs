use crate::{
    components::{
        button::{Button, ButtonVariant},
        group::{Item, TileColor},
        icon::{Icon, IconName, IconSize},
        page::Page,
    },
    hooks::use_models::use_models,
};
use dioxus::prelude::*;

/// A browser without WebGPU cannot run the models: say why and what to do about it.
#[component]
pub fn NoWebGpu() -> Element {
    let models = use_models();
    let mut rechecked = use_signal(|| false);

    rsx! {
        Page { title: "Transcribe",
            section {
                class: "card glass stack",
                style: "padding:28px;gap:16px",
                role: "alert",
                span {
                    class: "fileicon",
                    style: "width:56px;height:56px;border-radius:18px;background:var(--warning-soft);color:var(--warning);box-shadow:none",
                    Icon { name: IconName::Cpu, size: IconSize::Large }
                }
                h2 { class: "t2", "This browser can't run speech models yet" }
                p { class: "muted", style: "margin:0;max-width:60ch",
                    "Purr runs Whisper on your graphics card through WebGPU, so your audio never has to leave your computer. WebGPU isn't available here, so there's nothing to run the model on."
                }
                div { class: "group", style: "box-shadow:none",
                    Item {
                        icon: IconName::Globe,
                        tile: TileColor::Blue,
                        title: "Use a browser with WebGPU",
                        subtitle: "Current Chrome, Edge or Safari, or Firefox where WebGPU is turned on.",
                    }
                    Item {
                        icon: IconName::Bolt,
                        tile: TileColor::Orange,
                        title: "Turn on hardware acceleration",
                        subtitle: "In your browser's system settings. WebGPU needs it, and some company policies switch it off.",
                    }
                    Item {
                        icon: IconName::Desktop,
                        tile: TileColor::Indigo,
                        title: "Or use the desktop app",
                        subtitle: "It runs on your processor or graphics card without a browser, and keeps working offline.",
                    }
                }
                div { class: "row wrap",
                    Button {
                        variant: ButtonVariant::Primary,
                        icon: IconName::Retry,
                        id: "recheck-webgpu",
                        onclick: move |_| {
                            rechecked.set(true);
                            models.recheck_gpu();
                        },
                        "Check again"
                    }
                    if rechecked() {
                        span { class: "muted", role: "status",
                            "WebGPU still isn't available. Turn on hardware acceleration, restart the browser, then check again."
                        }
                    }
                }
            }
        }
    }
}
