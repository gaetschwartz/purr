use dioxus::prelude::*;

const HERO_CSS: Asset = asset!("/assets/styling/hero.css");

#[component]
pub fn Hero() -> Element {
    rsx! {
        document::Link { rel: "stylesheet", href: HERO_CSS }

        div {
            id: "hero",
            div { 
                style: "text-align: center; padding: 2rem 0;",
                h1 { 
                    style: "font-size: 3rem; margin-bottom: 1rem; color: #333;",
                    "🎵 Purr - Audio Transcription"
                }
                p { 
                    style: "font-size: 1.2rem; color: #666; margin-bottom: 2rem;",
                    "Powered by OpenAI Whisper AI • Built with Dioxus & Rust"
                }
            }
        }
    }
}
