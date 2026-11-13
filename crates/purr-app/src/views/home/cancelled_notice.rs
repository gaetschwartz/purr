use crate::{
    components::{
        banner::{Banner, BannerTone},
        button::{Button, ButtonSize, ButtonVariant},
    },
    main_app::Route,
    transcribe::session::{Cancelled, TranscribeSession},
};
use dioxus::prelude::*;

/// Tells that a run was stopped and offers to start it again.
#[component]
pub fn CancelledNotice(cancelled: Cancelled, session: TranscribeSession) -> Element {
    let navigator = use_navigator();
    rsx! {
        Banner {
            tone: BannerTone::Info,
            title: "Transcription cancelled",
            role: "status",
            id: "cancelled-notice",
            actions: rsx! {
                Button {
                    variant: ButtonVariant::Primary,
                    size: ButtonSize::Small,
                    id: "restart-cancelled",
                    onclick: move |_| {
                        if let Some(id) = session.restart_cancelled() {
                            navigator.push(Route::Transcription { id });
                        }
                    },
                    "Start again"
                }
                Button {
                    size: ButtonSize::Small,
                    onclick: move |_| session.dismiss_cancelled(),
                    "Dismiss"
                }
            },
            if let Some(detail) = cancelled.detail {
                "{detail}"
            }
        }
    }
}
