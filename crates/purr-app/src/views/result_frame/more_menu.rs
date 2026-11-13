use crate::{
    components::{
        button::{Button, ButtonSize, ButtonVariant},
        icon::IconName,
        layer::PopoverAlign,
        popover::Popover,
        segmented::{Segment, Segmented},
        slider::Slider,
        switch::Switch,
        toast::use_toasts,
    },
    settings_store::use_settings,
    transcribe::{
        session::use_transcribe_session,
        view_options::{BROWSER_REASON, Details, Offer},
    },
    views::settings::changes::report,
};
use dioxus::prelude::*;
use purr_common::transcript_view::{TranscriptLayout, UnsureShare};

/// The DOM id of the More button, which the menu and the model picker hang from.
pub const MORE_ID: &str = "more-btn";

/// What the More menu says about the transcription.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TranscriptionFacts {
    pub model_label: String,
    pub language_text: String,
    /// What ran it, when known.
    pub engine_text: Option<String>,
}

/// A rare action on the transcription, offered in the More menu.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoreAction {
    RunAgain,
    NewTranscription,
    Rename,
    Delete,
}

impl MoreAction {
    const fn label(self) -> &'static str {
        match self {
            Self::RunAgain => "Run again…",
            Self::NewTranscription => "New transcription",
            Self::Rename => "Rename…",
            Self::Delete => "Delete…",
        }
    }

    const fn variant(self) -> ButtonVariant {
        match self {
            Self::RunAgain => ButtonVariant::Secondary,
            Self::NewTranscription | Self::Rename => ButtonVariant::Plain,
            Self::Delete => ButtonVariant::Danger,
        }
    }

    const fn icon(self) -> Option<IconName> {
        match self {
            Self::RunAgain => Some(IconName::Retry),
            Self::NewTranscription | Self::Rename | Self::Delete => None,
        }
    }

    const fn opens_dialog(self) -> bool {
        match self {
            Self::RunAgain | Self::Rename | Self::Delete => true,
            Self::NewTranscription => false,
        }
    }
}

/// One line saying why an option is not there.
#[component]
fn ReasonRow(title: &'static str, reason: &'static str) -> Element {
    rsx! {
        div { class: "prow na",
            span { class: "t",
                "{title}"
                small { "{reason}" }
            }
        }
    }
}

/// An option row: a title and its switch, or the reason it is missing.
#[component]
fn OfferRow(
    title: &'static str,
    label_id: &'static str,
    offer: Offer,
    checked: bool,
    on_toggle: EventHandler<bool>,
) -> Element {
    match offer {
        Offer::Switch => rsx! {
            div { class: "prow",
                span { class: "t", id: label_id, "{title}" }
                Switch {
                    labelled_by: label_id,
                    checked,
                    onchange: move |on| on_toggle.call(on),
                }
            }
        },
        Offer::Reason(reason) => rsx! {
            ReasonRow { title, reason }
        },
        Offer::Absent => rsx! {},
    }
}

/// The More menu: this transcription's facts and rare actions, then the view options.
///
/// Render it only while open; choosing an action closes it first. The view options are the
/// remembered settings; `details` says which of the Details rows the transcript can offer.
#[component]
pub fn MoreMenu(
    facts: TranscriptionFacts,
    actions: Vec<MoreAction>,
    on_action: EventHandler<MoreAction>,
    details: Details,
    on_close: EventHandler<()>,
) -> Element {
    let settings = use_settings();
    let session = use_transcribe_session();
    let toasts = use_toasts();
    let view = settings.with(|settings| settings.ui.transcript);
    let (beside, below): (Vec<MoreAction>, Vec<MoreAction>) = actions
        .into_iter()
        .partition(|action| *action == MoreAction::RunAgain);
    let button = move |action: MoreAction| {
        rsx! {
            Button {
                key: "{action:?}",
                variant: action.variant(),
                size: ButtonSize::Small,
                icon: action.icon(),
                aria_haspopup: action.opens_dialog().then_some("dialog"),
                onclick: move |_| {
                    on_close.call(());
                    on_action.call(action);
                },
                "{action.label()}"
            }
        }
    };
    let share = view.unsure_share.percent();
    rsx! {
        Popover {
            anchor: MORE_ID,
            label: "More",
            align: PopoverAlign::End,
            on_close,
            div { class: "more",
                div { class: "ph", "This transcription" }
                div { class: "prow",
                    span { class: "t",
                        "{facts.model_label}"
                        small { "{facts.language_text}" }
                        if let Some(engine) = &facts.engine_text {
                            small { "Ran on {engine}" }
                        }
                    }
                    for action in beside {
                        {button(action)}
                    }
                }
                if !below.is_empty() {
                    div { class: "prow acts",
                        for action in below {
                            {button(action)}
                        }
                    }
                }
                hr {}
                div { class: "ph", "View" }
                div { class: "prow",
                    span { class: "t", id: "layout-label", "Layout" }
                    Segmented {
                        label: "Layout",
                        value: view.layout,
                        segments: vec![
                            Segment::new(TranscriptLayout::Segments, "Segments").with_icon(IconName::List),
                            Segment::new(TranscriptLayout::Reading, "Reading").with_icon(IconName::Text),
                        ],
                        onchange: move |layout| report(toasts, settings.update(|s| s.ui.transcript.layout = layout)),
                    }
                }
                div { class: "prow",
                    span { class: "t", id: "ts-label", "Timestamps" }
                    Switch {
                        labelled_by: "ts-label",
                        checked: view.show_times,
                        onchange: move |on| report(toasts, settings.update(|s| s.ui.transcript.show_times = on)),
                    }
                }
                hr {}
                div { class: "ph", "Details" }
                match details {
                    Details::Browser => rsx! {
                        ReasonRow { title: "Unsure words, word playback, beams", reason: BROWSER_REASON }
                    },
                    Details::Desktop { unsure, word_seek, beams } => rsx! {
                        OfferRow {
                            title: "Mark unsure words",
                            label_id: "unsure-label",
                            offer: unsure,
                            checked: view.mark_unsure,
                            on_toggle: move |on| report(toasts, settings.update(|s| s.ui.transcript.mark_unsure = on)),
                        }
                        if unsure == Offer::Switch && view.mark_unsure {
                            div { class: "prow sub",
                                div { class: "slider-row",
                                    Slider {
                                        value: f64::from(share),
                                        min: f64::from(UnsureShare::MIN),
                                        max: f64::from(UnsureShare::MAX),
                                        step: f64::from(UnsureShare::STEP),
                                        label: "Share of words to mark",
                                        value_text: format!("Least sure {share}% of words"),
                                        onchange: move |percent: f64| {
                                            let share = UnsureShare::new(percent.round() as u8);
                                            report(toasts, settings.update(|s| s.ui.transcript.unsure_share = share));
                                        },
                                    }
                                }
                                output { "{share}% of words" }
                            }
                        }
                        OfferRow {
                            title: "Play from a word",
                            label_id: "seek-label",
                            offer: word_seek,
                            checked: view.word_seek,
                            on_toggle: move |on| report(toasts, settings.update(|s| s.ui.transcript.word_seek = on)),
                        }
                        OfferRow {
                            title: "Decoder beams",
                            label_id: "beams-label",
                            offer: beams,
                            checked: view.show_beams,
                            on_toggle: move |on| {
                                session.set_beam_telemetry(on);
                                report(toasts, settings.update(|s| s.ui.transcript.show_beams = on));
                            },
                        }
                    },
                }
            }
        }
    }
}
