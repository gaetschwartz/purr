use crate::components::{
    button::{Button, ButtonSize, ButtonVariant},
    icon::IconName,
    layer::PopoverAlign,
    popover::Popover,
    segmented::{Segment, Segmented},
    switch::Switch,
};
use dioxus::prelude::*;

/// The DOM id of the More button, which the menu and the model picker hang from.
pub const MORE_ID: &str = "more-btn";

/// How the transcript is laid out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layout {
    Segments,
    Reading,
}

/// What the More menu says about the transcription.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TranscriptionFacts {
    pub model_label: String,
    pub language_text: String,
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

/// The More menu: this transcription's facts and rare actions, then the view options.
///
/// Render it only while open; choosing an action closes it first.
#[component]
pub fn MoreMenu(
    facts: TranscriptionFacts,
    actions: Vec<MoreAction>,
    on_action: EventHandler<MoreAction>,
    layout: Signal<Layout>,
    show_times: Signal<bool>,
    on_close: EventHandler<()>,
) -> Element {
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
                        value: layout(),
                        segments: vec![
                            Segment::new(Layout::Segments, "Segments").with_icon(IconName::List),
                            Segment::new(Layout::Reading, "Reading").with_icon(IconName::Text),
                        ],
                        onchange: move |next| layout.set(next),
                    }
                }
                div { class: "prow",
                    span { class: "t", id: "ts-label", "Timestamps" }
                    Switch {
                        labelled_by: "ts-label",
                        checked: show_times(),
                        onchange: move |on| show_times.set(on),
                    }
                }
            }
        }
    }
}
