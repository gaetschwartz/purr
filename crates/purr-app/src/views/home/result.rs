mod transcript_card;

use crate::{
    components::{
        audio_player::{AudioPlayer, use_audio_player},
        button::{Button, ButtonSize, ButtonVariant},
        export_menu::{ExportMenu, ExportSource, copy_export},
        icon::{Icon, IconName},
        model_picker::ModelPicker,
        page::Page,
        toast::use_toasts,
    },
    shortcuts::{Key, KeyChord, Shortcut, use_shortcut},
    transcribe::{
        export_format::ExportFormat,
        language_choice::LanguageChoice,
        run::Completion,
        session::{RunRecord, use_transcribe_session},
        timing,
    },
    utils::{language_label::language_label, primary_key::use_primary_key_label},
};
use dioxus::prelude::*;
use std::rc::Rc;
use transcript_card::{SEARCH_ID, TranscriptCard};

/// A finished transcription: details, the player, the transcript and its exports.
#[component]
pub fn ResultView(record: RunRecord, completion: Completion) -> Element {
    let session = use_transcribe_session();
    let toasts = use_toasts();
    let primary_key = use_primary_key_label();
    let player = use_audio_player(&record.file);
    let mut picking = use_signal(|| false);

    let segments = Rc::new(record.run.transcript.segments());
    let overview = session.overview(&record.file.id);
    let duration = overview.duration().unwrap_or(completion.audio_seconds);
    let source = ExportSource {
        file_name: record.file.name.clone(),
        model: record.model.id.clone(),
        language: record.language_code(),
        duration_seconds: duration,
        segments: Rc::clone(&segments),
    };

    let copy_source = source.clone();
    use_shortcut(
        Shortcut::new(KeyChord::primary(Key::Char('c')).with_shift()).in_inputs(),
        move || copy_export(&copy_source, ExportFormat::Txt, toasts),
    );
    use_shortcut(
        Shortcut::new(KeyChord::primary(Key::Char('f'))).in_inputs(),
        move || {
            document::eval(&format!("document.getElementById({SEARCH_ID:?})?.focus()"));
        },
    );

    let language_text = match (&record.language, record.language_code()) {
        (LanguageChoice::Automatic, Some(code)) => format!("{} (detected)", language_label(&code)),
        (LanguageChoice::Automatic, None) => "Language not reported".to_owned(),
        (LanguageChoice::Spoken(_), _) => record.language.label(),
    };
    let speed = timing::speed(duration, completion.processing_seconds)
        .map(|speed| format!(" · {} real time", timing::speed_text(speed)))
        .unwrap_or_default();
    let peaks = match &overview {
        crate::transcribe::chosen_file::AudioOverview::Ready { peaks, .. } => {
            Some(Rc::clone(peaks))
        }
        crate::transcribe::chosen_file::AudioOverview::Reading
        | crate::transcribe::chosen_file::AudioOverview::Undecodable(_) => None,
    };
    let copy_now = source.clone();

    rsx! {
        Page {
            title: record.file.name.clone(),
            file_title: true,
            eyebrow: rsx! {
                p { class: "sub", style: "margin:0 0 2px",
                    Button {
                        variant: ButtonVariant::Plain,
                        size: ButtonSize::Small,
                        icon: IconName::Back,
                        id: "back-new",
                        style: "margin-left:-8px",
                        onclick: move |_| session.new_transcription(),
                        "New transcription"
                    }
                }
            },
            actions: rsx! {
                div { class: "row wrap",
                    Button {
                        icon: IconName::Retry,
                        id: "run-again",
                        aria_haspopup: "dialog",
                        onclick: move |_| picking.toggle(),
                        "Run again"
                    }
                    Button {
                        icon: IconName::Copy,
                        id: "copy-btn",
                        disabled: segments.is_empty(),
                        onclick: move |_| copy_export(&copy_now, ExportFormat::Txt, toasts),
                        "Copy"
                    }
                    ExportMenu { id: "export-btn", source }
                }
            },
            div { class: "metas tnum", aria_label: "Details",
                span {
                    Icon { name: IconName::Clock }
                    {timing::clock(duration)}
                }
                span {
                    Icon { name: IconName::Cube }
                    "{record.model.label}"
                }
                span {
                    Icon { name: IconName::Globe }
                    "{language_text}"
                }
                span {
                    Icon { name: IconName::Bolt }
                    "Took {timing::took(completion.processing_seconds)}{speed}"
                }
                span {
                    Icon { name: IconName::Lock }
                    "Kept on this device until you close Purr"
                }
            }
            AudioPlayer { player, duration: Some(duration), peaks }
            TranscriptCard { segments: Rc::clone(&segments), player }
            p { class: "foot", style: "text-align:center",
                "Click any line to play from there. {primary_key}+F searches, {primary_key}+Shift+C copies everything."
            }
        }
        if picking() {
            ModelPicker {
                anchor: "run-again",
                selected: record.model.id,
                on_close: move |()| picking.set(false),
                on_pick: move |entry| session.run_again(entry),
            }
        }
    }
}
