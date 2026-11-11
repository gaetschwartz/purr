use super::{
    button::{Button, ButtonSize, ButtonVariant},
    icon::IconName,
    toast::use_toasts,
};
use crate::{
    platform::export::{SaveOutcome, TextFile, copy_to_clipboard, save_text_file},
    transcript::{Transcript, clock_time},
};
use dioxus::prelude::*;

/// The transcript as timed segments, with copy and download actions once it is complete.
#[component]
pub fn TranscriptPanel(transcript: Transcript, complete: bool, file_stem: String) -> Element {
    let toasts = use_toasts();
    let segments = transcript.segments();

    let copy_text = {
        let transcript = transcript.clone();
        move |_| {
            let text = transcript.text();
            spawn(async move {
                match copy_to_clipboard(&text).await {
                    Ok(()) => toasts.success("Transcript copied", "The text is on the clipboard."),
                    Err(error) => toasts.error("Couldn't copy the transcript", error.to_string()),
                };
            });
        }
    };
    let save_txt = {
        let transcript = transcript.clone();
        let file_name = format!("{file_stem}.txt");
        move |_| {
            let text = transcript.text();
            let file_name = file_name.clone();
            spawn(async move {
                let saved = save_text_file(TextFile {
                    file_name: &file_name,
                    content_type: "text/plain;charset=utf-8",
                    content: &text,
                })
                .await;
                match saved {
                    Ok(SaveOutcome::Saved) => {
                        toasts.success("Text file saved", file_name);
                    }
                    Ok(SaveOutcome::Cancelled) => {}
                    Err(error) => {
                        toasts.error("Couldn't save the text file", error.to_string());
                    }
                }
            });
        }
    };
    let save_srt = {
        let file_name = format!("{file_stem}.srt");
        move |_| {
            let subtitles = transcript.to_srt();
            let file_name = file_name.clone();
            spawn(async move {
                let saved = save_text_file(TextFile {
                    file_name: &file_name,
                    content_type: "application/x-subrip;charset=utf-8",
                    content: &subtitles,
                })
                .await;
                match saved {
                    Ok(SaveOutcome::Saved) => {
                        toasts.success("Subtitles saved", file_name);
                    }
                    Ok(SaveOutcome::Cancelled) => {}
                    Err(error) => {
                        toasts.error("Couldn't save the subtitles", error.to_string());
                    }
                }
            });
        }
    };

    rsx! {
        if segments.is_empty() {
            p { class: "faint", style: "margin:0",
                if complete {
                    "No speech was recognised in this audio."
                } else {
                    "Text appears here as soon as the first sentence is recognised."
                }
            }
        } else {
            ol {
                class: "live",
                style: "margin:0;padding:0",
                aria_live: if complete { "off" } else { "polite" },
                for (index , segment) in segments.iter().enumerate() {
                    li { key: "{index}", class: "seg-row",
                        span { class: "ts", {clock_time(segment.start_time)} }
                        span { class: "tx", "{segment.text.trim()}" }
                    }
                }
            }
        }

        if complete && !segments.is_empty() {
            div { class: "row wrap",
                Button { icon: IconName::Copy, onclick: copy_text, "Copy text" }
                Button {
                    variant: ButtonVariant::Plain,
                    size: ButtonSize::Small,
                    icon: IconName::Download,
                    onclick: save_txt,
                    "Download .txt"
                }
                Button {
                    variant: ButtonVariant::Plain,
                    size: ButtonSize::Small,
                    icon: IconName::Download,
                    onclick: save_srt,
                    "Download .srt"
                }
            }
        }
    }
}
