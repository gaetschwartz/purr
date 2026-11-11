use super::{Button, ButtonSize, ButtonVariant};
use crate::{
    platform::export::{TextFile, copy_to_clipboard, save_text_file},
    transcript::{Transcript, clock_time},
};
use dioxus::prelude::*;

#[derive(Debug, Clone, PartialEq, Eq)]
enum ActionOutcome {
    Copied,
    Saved(&'static str),
    Failed(String),
}

/// The transcript as timed segments, with copy and download actions once it is complete.
#[component]
pub fn TranscriptPanel(transcript: Transcript, complete: bool, file_stem: String) -> Element {
    let mut outcome = use_signal(|| None::<ActionOutcome>);
    let segments = transcript.segments();

    let copy_text = {
        let transcript = transcript.clone();
        move |_| {
            let text = transcript.text();
            spawn(async move {
                outcome.set(Some(match copy_to_clipboard(&text).await {
                    Ok(()) => ActionOutcome::Copied,
                    Err(error) => ActionOutcome::Failed(error.to_string()),
                }));
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
                outcome.set(Some(match saved {
                    Ok(()) => ActionOutcome::Saved("Text file"),
                    Err(error) => ActionOutcome::Failed(error.to_string()),
                }));
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
                outcome.set(Some(match saved {
                    Ok(()) => ActionOutcome::Saved("Subtitles"),
                    Err(error) => ActionOutcome::Failed(error.to_string()),
                }));
            });
        }
    };

    rsx! {
        div { class: "text-left space-y-4",
            if segments.is_empty() {
                p { class: "text-gray-600 dark:text-gray-300",
                    if complete {
                        "No speech was recognised in this audio."
                    } else {
                        "Waiting for the first words..."
                    }
                }
            } else {
                ol { class: "space-y-2", aria_live: if complete { "off" } else { "polite" },
                    for (index , segment) in segments.iter().enumerate() {
                        li { key: "{index}", class: "flex gap-3 leading-relaxed",
                            span { class: "font-mono text-xs text-gray-500 dark:text-gray-400 pt-1 w-12 shrink-0",
                                {clock_time(segment.start_time)}
                            }
                            span { class: "text-gray-800 dark:text-gray-100 whitespace-pre-wrap", "{segment.text.trim()}" }
                        }
                    }
                }
            }

            if complete && !segments.is_empty() {
                div { class: "flex flex-wrap items-center gap-3 pt-2",
                    Button { size: ButtonSize::Small, onclick: copy_text, "Copy text" }
                    Button {
                        size: ButtonSize::Small,
                        variant: ButtonVariant::Secondary,
                        onclick: save_txt,
                        "Download .txt"
                    }
                    Button {
                        size: ButtonSize::Small,
                        variant: ButtonVariant::Secondary,
                        onclick: save_srt,
                        "Download .srt"
                    }
                    span { class: "text-sm", role: "status",
                        match outcome() {
                            Some(ActionOutcome::Copied) => rsx! {
                                span { class: "text-green-700 dark:text-green-400", "Copied to the clipboard" }
                            },
                            Some(ActionOutcome::Saved(what)) => rsx! {
                                span { class: "text-green-700 dark:text-green-400", "{what} saved" }
                            },
                            Some(ActionOutcome::Failed(message)) => rsx! {
                                span { class: "text-red-600 dark:text-red-400", "{message}" }
                            },
                            None => rsx! {},
                        }
                    }
                }
            }
        }
    }
}
