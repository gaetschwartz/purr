use crate::{
    components::{
        button::{Button, ButtonVariant},
        sheet::Sheet,
    },
    history::store::use_history,
};
use dioxus::prelude::*;
use purr_common::history::{
    entry::HistoryId,
    title::{MAX_TITLE_CHARS, TitleProblem, clean_title},
};

const INPUT_ID: &str = "rename-input";

fn problem_text(problem: TitleProblem) -> String {
    match problem {
        TitleProblem::Empty => "Type a name for this transcription.".to_owned(),
        TitleProblem::TooLong => format!("Use at most {MAX_TITLE_CHARS} characters."),
    }
}

/// A sheet that renames a transcription; the new name is checked before it is saved.
#[component]
pub fn RenameDialog(id: HistoryId, title: String, on_close: EventHandler<()>) -> Element {
    let store = use_history();
    let mut draft = use_signal(|| title.clone());
    let mut problem = use_signal(|| None::<TitleProblem>);

    let submit = use_callback(move |()| match clean_title(&draft()) {
        Ok(next) => {
            if next != title {
                store.rename(id.clone(), next);
            }
            on_close.call(());
        }
        Err(found) => problem.set(Some(found)),
    });

    rsx! {
        Sheet {
            title: "Rename transcript",
            on_close,
            actions: rsx! {
                Button { onclick: move |_| on_close.call(()), "Cancel" }
                Button {
                    variant: ButtonVariant::Primary,
                    id: "rename-ok",
                    onclick: move |_| submit.call(()),
                    "Rename"
                }
            },
            label { class: "sr", r#for: INPUT_ID, "Name" }
            input {
                class: "text-input",
                id: INPUT_ID,
                value: "{draft}",
                autofocus: true,
                autocomplete: "off",
                spellcheck: "false",
                aria_invalid: problem().is_some().then_some("true"),
                aria_describedby: if problem().is_some() { "rename-problem rename-help" } else { "rename-help" },
                onmounted: move |_| {
                    document::eval(&format!("document.getElementById({INPUT_ID:?})?.select()"));
                },
                oninput: move |event| {
                    draft.set(event.value());
                    problem.set(None);
                },
                onkeydown: move |event: KeyboardEvent| {
                    if event.key() == Key::Named(NamedKey::Enter) {
                        event.prevent_default();
                        submit.call(());
                    }
                },
            }
            if let Some(found) = problem() {
                p {
                    class: "foot",
                    id: "rename-problem",
                    role: "alert",
                    style: "color:var(--danger)",
                    "{problem_text(found)}"
                }
            }
            p { class: "foot", id: "rename-help",
                "Only the name in Purr changes. The audio file keeps its name."
            }
        }
    }
}
