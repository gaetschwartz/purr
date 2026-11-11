use crate::platform::IS_WEB;
use dioxus::prelude::*;

/// The document title for a view: "Transcribe · Purr" in a browser, "Purr" for the desktop window.
#[must_use]
pub fn document_title(view_title: &str) -> String {
    if IS_WEB {
        format!("{view_title} · Purr")
    } else {
        "Purr".to_owned()
    }
}

/// The frame of every view: document title, large title (focused when the view opens), optional
/// subtitle and trailing actions, then the content in an 880 px column (1040 px when `wide`).
#[component]
pub fn Page(
    #[props(into)] title: String,
    #[props(into)] subtitle: Option<String>,
    #[props(default)] wide: bool,
    /// A small row above the title, such as a back link.
    eyebrow: Option<Element>,
    /// The title is a file name that may be long and unbroken, so it wraps anywhere.
    #[props(default)]
    file_title: bool,
    actions: Option<Element>,
    children: Element,
) -> Element {
    let class = if wide { "content wide" } else { "content" };
    rsx! {
        document::Title { {document_title(&title)} }
        div { class,
            div { class: "topbar",
                div { class: "titles",
                    {eyebrow}
                    h1 {
                        class: if file_title { "lt file" } else { "lt" },
                        title: file_title.then(|| title.clone()),
                        tabindex: "-1",
                        onmounted: move |event: MountedEvent| async move {
                            if let Err(error) = event.data().set_focus(true).await {
                                tracing::debug!("Could not focus the page heading: {error}");
                            }
                        },
                        "{title}"
                    }
                    if let Some(subtitle) = &subtitle {
                        p { class: "sub", "{subtitle}" }
                    }
                }
                {actions}
            }
            {children}
        }
    }
}
