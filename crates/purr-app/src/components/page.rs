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

/// How a [`Page`] shows its title.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TitleStyle {
    /// A large heading in a top bar, with the subtitle, eyebrow and actions.
    #[default]
    Large,
    /// A heading for screen readers only; the page starts with its content.
    Hidden,
}

/// The frame of every view: document title, a title heading (focused when the view opens),
/// optional subtitle and trailing actions, then the content in an 880 px column (1040 px when
/// `wide`).
#[component]
pub fn Page(
    #[props(into)] title: String,
    #[props(into)] subtitle: Option<String>,
    #[props(default)] wide: bool,
    #[props(default)] title_style: TitleStyle,
    /// Classes added to the content column.
    #[props(into, default)]
    class: String,
    /// A small row above the title, such as a back link.
    eyebrow: Option<Element>,
    actions: Option<Element>,
    children: Element,
) -> Element {
    let mut column = String::from(if wide { "content wide" } else { "content" });
    if !class.is_empty() {
        column.push(' ');
        column.push_str(&class);
    }
    let focus = move |event: MountedEvent| async move {
        if let Err(error) = event.data().set_focus(true).await {
            tracing::debug!("Could not focus the page heading: {error}");
        }
    };
    rsx! {
        document::Title { {document_title(&title)} }
        div { class: column,
            match title_style {
                TitleStyle::Large => rsx! {
                    div { class: "topbar",
                        div { class: "titles",
                            {eyebrow}
                            h1 { class: "lt", tabindex: "-1", onmounted: focus, "{title}" }
                            if let Some(subtitle) = &subtitle {
                                p { class: "sub", "{subtitle}" }
                            }
                        }
                        {actions}
                    }
                },
                TitleStyle::Hidden => rsx! {
                    h1 {
                        class: "sr",
                        dir: "auto",
                        tabindex: "-1",
                        onmounted: focus,
                        "{title}"
                    }
                },
            }
            {children}
        }
    }
}
