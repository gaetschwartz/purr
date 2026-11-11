use super::sections::SettingsSection;
use crate::platform::IS_WEB;
use dioxus::prelude::*;

fn jump_script(anchor: &str) -> String {
    format!(
        "const section = document.getElementById({anchor:?}); if (section) {{ section.scrollIntoView({{ block: 'start' }}); const heading = section.querySelector('h2'); if (heading) {{ heading.tabIndex = -1; heading.focus({{ preventScroll: true }}); }} }}"
    )
}

/// Pills at the top of the page that jump to a group of settings.
#[component]
pub fn SectionNav() -> Element {
    rsx! {
        nav { class: "sect-nav", aria_label: "Settings sections",
            for section in SettingsSection::visible(IS_WEB) {
                button {
                    key: "{section.anchor()}",
                    id: "j-{section.anchor()}",
                    r#type: "button",
                    onclick: move |_| {
                        document::eval(&jump_script(section.anchor()));
                    },
                    "{section.label()}"
                }
            }
        }
    }
}
