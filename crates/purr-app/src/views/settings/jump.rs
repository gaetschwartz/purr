use super::sections::SettingsSection;
use crate::main_app::Route;
use dioxus::{prelude::*, router::Navigator};

fn script(anchor: &str) -> String {
    format!(
        "let tries = 0; const timer = setInterval(() => {{ const target = document.getElementById({anchor:?}); tries += 1; if (target || tries > 40) {{ clearInterval(timer); target?.scrollIntoView({{ block: 'start' }}); }} }}, 50);"
    )
}

/// Opens Settings scrolled to `section`, waiting for the page to render it.
pub fn open_section(navigator: Navigator, section: SettingsSection) {
    navigator.push(Route::Settings {});
    document::eval(&script(section.anchor()));
}
