use dioxus::prelude::*;

const ASK_PLATFORM: &str =
    "dioxus.send(/Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent));";

/// How shortcut hints name Cmd or Ctrl on this computer: `⌘` on Apple systems, `Ctrl` elsewhere.
pub fn use_primary_key_label() -> ReadSignal<&'static str> {
    let mut label = use_signal(|| {
        if cfg!(target_os = "macos") {
            "⌘"
        } else {
            "Ctrl"
        }
    });
    use_future(move || async move {
        match document::eval(ASK_PLATFORM).recv::<bool>().await {
            Ok(true) => label.set("⌘"),
            Ok(false) => label.set("Ctrl"),
            Err(error) => tracing::debug!("Could not tell the keyboard platform: {error}"),
        }
    });
    label.into()
}
