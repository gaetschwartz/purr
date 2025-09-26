/// Frontend binary for purr-ui
use purr_ui::App;

fn main() {
    // Initialize logging capture for desktop builds
    if let Err(e) = purr_ui::platform::logging::init_logging_capture() {
        eprintln!("Failed to initialize logging: {e}");
    }

    dioxus::launch(App);
}
