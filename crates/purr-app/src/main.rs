/// Frontend binary for purr-app
use purr_app::App;

fn main() {
    #[cfg(all(feature = "web", target_arch = "wasm32", not(feature = "desktop")))]
    purr_wasm::init();

    // Initialize logging capture for desktop builds
    if let Err(e) = purr_app::platform::logging::init_logging_capture() {
        eprintln!("Failed to initialize logging: {e}");
    }

    dioxus::launch(App);
}
