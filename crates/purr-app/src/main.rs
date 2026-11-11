/// Frontend binary for purr-app
use purr_app::App;

fn main() {
    #[cfg(purr_web)]
    purr_wasm::init();

    // Initialize logging capture for desktop builds
    if let Err(e) = purr_app::platform::logging::init_logging_capture() {
        eprintln!("Failed to initialize logging: {e}");
    }

    #[cfg(feature = "desktop")]
    dioxus::LaunchBuilder::new()
        .with_cfg(
            dioxus::desktop::Config::new()
                .with_window(dioxus::desktop::WindowBuilder::new().with_title("Purr")),
        )
        .launch(App);

    #[cfg(not(feature = "desktop"))]
    dioxus::launch(App);
}
