/// Frontend binary for purr-app
use purr_app::App;

#[cfg(feature = "desktop")]
fn desktop_config() -> dioxus::desktop::Config {
    use dioxus::desktop::{
        Config, LogicalSize, WindowBuilder, icon_from_memory, tao::window::Icon,
    };

    let window = WindowBuilder::new()
        .with_title("Purr")
        .with_inner_size(LogicalSize::new(1180.0, 800.0))
        .with_min_inner_size(LogicalSize::new(360.0, 480.0));
    let config = Config::new().with_window(window);
    match icon_from_memory::<Icon>(include_bytes!("../../../res/icons/icon.png")) {
        Ok(icon) => config.with_icon(icon),
        Err(error) => {
            tracing::warn!("Could not load the window icon: {error}");
            config
        }
    }
}

fn main() {
    #[cfg(purr_web)]
    purr_wasm::init();

    // Initialize logging capture for desktop builds
    if let Err(e) = purr_app::platform::logging::init_logging_capture() {
        eprintln!("Failed to initialize logging: {e}");
    }

    #[cfg(feature = "desktop")]
    dioxus::LaunchBuilder::new()
        .with_cfg(desktop_config())
        .launch(App);

    #[cfg(not(feature = "desktop"))]
    dioxus::launch(App);
}
