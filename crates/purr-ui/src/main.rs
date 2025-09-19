/// Frontend binary for purr-ui
/// This is the client-side WASM application

use purr_ui::App;

fn main() {
    // Launch client-side only (WASM)
    dioxus::launch(App);
}
