use crate::views::DragDropZone;
use dioxus::prelude::*;

/// The Home page component that will be rendered when the current route is `[Route::Home]`
/// Contains a drop zone to upload a file, if clicked, will open a file dialog
/// If a file is uploaded, it will be sent to the backend to be processed
/// the drop zone should be surrounded by a dashed border
#[component]
pub fn Home() -> Element {
    rsx! {
        DragDropZone {}
    }
}
