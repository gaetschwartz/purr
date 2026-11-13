use dioxus::{prelude::*, router::root_router};

/// Gives the components below it the app's router.
///
/// Overlays render under the layer outlet, which sits beside the router rather than below it, so
/// anything inside that needs `use_navigator` or a `Link` has to be wrapped in this.
#[component]
pub fn RouterBridge(children: Element) -> Element {
    use_hook(|| match root_router() {
        Some(router) => {
            provide_context(router);
        }
        None => tracing::warn!("No router to hand to the overlay content"),
    });
    children
}
