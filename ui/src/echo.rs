use dioxus::prelude::*;

#[cfg(feature = "api")]
use api;

const ECHO_CSS: Asset = asset!("/assets/styling/echo.css");

/// Echo component that demonstrates fullstack server functions.
#[component]
pub fn Echo() -> Element {
    let mut response = use_signal(|| String::new());

    rsx! {
        document::Link { rel: "stylesheet", href: ECHO_CSS }
        div {
            id: "echo",
            h4 { "ServerFn Echo" }
            input {
                placeholder: "Type here to echo...",
                oninput:  move |event| async move {
                    #[cfg(feature = "api")]
                    {
                        let data = api::echo(event.value()).await.unwrap();
                        response.set(data);
                    }
                    #[cfg(not(feature = "api"))]
                    {
                        response.set(format!("Echo: {}", event.value()));
                    }
                },
            }

            if !response().is_empty() {
                p {
                    "Server echoed: "
                    i { "{response}" }
                }
            }
        }
    }
}
