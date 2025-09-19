use dioxus::prelude::*;

/// The Transcription page component that displays the transcription functionality
/// for an uploaded audio file
#[component]
pub fn Transcription(file_path: String) -> Element {
    let navigator = use_navigator();

    let handle_back = move |_| {
        navigator.go_back();
    };

    rsx! {
        div { class: "flex flex-col h-screen bg-gray-50",
            // Header with file info and back button
            div { class: "bg-white shadow-sm border-b border-gray-200 px-6 py-4",
                div { class: "flex items-center justify-between max-w-4xl mx-auto",
                    div { class: "flex items-center space-x-4",
                        button {
                            class: "flex items-center px-3 py-2 text-sm font-medium text-gray-600 hover:text-teal-600 transition-colors",
                            onclick: handle_back,
                            svg {
                                class: "w-4 h-4 mr-2",
                                fill: "none",
                                stroke: "currentColor",
                                view_box: "0 0 24 24",
                                path {
                                    stroke_linecap: "round",
                                    stroke_linejoin: "round",
                                    stroke_width: "2",
                                    d: "M15 19l-7-7 7-7",
                                }
                            }
                            "Back"
                        }
                        h1 { class: "text-xl font-semibold text-gray-900", "Transcription" }
                    }
                }
            }

            // Main content area
            div { class: "flex-1 max-w-4xl mx-auto w-full px-6 py-8",
                // Transcription status and content
                div { class: "bg-white rounded-lg shadow-sm border border-gray-200 p-6",
                    div { class: "text-center py-12",
                        // Loading animation
                        div { class: "flex justify-center mb-4",
                            div { class: "animate-spin rounded-full h-12 w-12 border-b-2 border-teal-500" }
                        }
                        h3 { class: "text-lg font-medium text-gray-900 mb-2",
                            "Transcribing: {file_path}"
                        }
                        p { class: "text-gray-600 mb-6",
                            "Please wait while we process your audio file..."
                        }

                        // Progress indicator (placeholder)
                        div { class: "max-w-md mx-auto",
                            div { class: "bg-gray-200 rounded-full h-2",
                                div {
                                    class: "bg-teal-500 h-2 rounded-full transition-all duration-300",
                                    style: "width: 45%",
                                }
                            }
                            p { class: "text-sm text-gray-500 mt-2", "Processing... 45%" }
                        }

                        // Placeholder for transcription results
                        div { class: "mt-8 p-4 bg-gray-50 rounded-lg border-2 border-dashed border-gray-300",
                            p { class: "text-gray-500 italic",
                                "Transcription results will appear here once processing is complete."
                            }
                        }
                    }
                }
            }
        }
    }
}
