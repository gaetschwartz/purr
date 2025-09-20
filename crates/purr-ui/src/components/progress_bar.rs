use dioxus::prelude::*;

/// Props for the ProgressBar component
#[derive(Props, Clone, PartialEq)]
pub struct ProgressBarProps {
    /// Progress value (0-100)
    pub value: f32,
    /// Label to display above the progress bar
    #[props(optional)]
    pub label: Option<String>,
    /// Whether to show the percentage text
    #[props(default = true)]
    pub show_percentage: bool,
    /// Additional CSS classes
    #[props(default = "")]
    pub class: &'static str,
}

/// Progress bar component for showing upload/processing progress
#[component]
pub fn ProgressBar(props: ProgressBarProps) -> Element {
    let clamped_value = props.value.clamp(0.0, 100.0);

    rsx! {
        div { class: "w-full max-w-xs mx-auto {props.class}",

            if let Some(label) = &props.label {
                p { class: "text-sm font-medium text-gray-600 mb-2", "{label}" }
            }

            div { class: "bg-gray-200 rounded-full h-2 mb-2",
                div {
                    class: "bg-teal-500 h-2 rounded-full transition-all duration-300",
                    style: "width: {clamped_value}%",
                }
            }

            if props.show_percentage {
                p { class: "text-sm text-gray-600 text-center", "{clamped_value:.1}%" }
            }
        }
    }
}
