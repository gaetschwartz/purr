use dioxus::prelude::*;

/// Available icon types
#[derive(Clone, Copy, PartialEq)]
pub enum IconType {
    Upload,
    Check,
    BackArrow,
    Error,
    Logo,
    LiveTranscription,
    Settings,
    Logs,
}

/// Props for the Icon component
#[derive(Props, Clone, PartialEq)]
pub struct IconProps {
    /// The type of icon to display
    pub icon_type: IconType,
    /// CSS classes to apply to the icon
    pub class: Option<&'static str>,
}

/// Reusable Icon component that loads SVGs from assets
#[component]
pub fn Icon(props: IconProps) -> Element {
    let (svg_content, default_classes) = match props.icon_type {
        IconType::Upload => (
            include_str!("../../assets/icons/upload.svg"),
            "w-8 h-8 text-teal-600",
        ),
        IconType::Check => (
            include_str!("../../assets/icons/check.svg"),
            "w-8 h-8 text-green-600",
        ),
        IconType::BackArrow => (include_str!("../../assets/icons/back-arrow.svg"), "w-4 h-4"),
        IconType::Error => (
            include_str!("../../assets/icons/error.svg"),
            "w-10 h-10 text-red-600",
        ),
        IconType::Logo => (
            include_str!("../../assets/icons/logo.svg"),
            "w-8 h-8 text-teal-600",
        ),
        IconType::LiveTranscription => (
            include_str!("../../assets/icons/live-transcription.svg"),
            "w-5 h-5 text-blue-600",
        ),
        IconType::Settings => (
            include_str!("../../assets/icons/settings.svg"),
            "w-5 h-5 text-gray-600",
        ),
        IconType::Logs => (
            include_str!("../../assets/icons/logs.svg"),
            "w-5 h-5 text-gray-600",
        ),
    };

    let classes = props.class.unwrap_or(default_classes);

    rsx! {
        div { class: "{classes}", dangerous_inner_html: "{svg_content}" }
    }
}
