use dioxus::prelude::*;

/// Card variants for different styles
#[allow(dead_code)]
#[derive(Clone, Copy, PartialEq)]
pub enum CardVariant {
    Default,
    Success,
    Warning,
    Error,
    Info,
}

/// Props for the Card component
#[derive(Props, Clone, PartialEq)]
pub struct CardProps {
    /// Card content
    pub children: Element,
    /// Card variant
    #[props(default = CardVariant::Default)]
    pub variant: CardVariant,
    /// Additional CSS classes
    #[props(default = "")]
    pub class: &'static str,
    /// Whether the card has padding
    #[props(default = true)]
    pub padding: bool,
}

/// Reusable Card component for consistent containers
#[component]
pub fn Card(props: CardProps) -> Element {
    let base_classes = "rounded-xl shadow-lg border";

    let variant_classes = match props.variant {
        CardVariant::Default => "bg-white/90 backdrop-blur-sm border-gray-200",
        CardVariant::Success => "bg-gradient-to-r from-green-50 to-emerald-50 border-green-200",
        CardVariant::Warning => "bg-gradient-to-r from-yellow-50 to-amber-50 border-yellow-200",
        CardVariant::Error => "bg-red-50 border-red-200",
        CardVariant::Info => "bg-gradient-to-r from-blue-50 to-indigo-50 border-blue-200",
    };

    let padding_classes = if props.padding { "p-8" } else { "" };

    let full_classes = format!(
        "{} {} {} {}",
        base_classes, variant_classes, padding_classes, props.class
    );

    rsx! {
        div { class: "{full_classes}", {props.children} }
    }
}
