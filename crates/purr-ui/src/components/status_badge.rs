use dioxus::prelude::*;

/// Status types for different states
#[derive(Clone, Copy, PartialEq)]
pub enum StatusType {
    Processing,
    Success,
    Error,
    Warning,
    Info,
}

/// Props for the StatusBadge component
#[derive(Props, Clone, PartialEq)]
pub struct StatusBadgeProps {
    /// Badge text
    pub text: String,
    /// Status type
    pub status: StatusType,
    /// Additional CSS classes
    #[props(default = "")]
    pub class: &'static str,
}

/// Status badge component for displaying various states
#[component]
pub fn StatusBadge(props: StatusBadgeProps) -> Element {
    let base_classes = "inline-flex items-center px-4 py-2 rounded-full text-sm font-medium";
    
    let status_classes = match props.status {
        StatusType::Processing => "bg-teal-100 text-teal-800",
        StatusType::Success => "bg-green-100 text-green-800",
        StatusType::Error => "bg-red-100 text-red-800",
        StatusType::Warning => "bg-yellow-100 text-yellow-800",
        StatusType::Info => "bg-blue-100 text-blue-800",
    };
    
    let full_classes = format!("{} {} {}", base_classes, status_classes, props.class);
    
    rsx! {
        div {
            class: "{full_classes}",
            "{props.text}"
        }
    }
}