use dioxus::prelude::*;
use super::{Icon, IconType};

/// Button variants for different styles
#[derive(Clone, Copy, PartialEq)]
pub enum ButtonVariant {
    Primary,
    Secondary,
    Danger,
    Ghost,
}

/// Button sizes
#[derive(Clone, Copy, PartialEq)]
pub enum ButtonSize {
    Small,
    Medium,
    Large,
}

/// Props for the Button component
#[derive(Props, Clone, PartialEq)]
pub struct ButtonProps {
    /// Button text
    pub children: Element,
    /// Click handler
    #[props(optional)]
    pub onclick: Option<EventHandler<MouseEvent>>,
    /// Button variant
    #[props(default = ButtonVariant::Primary)]
    pub variant: ButtonVariant,
    /// Button size
    #[props(default = ButtonSize::Medium)]
    pub size: ButtonSize,
    /// Optional icon to display
    #[props(optional)]
    pub icon: Option<IconType>,
    /// Whether the button is disabled
    #[props(default = false)]
    pub disabled: bool,
    /// Additional CSS classes
    #[props(default = "")]
    pub class: &'static str,
}

/// Reusable Button component with consistent styling
#[component]
pub fn Button(props: ButtonProps) -> Element {
    let base_classes = "inline-flex items-center justify-center font-medium rounded-lg transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-offset-2";
    
    let variant_classes = match props.variant {
        ButtonVariant::Primary => "bg-teal-500 text-white hover:bg-teal-600 focus:ring-teal-500 shadow-md hover:shadow-lg",
        ButtonVariant::Secondary => "bg-gray-100 text-gray-700 hover:bg-gray-200 focus:ring-gray-500",
        ButtonVariant::Danger => "bg-red-500 text-white hover:bg-red-600 focus:ring-red-500 shadow-md hover:shadow-lg",
        ButtonVariant::Ghost => "text-gray-600 hover:text-teal-600 hover:bg-teal-50 focus:ring-teal-500",
    };
    
    let size_classes = match props.size {
        ButtonSize::Small => "px-3 py-1.5 text-sm",
        ButtonSize::Medium => "px-4 py-2 text-sm",
        ButtonSize::Large => "px-6 py-3 text-base",
    };
    
    let disabled_classes = if props.disabled {
        "opacity-50 cursor-not-allowed"
    } else {
        "cursor-pointer"
    };
    
    let full_classes = format!("{} {} {} {} {}", base_classes, variant_classes, size_classes, disabled_classes, props.class);
    
    rsx! {
        button {
            class: "{full_classes}",
            disabled: props.disabled,
            onclick: move |evt| {
                if !props.disabled {
                    if let Some(handler) = &props.onclick {
                        handler.call(evt);
                    }
                }
            },
            
            if let Some(icon_type) = props.icon {
                Icon {
                    icon_type,
                    class: "w-4 h-4 mr-2"
                }
            }
            
            {props.children}
        }
    }
}