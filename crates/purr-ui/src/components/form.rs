use dioxus::prelude::*;

/// Props for the Slider component
#[derive(Props, Clone, PartialEq)]
pub struct SliderProps {
    /// Current value
    pub value: f64,
    /// Minimum value
    #[props(default = 0.0)]
    pub min: f64,
    /// Maximum value
    #[props(default = 1.0)]
    pub max: f64,
    /// Step increment
    #[props(default = 0.01)]
    pub step: f64,
    /// Label text
    pub label: &'static str,
    /// Description text
    #[props(default = "")]
    pub description: &'static str,
    /// Callback when value changes
    pub onchange: EventHandler<f64>,
    /// Additional CSS classes
    #[props(default = "")]
    pub class: &'static str,
}

/// Reusable Slider component for numeric inputs
#[component]
pub fn Slider(props: SliderProps) -> Element {
    rsx! {
        div { class: "space-y-2 {props.class}",
            div { class: "flex justify-between items-center",
                label { class: "text-sm font-medium text-gray-700", "{props.label}" }
                span { class: "text-sm text-gray-500 bg-gray-100 px-2 py-1 rounded", "{props.value:.2}" }
            }
            if !props.description.is_empty() {
                p { class: "text-xs text-gray-500", "{props.description}" }
            }
            input {
                r#type: "range",
                class: "w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer slider",
                min: "{props.min}",
                max: "{props.max}",
                step: "{props.step}",
                value: "{props.value}",
                oninput: move |event| {
                    if let Ok(value) = event.value().parse::<f64>() {
                        props.onchange.call(value);
                    }
                }
            }
        }
    }
}

/// Props for the Toggle component
#[derive(Props, Clone, PartialEq)]
pub struct ToggleProps {
    /// Current checked state
    pub checked: bool,
    /// Label text
    pub label: &'static str,
    /// Description text
    #[props(default = "")]
    pub description: &'static str,
    /// Callback when toggled
    pub onchange: EventHandler<bool>,
    /// Additional CSS classes
    #[props(default = "")]
    pub class: &'static str,
}

/// Reusable Toggle component for boolean settings
#[component]
pub fn Toggle(props: ToggleProps) -> Element {
    rsx! {
        div { class: "flex items-start space-x-3 {props.class}",
            button {
                class: format!(
                    "relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-teal-600 focus:ring-offset-2 {}",
                    if props.checked { "bg-teal-600" } else { "bg-gray-200" }
                ),
                onclick: move |_| props.onchange.call(!props.checked),
                span {
                    class: format!(
                        "pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out {}",
                        if props.checked { "translate-x-5" } else { "translate-x-0" }
                    )
                }
            }
            div { class: "flex-1",
                label { class: "text-sm font-medium text-gray-700 cursor-pointer",
                    onclick: move |_| props.onchange.call(!props.checked),
                    "{props.label}"
                }
                if !props.description.is_empty() {
                    p { class: "text-xs text-gray-500", "{props.description}" }
                }
            }
        }
    }
}

/// Props for the Select component
#[derive(Props, Clone, PartialEq)]
pub struct SelectProps {
    /// Current selected value
    pub value: String,
    /// Available options
    pub options: Vec<SelectOption>,
    /// Label text
    pub label: &'static str,
    /// Description text
    #[props(default = "")]
    pub description: &'static str,
    /// Callback when selection changes
    pub onchange: EventHandler<String>,
    /// Additional CSS classes
    #[props(default = "")]
    pub class: &'static str,
}

/// Option for Select component
#[derive(Clone, PartialEq)]
pub struct SelectOption {
    pub value: String,
    pub label: String,
}

/// Reusable Select dropdown component
#[component]
pub fn Select(props: SelectProps) -> Element {
    rsx! {
        div { class: "space-y-2 {props.class}",
            label { class: "block text-sm font-medium text-gray-700", "{props.label}" }
            if !props.description.is_empty() {
                p { class: "text-xs text-gray-500", "{props.description}" }
            }
            select {
                class: "w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm bg-white focus:outline-none focus:ring-2 focus:ring-teal-600 focus:border-teal-600",
                value: "{props.value}",
                onchange: move |event| props.onchange.call(event.value()),
                for option in &props.options {
                    option { value: "{option.value}", selected: option.value == props.value, "{option.label}" }
                }
            }
        }
    }
}

/// Props for the NumberInput component
#[derive(Props, Clone, PartialEq)]
pub struct NumberInputProps {
    /// Current value
    pub value: u32,
    /// Minimum value
    #[props(default = 0)]
    pub min: u32,
    /// Maximum value
    #[props(default = 10000)]
    pub max: u32,
    /// Label text
    pub label: &'static str,
    /// Description text
    #[props(default = "")]
    pub description: &'static str,
    /// Callback when value changes
    pub onchange: EventHandler<u32>,
    /// Additional CSS classes
    #[props(default = "")]
    pub class: &'static str,
}

/// Reusable NumberInput component for numeric settings
#[component]
pub fn NumberInput(props: NumberInputProps) -> Element {
    rsx! {
        div { class: "space-y-2 {props.class}",
            label { class: "block text-sm font-medium text-gray-700", "{props.label}" }
            if !props.description.is_empty() {
                p { class: "text-xs text-gray-500", "{props.description}" }
            }
            input {
                r#type: "number",
                class: "w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm bg-white focus:outline-none focus:ring-2 focus:ring-teal-600 focus:border-teal-600",
                min: "{props.min}",
                max: "{props.max}",
                value: "{props.value}",
                oninput: move |event| {
                    if let Ok(value) = event.value().parse::<u32>() {
                        if value >= props.min && value <= props.max {
                            props.onchange.call(value);
                        }
                    }
                }
            }
        }
    }
}

/// Props for the Section component
#[derive(Props, Clone, PartialEq)]
pub struct SectionProps {
    /// Section title
    pub title: &'static str,
    /// Section description
    #[props(default = "")]
    pub description: &'static str,
    /// Section content
    pub children: Element,
    /// Additional CSS classes
    #[props(default = "")]
    pub class: &'static str,
}

/// Reusable Section component for organizing settings
#[component]
pub fn Section(props: SectionProps) -> Element {
    rsx! {
        div { class: "space-y-6 {props.class}",
            div { class: "border-b border-gray-200 pb-4",
                h3 { class: "text-lg font-medium text-gray-900", "{props.title}" }
                if !props.description.is_empty() {
                    p { class: "mt-1 text-sm text-gray-600", "{props.description}" }
                }
            }
            div { class: "space-y-4", {props.children} }
        }
    }
}