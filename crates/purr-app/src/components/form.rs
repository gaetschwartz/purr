use dioxus::prelude::*;

const LABEL: &str = "block text-sm font-medium text-gray-700 dark:text-gray-200";
const HINT: &str = "text-xs text-gray-500 dark:text-gray-400";
const FIELD: &str = "w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-teal-600 focus:border-teal-600 disabled:opacity-60";

fn field_id(label: &str) -> String {
    let slug: String = label
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect();
    format!("field-{slug}")
}

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
    let id = field_id(props.label);
    rsx! {
        div { class: "space-y-2 {props.class}",
            div { class: "flex justify-between items-center",
                label { class: LABEL, r#for: "{id}", "{props.label}" }
                span { class: "text-sm text-gray-500 dark:text-gray-300 bg-gray-100 dark:bg-gray-700 px-2 py-1 rounded",
                    "{props.value:.2}"
                }
            }
            if !props.description.is_empty() {
                p { class: HINT, "{props.description}" }
            }
            input {
                id: "{id}",
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
                },
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
    let id = field_id(props.label);
    rsx! {
        div { class: "flex items-start space-x-3 {props.class}",
            button {
                id: "{id}",
                r#type: "button",
                role: "switch",
                aria_checked: "{props.checked}",
                class: format!(
                    "relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-teal-600 focus:ring-offset-2 {}",
                    if props.checked { "bg-teal-600" } else { "bg-gray-200 dark:bg-gray-600" },
                ),
                onclick: move |_| props.onchange.call(!props.checked),
                span {
                    class: format!(
                        "pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out {}",
                        if props.checked { "translate-x-5" } else { "translate-x-0" },
                    ),
                }
            }
            div { class: "flex-1",
                label {
                    class: "text-sm font-medium text-gray-700 dark:text-gray-200 cursor-pointer",
                    r#for: "{id}",
                    "{props.label}"
                }
                if !props.description.is_empty() {
                    p { class: HINT, "{props.description}" }
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
    /// Whether the control rejects input
    #[props(default = false)]
    pub disabled: bool,
    /// Additional CSS classes
    #[props(default = "")]
    pub class: &'static str,
}

/// Option for Select component
#[derive(Clone, PartialEq, Eq)]
pub struct SelectOption {
    pub value: String,
    pub label: String,
    pub disabled: bool,
}

impl SelectOption {
    #[must_use]
    pub fn new(value: impl Into<String>, label: impl Into<String>) -> Self {
        Self {
            value: value.into(),
            label: label.into(),
            disabled: false,
        }
    }
}

/// Reusable Select dropdown component
#[component]
pub fn Select(props: SelectProps) -> Element {
    let id = field_id(props.label);
    rsx! {
        div { class: "space-y-2 {props.class}",
            label { class: LABEL, r#for: "{id}", "{props.label}" }
            if !props.description.is_empty() {
                p { class: HINT, "{props.description}" }
            }
            select {
                id: "{id}",
                class: FIELD,
                value: "{props.value}",
                disabled: props.disabled,
                onchange: move |event| props.onchange.call(event.value()),
                for option in &props.options {
                    option {
                        value: "{option.value}",
                        selected: option.value == props.value,
                        disabled: option.disabled,
                        "{option.label}"
                    }
                }
            }
        }
    }
}

/// Props for the `NumberInput` component
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

/// Reusable `NumberInput` component for numeric settings
#[component]
pub fn NumberInput(props: NumberInputProps) -> Element {
    let id = field_id(props.label);
    rsx! {
        div { class: "space-y-2 {props.class}",
            label { class: LABEL, r#for: "{id}", "{props.label}" }
            if !props.description.is_empty() {
                p { class: HINT, "{props.description}" }
            }
            input {
                id: "{id}",
                r#type: "number",
                class: FIELD,
                min: "{props.min}",
                max: "{props.max}",
                value: "{props.value}",
                oninput: move |event| {
                    if let Ok(value) = event.value().parse::<u32>()
                        && value >= props.min
                        && value <= props.max
                    {
                        props.onchange.call(value);
                    }
                },
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
            div { class: "border-b border-gray-200 dark:border-gray-700 pb-4",
                h2 { class: "text-lg font-medium text-gray-900 dark:text-gray-100", "{props.title}" }
                if !props.description.is_empty() {
                    p { class: "mt-1 text-sm text-gray-600 dark:text-gray-400", "{props.description}" }
                }
            }
            div { class: "space-y-4", {props.children} }
        }
    }
}
