use dioxus::prelude::*;

/// Where `value` sits between `min` and `max`, as a percentage for the filled track.
fn fill_percent(value: f64, min: f64, max: f64) -> f64 {
    if max <= min {
        return 0.0;
    }
    ((value - min) / (max - min) * 100.0).clamp(0.0, 100.0)
}

/// A range slider with a 5 px track and a 24 px knob; `ends` shows the limits beside it.
#[component]
pub fn Slider(
    value: f64,
    min: f64,
    max: f64,
    step: f64,
    #[props(into)] label: Option<String>,
    #[props(into)] labelled_by: Option<String>,
    #[props(into)] value_text: Option<String>,
    ends: Option<(String, String)>,
    #[props(default)] disabled: bool,
    onchange: EventHandler<f64>,
    #[props(extends = GlobalAttributes)] attributes: Vec<Attribute>,
) -> Element {
    let percent = fill_percent(value, min, max);
    let input = rsx! {
        input {
            class: "slider",
            r#type: "range",
            min: "{min}",
            max: "{max}",
            step: "{step}",
            value: "{value}",
            style: "--p:{percent}%",
            aria_label: label,
            aria_labelledby: labelled_by,
            aria_valuetext: value_text,
            disabled,
            oninput: move |event| {
                match event.value().parse::<f64>() {
                    Ok(parsed) => onchange.call(parsed),
                    Err(error) => tracing::warn!("The slider sent an unreadable value: {error}"),
                }
            },
            ..attributes,
        }
    };
    match ends {
        Some((low, high)) => rsx! {
            div { class: "slider-row",
                span { class: "end", aria_hidden: "true", "{low}" }
                {input}
                span { class: "end", aria_hidden: "true", "{high}" }
            }
        },
        None => input,
    }
}

#[cfg(test)]
mod tests {
    use super::fill_percent;

    #[test]
    fn the_track_fill_follows_the_value() {
        assert!((fill_percent(0.3, 0.0, 1.0) - 30.0).abs() < 1e-9);
        assert!((fill_percent(5.0, 0.0, 1.0) - 100.0).abs() < 1e-9);
        assert!(fill_percent(1.0, 1.0, 1.0).abs() < 1e-9);
    }
}
