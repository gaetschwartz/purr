use dioxus::prelude::*;

/// How much of the work a [`ProgressBar`] shows as done.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Progress {
    /// A fraction between 0 and 1.
    Determinate(f64),
    /// Work whose length is unknown; the bar sweeps.
    Indeterminate,
}

/// Colour of the filled part of a [`ProgressBar`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ProgressTone {
    #[default]
    Accent,
    Warn,
    Bad,
}

/// The rounded progress track (`role="progressbar"`); `thin` draws the 4 px variant.
#[component]
pub fn ProgressBar(
    progress: Progress,
    #[props(into)] label: String,
    #[props(into)] value_text: Option<String>,
    #[props(default)] tone: ProgressTone,
    #[props(default)] thin: bool,
) -> Element {
    let mut class = String::from("bar");
    if thin {
        class.push_str(" thin");
    }
    match tone {
        ProgressTone::Accent => {}
        ProgressTone::Warn => class.push_str(" warn"),
        ProgressTone::Bad => class.push_str(" bad"),
    }
    let (percent, now) = match progress {
        Progress::Determinate(fraction) => {
            let percent = (fraction * 100.0).clamp(0.0, 100.0);
            (percent, Some(format!("{percent:.0}")))
        }
        Progress::Indeterminate => {
            class.push_str(" indet");
            (35.0, None)
        }
    };
    rsx! {
        div {
            class,
            role: "progressbar",
            aria_label: label,
            aria_valuemin: "0",
            aria_valuemax: "100",
            aria_valuenow: now,
            aria_valuetext: value_text,
            i { style: "width:{percent}%" }
        }
    }
}
