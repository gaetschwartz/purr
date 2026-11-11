use crate::model_catalog::family::Ratings;
use dioxus::prelude::*;

const DOTS: u8 = 5;

#[component]
fn Meter(#[props(into)] name: String, filled: u8) -> Element {
    rsx! {
        span { class: "meter",
            "{name}"
            span {
                class: "dots",
                role: "img",
                aria_label: "{name} {filled} of {DOTS}",
                for position in 1..=DOTS {
                    i { key: "{position}", class: if position <= filled { "on" } }
                }
            }
        }
    }
}

/// The speed and accuracy of a model as two rows of five dots.
#[component]
pub fn ModelMeters(ratings: Ratings) -> Element {
    rsx! {
        span { class: "meters",
            Meter { name: "Speed", filled: ratings.speed }
            Meter { name: "Accuracy", filled: ratings.accuracy }
        }
    }
}
