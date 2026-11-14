use dioxus::prelude::*;
use purr_common::model::family::ModelFamily;

/// The credit a model family's licence requires wherever the app is distributed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelCredit {
    pub title: &'static str,
    /// `parakeet-tdt-0.6b-v3 by NVIDIA, licensed CC BY 4.0.`
    pub line: String,
    /// What the downloaded files change about the original, when they change anything.
    pub changes: Option<&'static str>,
    pub origin_url: &'static str,
    pub license_url: &'static str,
}

/// The credit of every model family whose licence asks for one.
#[must_use]
pub fn model_credits() -> Vec<ModelCredit> {
    ModelFamily::all()
        .filter_map(|family| {
            let attribution = family.attribution();
            attribution
                .license
                .requires_attribution()
                .then(|| ModelCredit {
                    title: family.label(),
                    line: format!(
                        "{} by {}, licensed {}.",
                        attribution.work,
                        attribution.author,
                        attribution.license.label()
                    ),
                    changes: attribution.changes,
                    origin_url: attribution.origin_url,
                    license_url: attribution.license.url(),
                })
        })
        .collect()
}

#[component]
pub fn ModelCredits() -> Element {
    let credits = use_hook(model_credits);
    if credits.is_empty() {
        return rsx! {};
    }
    rsx! {
        section {
            class: "card surface",
            style: "display:flex;flex-direction:column;gap:14px",
            aria_label: "Models",
            h2 { "Models" }
            for credit in credits {
                div { key: "{credit.title}",
                    p { "{credit.title}" }
                    p { class: "foot", "{credit.line}" }
                    if let Some(changes) = credit.changes {
                        p { class: "foot", "{changes}" }
                    }
                    p { class: "foot", "Source: {credit.origin_url}" }
                    p { class: "foot", "Licence: {credit.license_url}" }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use purr_common::model::family::ParakeetVariant;

    #[test]
    fn only_families_whose_licence_asks_for_credit_are_listed() {
        let credits = model_credits();
        assert_eq!(credits.len(), 1);
        assert_eq!(credits[0].title, "Parakeet TDT v3");
    }

    #[test]
    fn a_credit_names_the_work_its_author_and_the_licence() {
        let credit = &model_credits()[0];
        assert_eq!(
            credit.line,
            "parakeet-tdt-0.6b-v3 by NVIDIA, licensed CC BY 4.0."
        );
        assert_eq!(
            credit.license_url,
            "https://creativecommons.org/licenses/by/4.0/"
        );
    }

    #[test]
    fn a_credit_carries_what_the_files_change_and_where_the_original_is() {
        let attribution = ModelFamily::Parakeet(ParakeetVariant::TdtV3).attribution();
        let credit = &model_credits()[0];
        assert_eq!(credit.changes, attribution.changes);
        assert!(credit.changes.is_some());
        assert_eq!(credit.origin_url, attribution.origin_url);
    }
}
