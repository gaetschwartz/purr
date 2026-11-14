use super::family::ModelLine;
use purr_common::model::family::ModelFamily;

/// What the lists say about who made a model, for a family whose licence asks for credit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Credit {
    /// `parakeet-tdt-0.6b-v3 by NVIDIA, licensed CC BY 4.0.`
    pub line: String,
    /// What the downloaded files change about the original, when they change anything.
    pub changes: Option<&'static str>,
    pub license_url: &'static str,
    pub origin_url: &'static str,
}

/// The credit `family` is shown with; `None` when its licence asks for none.
#[must_use]
pub fn credit(family: ModelFamily) -> Option<Credit> {
    let attribution = family.attribution();
    attribution.license.requires_attribution().then(|| Credit {
        line: format!(
            "{} by {}, licensed {}.",
            attribution.work,
            attribution.author,
            attribution.license.label()
        ),
        changes: attribution.changes,
        license_url: attribution.license.url(),
        origin_url: attribution.origin_url,
    })
}

/// What the picker titles a line of models with: its name, then its author and licence when the
/// licence asks for credit.
#[must_use]
pub fn heading(family: ModelFamily) -> String {
    let line = ModelLine::of(family).label();
    let attribution = family.attribution();
    if attribution.license.requires_attribution() {
        format!(
            "{line} · {} · {}",
            attribution.author,
            attribution.license.label()
        )
    } else {
        line.to_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use purr_common::model::family::{ParakeetVariant, WhisperSize};

    #[test]
    fn a_family_under_creative_commons_attribution_is_credited_with_its_changes() {
        let credit = credit(ModelFamily::Parakeet(ParakeetVariant::TdtV3)).unwrap();
        assert_eq!(
            credit.line,
            "parakeet-tdt-0.6b-v3 by NVIDIA, licensed CC BY 4.0."
        );
        assert_eq!(
            credit.changes,
            ModelFamily::Parakeet(ParakeetVariant::TdtV3)
                .attribution()
                .changes
        );
        assert_eq!(
            credit.license_url,
            "https://creativecommons.org/licenses/by/4.0/"
        );
    }

    #[test]
    fn a_family_whose_licence_asks_for_no_credit_shows_none() {
        assert_eq!(credit(ModelFamily::Whisper(WhisperSize::Tiny)), None);
    }

    #[test]
    fn a_picker_heading_names_the_line_and_adds_the_credit_when_one_is_due() {
        assert_eq!(heading(ModelFamily::Whisper(WhisperSize::Base)), "Whisper");
        assert_eq!(
            heading(ModelFamily::Parakeet(ParakeetVariant::TdtV3)),
            "Parakeet · NVIDIA · CC BY 4.0"
        );
    }
}
