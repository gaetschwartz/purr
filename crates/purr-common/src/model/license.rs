/// The terms a model is released under.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelLicense {
    Mit,
    Apache2,
    CcBy4,
}

impl ModelLicense {
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Mit => "MIT",
            Self::Apache2 => "Apache 2.0",
            Self::CcBy4 => "CC BY 4.0",
        }
    }

    /// Whether the licence makes whoever shows the model credit its author where it is offered.
    #[must_use]
    pub const fn requires_attribution(self) -> bool {
        match self {
            Self::Mit | Self::Apache2 => false,
            Self::CcBy4 => true,
        }
    }

    #[must_use]
    pub const fn url(self) -> &'static str {
        match self {
            Self::Mit => "https://opensource.org/license/mit",
            Self::Apache2 => "https://www.apache.org/licenses/LICENSE-2.0",
            Self::CcBy4 => "https://creativecommons.org/licenses/by/4.0/",
        }
    }
}

/// Who made a model, under which terms, and what was done to it before Purr runs it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Attribution {
    pub license: ModelLicense,
    pub author: &'static str,
    pub work: &'static str,
    pub origin_url: &'static str,
    /// What the files Purr downloads change about the original, when they change anything
    pub changes: Option<&'static str>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_creative_commons_attribution_asks_for_credit_where_the_model_is_offered() {
        assert!(!ModelLicense::Mit.requires_attribution());
        assert!(!ModelLicense::Apache2.requires_attribution());
        assert!(ModelLicense::CcBy4.requires_attribution());
    }
}
