/// The terms a model is released under.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelLicense {
    Mit,
    CcBy4,
}

impl ModelLicense {
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Mit => "MIT",
            Self::CcBy4 => "CC BY 4.0",
        }
    }

    #[must_use]
    pub const fn url(self) -> &'static str {
        match self {
            Self::Mit => "https://opensource.org/license/mit",
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
