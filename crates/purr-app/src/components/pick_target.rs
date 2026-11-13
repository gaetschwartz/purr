/// Which setting a model or language picker edits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PickTarget {
    /// What the next transcription uses.
    #[default]
    Current,
    /// What every launch starts with, or "Last used" to keep whatever was picked last.
    Default,
}
