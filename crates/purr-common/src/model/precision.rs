/// How the weights are stored; quantized ones are smaller and slightly less accurate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    Full,
    Quantized {
        bits: u8,
    },
    /// Encoder and decoder use different precisions, as the browser models do.
    Mixed,
}
