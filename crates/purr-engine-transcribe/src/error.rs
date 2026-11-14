use purr_core::engine::error::OpenError;

/// What goes wrong inside the transcribe.cpp engine.
#[derive(Debug, thiserror::Error)]
pub enum TranscribeEngineError {
    #[error("the transcribe.cpp engine cannot decode yet")]
    CannotDecodeYet,
}

impl From<TranscribeEngineError> for OpenError {
    fn from(error: TranscribeEngineError) -> Self {
        Self::Engine(Box::new(error))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_engine_error_stops_the_device_search() {
        assert!(matches!(
            OpenError::from(TranscribeEngineError::CannotDecodeYet),
            OpenError::Engine(_)
        ));
    }
}
