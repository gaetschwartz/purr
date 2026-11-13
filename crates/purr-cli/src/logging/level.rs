use tracing_subscriber::filter::LevelFilter;

/// Value of `--log-level`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, clap::ValueEnum)]
pub enum LogLevel {
    Error,
    Warn,
    #[default]
    Info,
    Debug,
    Trace,
}

impl LogLevel {
    pub const fn filter(self) -> LevelFilter {
        match self {
            Self::Error => LevelFilter::ERROR,
            Self::Warn => LevelFilter::WARN,
            Self::Info => LevelFilter::INFO,
            Self::Debug => LevelFilter::DEBUG,
            Self::Trace => LevelFilter::TRACE,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_level_maps_to_its_filter() {
        let table = [
            (LogLevel::Error, LevelFilter::ERROR),
            (LogLevel::Warn, LevelFilter::WARN),
            (LogLevel::Info, LevelFilter::INFO),
            (LogLevel::Debug, LevelFilter::DEBUG),
            (LogLevel::Trace, LevelFilter::TRACE),
        ];
        for (level, filter) in table {
            assert_eq!(level.filter(), filter, "{level:?}");
        }
    }

    #[test]
    fn the_default_level_is_info() {
        assert_eq!(LogLevel::default(), LogLevel::Info);
    }
}
