use super::mode::Mode;
use crate::color::Stream;
use crate::verbosity::Verbosity;

/// What a piece of text is for; the channel alone decides where it goes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Channel {
    /// What the command produces: the transcript, a report. Always stdout
    Result,
    /// Messages to a person: banner, progress, notices, statistics. Stdout, interactive mode only
    Status,
    /// The error that ends the run. Stderr
    Error,
    /// Diagnostic lines a flag asks for (`--trace-beams`). Stderr in both modes
    Trace,
    /// The `tracing` events. Stderr with `-v`, nothing otherwise; the log file is separate
    Log,
}

/// The stream the channel writes to, or `None` when its text is dropped.
pub const fn route(channel: Channel, mode: Mode, verbosity: Verbosity) -> Option<Stream> {
    match channel {
        Channel::Result => Some(Stream::Stdout),
        Channel::Status => match mode {
            Mode::Interactive => Some(Stream::Stdout),
            Mode::Plain => None,
        },
        Channel::Error | Channel::Trace => Some(Stream::Stderr),
        Channel::Log => {
            if verbosity.shows_diagnostics() {
                Some(Stream::Stderr)
            } else {
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::verbosity::VerbosityLevel;

    const CHANNELS: [Channel; 5] = [
        Channel::Result,
        Channel::Status,
        Channel::Error,
        Channel::Trace,
        Channel::Log,
    ];

    fn verbosity(count: u8) -> Verbosity {
        Verbosity {
            verbose: [
                VerbosityLevel::NORMAL,
                VerbosityLevel::VERBOSE,
                VerbosityLevel::DEBUG,
                VerbosityLevel::TRACE,
            ]
            .into_iter()
            .find(|level| **level == count)
            .unwrap_or(VerbosityLevel::NORMAL),
        }
    }

    #[test]
    fn the_matrix_of_mode_verbosity_and_channel() {
        use Stream::{Stderr, Stdout};
        let interactive = [Some(Stdout), Some(Stdout), Some(Stderr), Some(Stderr), None];
        let interactive_verbose = [
            Some(Stdout),
            Some(Stdout),
            Some(Stderr),
            Some(Stderr),
            Some(Stderr),
        ];
        let plain = [Some(Stdout), None, Some(Stderr), Some(Stderr), None];
        let plain_verbose = [Some(Stdout), None, Some(Stderr), Some(Stderr), Some(Stderr)];
        let table = [
            (Mode::Interactive, 0, interactive),
            (Mode::Interactive, 1, interactive_verbose),
            (Mode::Interactive, 3, interactive_verbose),
            (Mode::Plain, 0, plain),
            (Mode::Plain, 1, plain_verbose),
            (Mode::Plain, 3, plain_verbose),
        ];
        for (mode, count, expected) in table {
            let routed = CHANNELS.map(|channel| route(channel, mode, verbosity(count)));
            assert_eq!(routed, expected, "{mode:?} -v x{count}");
        }
    }
}
