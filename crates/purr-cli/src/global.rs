//! The parsed command line, readable from anywhere once `main` has set it.

use crate::cli::Cli;
use crate::color::{ColorChoice, Colors, Stream};
use crate::output::mode::Mode;
use crate::verbosity::{Verbosity, VerbosityLevel};
use std::sync::OnceLock;

struct Global {
    cli: Cli,
    colors: Colors,
    mode: Mode,
}

static GLOBAL: OnceLock<Global> = OnceLock::new();

/// Keeps the arguments for the rest of the process. Only the first call stores anything.
pub fn init(cli: Cli) -> &'static Cli {
    &GLOBAL
        .get_or_init(|| {
            let colors = Colors::detect(cli.color);
            let mode = Mode::decide(cli.plain, Stream::Stdout.is_terminal());
            Global { cli, colors, mode }
        })
        .cli
}

/// The arguments, once `init` has run.
pub fn args() -> Option<&'static Cli> {
    GLOBAL.get().map(|global| &global.cli)
}

pub fn verbosity() -> Verbosity {
    args().map_or(
        Verbosity {
            verbose: VerbosityLevel::NORMAL,
        },
        |cli| cli.verbosity,
    )
}

/// Which streams have colours; before `init`, what `--color auto` gives.
pub fn colors() -> Colors {
    GLOBAL
        .get()
        .map_or_else(|| Colors::detect(ColorChoice::Auto), |global| global.colors)
}

/// Whether the run is interactive or plain; before `init`, what no `--plain` gives.
pub fn mode() -> Mode {
    GLOBAL.get().map_or_else(
        || Mode::decide(false, Stream::Stdout.is_terminal()),
        |global| global.mode,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn before_init_the_defaults_apply() {
        assert_eq!(args().map(|_| ()), None);
        assert_eq!(verbosity().verbose, VerbosityLevel::NORMAL);
    }
}
