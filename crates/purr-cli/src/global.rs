//! The parsed command line, readable from anywhere once `main` has set it.

use crate::cli::Cli;
use crate::color::{ColorChoice, Colors};
use crate::fmt::{Verbosity, VerbosityLevel};
use std::sync::OnceLock;

struct Global {
    cli: Cli,
    colors: Colors,
}

static GLOBAL: OnceLock<Global> = OnceLock::new();

/// Keeps the arguments for the rest of the process. Only the first call stores anything.
pub fn init(cli: Cli) -> &'static Cli {
    &GLOBAL
        .get_or_init(|| {
            let colors = Colors::detect(cli.color);
            Global { cli, colors }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn before_init_the_defaults_apply() {
        assert_eq!(args().map(|_| ()), None);
        assert_eq!(verbosity().verbose, VerbosityLevel::NORMAL);
    }
}
