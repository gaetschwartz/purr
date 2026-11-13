//! Whether the CLI colours its output: the `--color` flag, the environment and the stream.

use crate::output::sink;

/// Value of `--color`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, clap::ValueEnum)]
pub enum ColorChoice {
    /// Follow the environment, then whether the stream is a terminal
    #[default]
    Auto,
    /// Colour even when the output is not a terminal
    Always,
    /// Never colour
    Never,
}

impl From<ColorChoice> for clap::ColorChoice {
    fn from(choice: ColorChoice) -> Self {
        match choice {
            ColorChoice::Auto => Self::Auto,
            ColorChoice::Always => Self::Always,
            ColorChoice::Never => Self::Never,
        }
    }
}

/// The output stream a piece of styled text is written to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stream {
    Stdout,
    Stderr,
}

impl Stream {
    pub fn is_terminal(self) -> bool {
        sink::is_terminal(self)
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Stdout => "stdout",
            Self::Stderr => "stderr",
        }
    }
}

/// The environment variables that take part in the decision.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ColorEnv {
    /// `NO_COLOR` is set to a non-empty value
    pub no_color: bool,
    /// `CLICOLOR_FORCE` is set to a value other than empty or `0`
    pub clicolor_force: bool,
    /// `CLICOLOR` is `0`
    pub clicolor_off: bool,
    /// `TERM` is `dumb`
    pub dumb_terminal: bool,
}

impl ColorEnv {
    pub fn from_process() -> Self {
        let var = |name: &str| std::env::var_os(name);
        Self {
            no_color: var("NO_COLOR").is_some_and(|value| !value.is_empty()),
            clicolor_force: var("CLICOLOR_FORCE")
                .is_some_and(|value| !value.is_empty() && value != "0"),
            clicolor_off: var("CLICOLOR").is_some_and(|value| value == "0"),
            dumb_terminal: var("TERM").is_some_and(|value| value == "dumb"),
        }
    }
}

/// An explicit `--color` wins, then `NO_COLOR`, `CLICOLOR_FORCE`, `CLICOLOR=0`, and last whether the
/// stream is a terminal that is not `dumb`.
pub const fn decide(choice: ColorChoice, env: ColorEnv, is_terminal: bool) -> bool {
    match choice {
        ColorChoice::Always => true,
        ColorChoice::Never => false,
        ColorChoice::Auto => {
            if env.no_color {
                false
            } else if env.clicolor_force {
                true
            } else if env.clicolor_off {
                false
            } else {
                is_terminal && !env.dumb_terminal
            }
        }
    }
}

/// The decision for each output stream, taken once.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Colors {
    stdout: bool,
    stderr: bool,
}

impl Colors {
    pub const fn new(stdout: bool, stderr: bool) -> Self {
        Self { stdout, stderr }
    }

    pub const fn resolve(
        choice: ColorChoice,
        env: ColorEnv,
        stdout_terminal: bool,
        stderr_terminal: bool,
    ) -> Self {
        Self::new(
            decide(choice, env, stdout_terminal),
            decide(choice, env, stderr_terminal),
        )
    }

    pub fn detect(choice: ColorChoice) -> Self {
        let stdout_terminal = Stream::Stdout.is_terminal();
        let stderr_terminal = Stream::Stderr.is_terminal();
        let resolved = Self::resolve(
            choice,
            ColorEnv::from_process(),
            stdout_terminal,
            stderr_terminal,
        );
        let usable = |stream: Stream, is_terminal: bool| {
            resolved.enabled(stream) && (!is_terminal || crate::vt::enable(stream))
        };
        Self::new(
            usable(Stream::Stdout, stdout_terminal),
            usable(Stream::Stderr, stderr_terminal),
        )
    }

    pub const fn enabled(self, stream: Stream) -> bool {
        match stream {
            Stream::Stdout => self.stdout,
            Stream::Stderr => self.stderr,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const NONE: ColorEnv = ColorEnv {
        no_color: false,
        clicolor_force: false,
        clicolor_off: false,
        dumb_terminal: false,
    };

    #[test]
    fn the_decision_follows_the_documented_order() {
        use ColorChoice::{Always, Auto, Never};
        let no_color = ColorEnv {
            no_color: true,
            ..NONE
        };
        let force = ColorEnv {
            clicolor_force: true,
            ..NONE
        };
        let off = ColorEnv {
            clicolor_off: true,
            ..NONE
        };
        let dumb = ColorEnv {
            dumb_terminal: true,
            ..NONE
        };
        let no_color_and_force = ColorEnv {
            no_color: true,
            clicolor_force: true,
            ..NONE
        };
        let force_and_off = ColorEnv {
            clicolor_force: true,
            clicolor_off: true,
            ..NONE
        };
        let table = [
            (Always, no_color, false, true),
            (Always, NONE, false, true),
            (Never, force, true, false),
            (Never, NONE, true, false),
            (Auto, NONE, true, true),
            (Auto, NONE, false, false),
            (Auto, no_color, true, false),
            (Auto, force, false, true),
            (Auto, no_color_and_force, true, false),
            (Auto, off, true, false),
            (Auto, force_and_off, false, true),
            (Auto, dumb, true, false),
        ];
        for (choice, env, is_terminal, expected) in table {
            assert_eq!(
                decide(choice, env, is_terminal),
                expected,
                "{choice:?} {env:?} terminal={is_terminal}"
            );
        }
    }

    #[test]
    fn stdout_and_stderr_are_decided_separately() {
        use ColorChoice::{Always, Auto, Never};
        let no_color = ColorEnv {
            no_color: true,
            ..NONE
        };
        let force = ColorEnv {
            clicolor_force: true,
            ..NONE
        };
        let table = [
            (Auto, NONE, true, true, (true, true)),
            (Auto, NONE, true, false, (true, false)),
            (Auto, NONE, false, true, (false, true)),
            (Auto, NONE, false, false, (false, false)),
            (Auto, no_color, true, true, (false, false)),
            (Auto, force, false, false, (true, true)),
            (Always, NONE, false, false, (true, true)),
            (Always, no_color, true, false, (true, true)),
            (Never, force, true, true, (false, false)),
        ];
        for (choice, env, stdout_terminal, stderr_terminal, (stdout, stderr)) in table {
            assert_eq!(
                Colors::resolve(choice, env, stdout_terminal, stderr_terminal),
                Colors::new(stdout, stderr),
                "{choice:?} {env:?} stdout_tty={stdout_terminal} stderr_tty={stderr_terminal}"
            );
        }
    }

    #[test]
    fn each_stream_has_its_own_answer() {
        let colors = Colors::new(false, true);
        assert_eq!(
            (
                colors.enabled(Stream::Stdout),
                colors.enabled(Stream::Stderr)
            ),
            (false, true)
        );
    }
}
