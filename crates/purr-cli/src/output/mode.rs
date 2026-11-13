use indicatif::ProgressDrawTarget;

/// Whether the CLI talks to a person or only produces its result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Status messages and progress bars go to stdout beside the result
    Interactive,
    /// Stdout carries the result and nothing else
    Plain,
}

impl Mode {
    /// Plain when asked for with `--plain`, and whenever stdout is not a terminal.
    pub const fn decide(plain_flag: bool, stdout_is_terminal: bool) -> Self {
        if plain_flag || !stdout_is_terminal {
            Self::Plain
        } else {
            Self::Interactive
        }
    }

    pub const fn is_plain(self) -> bool {
        matches!(self, Self::Plain)
    }

    pub fn draw_target(self) -> ProgressDrawTarget {
        match self {
            Self::Interactive => ProgressDrawTarget::stdout(),
            Self::Plain => ProgressDrawTarget::hidden(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_a_terminal_without_the_flag_is_interactive() {
        let table = [
            (false, true, Mode::Interactive),
            (false, false, Mode::Plain),
            (true, true, Mode::Plain),
            (true, false, Mode::Plain),
        ];
        for (plain_flag, terminal, expected) in table {
            assert_eq!(
                Mode::decide(plain_flag, terminal),
                expected,
                "--plain={plain_flag} terminal={terminal}"
            );
        }
    }

    #[test]
    fn plain_mode_draws_no_progress() {
        assert!(Mode::Plain.draw_target().is_hidden());
    }
}
