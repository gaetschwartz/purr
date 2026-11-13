//! The `-v` count.

use std::ops::Deref;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, clap::Args)]
pub struct Verbosity {
    #[clap(flatten)]
    pub verbose: VerbosityLevel,
}

impl Deref for Verbosity {
    type Target = VerbosityLevel;

    fn deref(&self) -> &Self::Target {
        &self.verbose
    }
}

#[allow(dead_code)]
impl Verbosity {
    pub const fn is_verbose(self) -> bool {
        self.verbose.verbose >= VerbosityLevel::VERBOSE.verbose
    }

    pub const fn is_debug(self) -> bool {
        self.verbose.verbose >= VerbosityLevel::DEBUG.verbose
    }

    pub const fn is_trace(self) -> bool {
        self.verbose.verbose >= VerbosityLevel::TRACE.verbose
    }

    /// Whether the diagnostic log is shown on the terminal at all.
    pub const fn shows_diagnostics(self) -> bool {
        self.is_verbose()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, clap::Args)]

pub struct VerbosityLevel {
    #[arg(short, long, global = true, action = clap::ArgAction::Count)]
    /// Show the diagnostic log on stderr (-v, -vv, -vvv for more verbosity)
    verbose: u8,
}

#[allow(dead_code)]
impl VerbosityLevel {
    pub const TRACE_VALUE: u8 = 3;
    pub const DEBUG_VALUE: u8 = 2;
    pub const VERBOSE_VALUE: u8 = 1;
    pub const NORMAL_VALUE: u8 = 0;
    pub const TRACE: Self = Self {
        verbose: Self::TRACE_VALUE,
    };
    pub const DEBUG: Self = Self {
        verbose: Self::DEBUG_VALUE,
    };
    pub const VERBOSE: Self = Self {
        verbose: Self::VERBOSE_VALUE,
    };
    pub const NORMAL: Self = Self {
        verbose: Self::NORMAL_VALUE,
    };
}

impl Deref for VerbosityLevel {
    type Target = u8;

    fn deref(&self) -> &Self::Target {
        &self.verbose
    }
}
