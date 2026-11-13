//! The CLI's palette: semantic roles, the one place that maps them to colours, and the extension
//! trait that styles anything displayable with a role.
//!
//! Only the eight base colours, bold, dim and italic are used so the output reads on dark and light
//! terminals alike.

use crate::global;
use crate::scope;
use clap::builder::styling::{AnsiColor, Effects, Style as ClapStyle, Styles};
use owo_colors::{AnsiColors, Style};
use std::fmt::{self, Display};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    /// Section headings and the banner
    Heading,
    /// The `Info:` tag and `info` log lines
    Info,
    /// The `Success:` tag and finished-action lines
    Success,
    /// The `Warning:` / `Notice:` tags, `warn` log lines and entries to look at
    Warning,
    /// The `Error:` tag, `error` log lines and failed entries
    Error,
    /// `debug` log lines
    Debug,
    /// `trace` log lines
    Trace,
    /// Names of things: models, devices, processors
    Name,
    /// A good state: downloaded, available, yes
    Positive,
    /// A bad state: incomplete, no, unusable
    Negative,
    /// A state that is neither: not built
    Caution,
    /// Sizes and durations
    Quantity,
    /// Secondary text: descriptions, notes, explanations
    Muted,
    /// File and directory paths, and inputs
    Path,
    /// Commands the user can type
    Command,
    /// A placeholder in the usage text
    Placeholder,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Hue {
    Red,
    Green,
    Yellow,
    Blue,
    Magenta,
    Cyan,
}

impl Hue {
    const fn owo(self) -> AnsiColors {
        match self {
            Self::Red => AnsiColors::Red,
            Self::Green => AnsiColors::Green,
            Self::Yellow => AnsiColors::Yellow,
            Self::Blue => AnsiColors::Blue,
            Self::Magenta => AnsiColors::Magenta,
            Self::Cyan => AnsiColors::Cyan,
        }
    }

    const fn clap(self) -> AnsiColor {
        match self {
            Self::Red => AnsiColor::Red,
            Self::Green => AnsiColor::Green,
            Self::Yellow => AnsiColor::Yellow,
            Self::Blue => AnsiColor::Blue,
            Self::Magenta => AnsiColor::Magenta,
            Self::Cyan => AnsiColor::Cyan,
        }
    }

    const fn indicatif(self) -> &'static str {
        match self {
            Self::Red => "red",
            Self::Green => "green",
            Self::Yellow => "yellow",
            Self::Blue => "blue",
            Self::Magenta => "magenta",
            Self::Cyan => "cyan",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Look {
    hue: Option<Hue>,
    bold: bool,
    dim: bool,
    italic: bool,
}

impl Look {
    const PLAIN: Self = Self {
        hue: None,
        bold: false,
        dim: false,
        italic: false,
    };

    const fn hue(hue: Hue) -> Self {
        Self {
            hue: Some(hue),
            ..Self::PLAIN
        }
    }

    const fn bold(self) -> Self {
        Self { bold: true, ..self }
    }

    const fn dim(self) -> Self {
        Self { dim: true, ..self }
    }

    const fn italic(self) -> Self {
        Self {
            italic: true,
            ..self
        }
    }

    fn owo(self) -> Style {
        let mut style = Style::new();
        if let Some(hue) = self.hue {
            style = style.color(hue.owo());
        }
        if self.bold {
            style = style.bold();
        }
        if self.dim {
            style = style.dimmed();
        }
        if self.italic {
            style = style.italic();
        }
        style
    }

    const fn clap(self) -> ClapStyle {
        let mut style = match self.hue {
            Some(hue) => hue.clap().on_default(),
            None => ClapStyle::new(),
        };
        let mut effects = Effects::new();
        if self.bold {
            effects = effects.insert(Effects::BOLD);
        }
        if self.dim {
            effects = effects.insert(Effects::DIMMED);
        }
        if self.italic {
            effects = effects.insert(Effects::ITALIC);
        }
        style = style.effects(effects);
        style
    }
}

impl Role {
    const fn look(self) -> Look {
        match self {
            Self::Heading | Self::Info => Look::hue(Hue::Blue).bold(),
            Self::Success | Self::Debug => Look::hue(Hue::Green).bold(),
            Self::Warning => Look::hue(Hue::Yellow).bold(),
            Self::Error => Look::hue(Hue::Red).bold(),
            Self::Trace => Look::hue(Hue::Magenta).bold(),
            Self::Name | Self::Positive => Look::hue(Hue::Green),
            Self::Negative => Look::hue(Hue::Red),
            Self::Caution | Self::Quantity => Look::hue(Hue::Yellow),
            Self::Muted => Look::PLAIN.dim(),
            Self::Path => Look::hue(Hue::Cyan),
            Self::Command => Look::hue(Hue::Cyan).bold(),
            Self::Placeholder => Look::hue(Hue::Cyan).italic(),
        }
    }

    pub const fn paint<T>(self, value: T) -> Painted<T> {
        Painted {
            value,
            role: self,
            gate: Gate::Ambient,
        }
    }
}

/// The colours of `--help` and of usage errors.
pub const CLAP_STYLES: Styles = Styles::styled()
    .header(Role::Heading.look().clap())
    .usage(Role::Heading.look().clap())
    .literal(Role::Command.look().clap())
    .placeholder(Role::Placeholder.look().clap())
    .error(Role::Error.look().clap())
    .valid(Role::Command.look().clap())
    .invalid(Role::Warning.look().clap());

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProgressPart {
    Spinner,
    Bar,
}

/// The style suffix of an indicatif template key (`{spinner:.green}`), empty without colours.
pub fn progress_style(part: ProgressPart, enabled: bool) -> String {
    if !enabled {
        return String::new();
    }
    match part {
        ProgressPart::Spinner => {
            format!(":.{}", Role::Positive.look().hue.map_or("", Hue::indicatif))
        }
        ProgressPart::Bar => format!(
            ":.{}/{}",
            Role::Path.look().hue.map_or("", Hue::indicatif),
            Role::Info.look().hue.map_or("", Hue::indicatif),
        ),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Gate {
    Ambient,
    Fixed(bool),
}

impl Gate {
    fn enabled(self) -> bool {
        match self {
            Self::Ambient => global::colors().enabled(scope::current()),
            Self::Fixed(enabled) => enabled,
        }
    }
}

/// A value shown with a role's style when colours are on for the stream it is formatted for (see
/// `scope`); otherwise exactly the value.
#[derive(Clone, Copy, Debug)]
pub struct Painted<T> {
    value: T,
    role: Role,
    gate: Gate,
}

impl<T> Painted<T> {
    /// Ignores the stream and the environment; for a writer that has already decided.
    pub fn fixed(self, enabled: bool) -> Self {
        Self {
            gate: Gate::Fixed(enabled),
            ..self
        }
    }
}

impl<T: Display> Display for Painted<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if !self.gate.enabled() {
            return self.value.fmt(f);
        }
        let style = self.role.look().owo();
        style.fmt_prefix(f)?;
        self.value.fmt(f)?;
        style.fmt_suffix(f)
    }
}

/// Styles anything displayable with the CLI's roles.
pub trait Themed: Display {
    fn painted(&self, role: Role) -> Painted<&Self> {
        role.paint(self)
    }

    fn heading(&self) -> Painted<&Self> {
        self.painted(Role::Heading)
    }

    fn info(&self) -> Painted<&Self> {
        self.painted(Role::Info)
    }

    fn success(&self) -> Painted<&Self> {
        self.painted(Role::Success)
    }

    fn warning(&self) -> Painted<&Self> {
        self.painted(Role::Warning)
    }

    fn error(&self) -> Painted<&Self> {
        self.painted(Role::Error)
    }

    fn name(&self) -> Painted<&Self> {
        self.painted(Role::Name)
    }

    fn positive(&self) -> Painted<&Self> {
        self.painted(Role::Positive)
    }

    fn negative(&self) -> Painted<&Self> {
        self.painted(Role::Negative)
    }

    fn quantity(&self) -> Painted<&Self> {
        self.painted(Role::Quantity)
    }

    fn muted(&self) -> Painted<&Self> {
        self.painted(Role::Muted)
    }

    fn path(&self) -> Painted<&Self> {
        self.painted(Role::Path)
    }

    fn command(&self) -> Painted<&Self> {
        self.painted(Role::Command)
    }
}

impl<T: Display + ?Sized> Themed for T {}

#[cfg(test)]
mod tests {
    use super::*;

    fn painted(role: Role, enabled: bool) -> String {
        role.paint("text").fixed(enabled).to_string()
    }

    #[test]
    fn every_role_has_its_escape_sequences_when_colours_are_on() {
        let expected = [
            (Role::Heading, "\u{1b}[34;1mtext\u{1b}[0m"),
            (Role::Info, "\u{1b}[34;1mtext\u{1b}[0m"),
            (Role::Success, "\u{1b}[32;1mtext\u{1b}[0m"),
            (Role::Warning, "\u{1b}[33;1mtext\u{1b}[0m"),
            (Role::Error, "\u{1b}[31;1mtext\u{1b}[0m"),
            (Role::Debug, "\u{1b}[32;1mtext\u{1b}[0m"),
            (Role::Trace, "\u{1b}[35;1mtext\u{1b}[0m"),
            (Role::Name, "\u{1b}[32mtext\u{1b}[0m"),
            (Role::Positive, "\u{1b}[32mtext\u{1b}[0m"),
            (Role::Negative, "\u{1b}[31mtext\u{1b}[0m"),
            (Role::Caution, "\u{1b}[33mtext\u{1b}[0m"),
            (Role::Quantity, "\u{1b}[33mtext\u{1b}[0m"),
            (Role::Muted, "\u{1b}[2mtext\u{1b}[0m"),
            (Role::Path, "\u{1b}[36mtext\u{1b}[0m"),
            (Role::Command, "\u{1b}[36;1mtext\u{1b}[0m"),
            (Role::Placeholder, "\u{1b}[36;3mtext\u{1b}[0m"),
        ];
        for (role, sequence) in expected {
            assert_eq!(painted(role, true), sequence, "{role:?}");
        }
    }

    #[test]
    fn without_colours_every_role_is_the_plain_text() {
        for role in [
            Role::Heading,
            Role::Info,
            Role::Success,
            Role::Warning,
            Role::Error,
            Role::Debug,
            Role::Trace,
            Role::Name,
            Role::Positive,
            Role::Negative,
            Role::Caution,
            Role::Quantity,
            Role::Muted,
            Role::Path,
            Role::Command,
            Role::Placeholder,
        ] {
            assert_eq!(painted(role, false), "text", "{role:?}");
        }
    }

    #[test]
    fn formatting_flags_reach_the_styled_value() {
        assert_eq!(
            format!("{:>6}", Role::Name.paint("ab").fixed(false)),
            "    ab"
        );
        assert_eq!(
            format!("{:>6}", Role::Name.paint("ab").fixed(true)),
            "\u{1b}[32m    ab\u{1b}[0m"
        );
    }

    #[test]
    fn the_extension_trait_applies_its_role() {
        assert_eq!(
            "text".path().fixed(true).to_string(),
            "\u{1b}[36mtext\u{1b}[0m"
        );
    }

    #[test]
    fn progress_templates_only_carry_colours_when_enabled() {
        assert_eq!(progress_style(ProgressPart::Spinner, false), "");
        assert_eq!(progress_style(ProgressPart::Spinner, true), ":.green");
        assert_eq!(progress_style(ProgressPart::Bar, false), "");
        assert_eq!(progress_style(ProgressPart::Bar, true), ":.cyan/blue");
    }
}
