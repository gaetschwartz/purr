//! Keyboard shortcuts: views register chords with [`use_shortcut`] while they are mounted.
//!
//! One document listener reports key presses that match a registered chord; it never fires while
//! a popover or sheet is open, nor in text fields unless the shortcut opts in with `in_inputs`.

use dioxus::prelude::*;
use serde::{Deserialize, Serialize};

/// A key, as `KeyboardEvent.key` names it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Key {
    /// A printable character, lower-cased.
    Char(char),
    Enter,
    Escape,
    Space,
    Tab,
    Backspace,
    Delete,
    Home,
    End,
    ArrowUp,
    ArrowDown,
    ArrowLeft,
    ArrowRight,
}

impl Key {
    /// Parses a `KeyboardEvent.key` value.
    #[must_use]
    pub fn from_dom(name: &str) -> Option<Self> {
        let named = match name {
            "Enter" => Some(Self::Enter),
            "Escape" => Some(Self::Escape),
            " " | "Spacebar" => Some(Self::Space),
            "Tab" => Some(Self::Tab),
            "Backspace" => Some(Self::Backspace),
            "Delete" => Some(Self::Delete),
            "Home" => Some(Self::Home),
            "End" => Some(Self::End),
            "ArrowUp" => Some(Self::ArrowUp),
            "ArrowDown" => Some(Self::ArrowDown),
            "ArrowLeft" => Some(Self::ArrowLeft),
            "ArrowRight" => Some(Self::ArrowRight),
            _ => None,
        };
        named.or_else(|| {
            let mut chars = name.chars();
            match (chars.next(), chars.next()) {
                (Some(single), None) => single.to_lowercase().next().map(Self::Char),
                _ => None,
            }
        })
    }

    fn dom_name(self) -> String {
        match self {
            Self::Char(c) => c.to_string(),
            Self::Enter => "enter".to_owned(),
            Self::Escape => "escape".to_owned(),
            Self::Space => " ".to_owned(),
            Self::Tab => "tab".to_owned(),
            Self::Backspace => "backspace".to_owned(),
            Self::Delete => "delete".to_owned(),
            Self::Home => "home".to_owned(),
            Self::End => "end".to_owned(),
            Self::ArrowUp => "arrowup".to_owned(),
            Self::ArrowDown => "arrowdown".to_owned(),
            Self::ArrowLeft => "arrowleft".to_owned(),
            Self::ArrowRight => "arrowright".to_owned(),
        }
    }

    /// Punctuation such as `?` needs Shift on some layouts and not others, so Shift is not compared.
    const fn ignores_shift(self) -> bool {
        match self {
            Self::Char(c) => !c.is_ascii_alphanumeric(),
            Self::Enter
            | Self::Escape
            | Self::Space
            | Self::Tab
            | Self::Backspace
            | Self::Delete
            | Self::Home
            | Self::End
            | Self::ArrowUp
            | Self::ArrowDown
            | Self::ArrowLeft
            | Self::ArrowRight => false,
        }
    }
}

/// A key with modifiers. `primary` is Cmd on Apple platforms and Ctrl elsewhere.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KeyChord {
    pub key: Key,
    pub primary: bool,
    pub shift: bool,
    pub alt: bool,
}

impl KeyChord {
    /// The key alone.
    #[must_use]
    pub const fn key(key: Key) -> Self {
        Self {
            key,
            primary: false,
            shift: false,
            alt: false,
        }
    }

    /// Cmd or Ctrl plus the key.
    #[must_use]
    pub const fn primary(key: Key) -> Self {
        Self {
            key,
            primary: true,
            shift: false,
            alt: false,
        }
    }

    /// Adds Shift.
    #[must_use]
    pub const fn with_shift(mut self) -> Self {
        self.shift = true;
        self
    }

    /// Adds Alt (Option).
    #[must_use]
    pub const fn with_alt(mut self) -> Self {
        self.alt = true;
        self
    }
}

/// A chord and where it applies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Shortcut {
    pub chord: KeyChord,
    /// Also fire while focus is in a text field, select or editable region.
    pub in_inputs: bool,
}

impl Shortcut {
    /// A shortcut that stays out of the way of text fields.
    #[must_use]
    pub const fn new(chord: KeyChord) -> Self {
        Self {
            chord,
            in_inputs: false,
        }
    }

    /// Lets the shortcut fire in text fields too, for chords that never type text (Cmd+O).
    #[must_use]
    pub const fn in_inputs(mut self) -> Self {
        self.in_inputs = true;
        self
    }
}

/// A key press as the document listener saw it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KeyPress {
    pub chord: KeyChord,
    pub in_editable: bool,
    pub modal_open: bool,
}

/// Whether `press` triggers `shortcut`.
#[must_use]
pub fn matches(shortcut: &Shortcut, press: &KeyPress) -> bool {
    let wanted = shortcut.chord;
    let got = press.chord;
    !press.modal_open
        && (shortcut.in_inputs || !press.in_editable)
        && wanted.key == got.key
        && wanted.primary == got.primary
        && wanted.alt == got.alt
        && (wanted.shift == got.shift || wanted.key.ignores_shift())
}

/// Identifies a registered shortcut.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ShortcutId(u64);

/// The shortcuts of the mounted views; the latest registration wins a shared chord.
#[derive(Default)]
pub struct ShortcutRegistry {
    entries: Vec<(ShortcutId, Shortcut, Callback<()>)>,
    next_id: u64,
}

impl ShortcutRegistry {
    fn register(&mut self, shortcut: Shortcut, handler: Callback<()>) -> ShortcutId {
        let id = ShortcutId(self.next_id);
        self.next_id += 1;
        self.entries.push((id, shortcut, handler));
        id
    }

    fn unregister(&mut self, id: ShortcutId) {
        self.entries.retain(|(entry, _, _)| *entry != id);
    }

    /// The shortcut a key press triggers, if any.
    #[must_use]
    pub fn resolve(&self, press: &KeyPress) -> Option<ShortcutId> {
        self.entries
            .iter()
            .rev()
            .find(|(_, shortcut, _)| matches(shortcut, press))
            .map(|(id, _, _)| *id)
    }

    fn handler(&self, id: ShortcutId) -> Option<Callback<()>> {
        self.entries
            .iter()
            .find(|(entry, _, _)| *entry == id)
            .map(|(_, _, handler)| *handler)
    }

    fn wire_chords(&self) -> Vec<WireChord> {
        self.entries
            .iter()
            .map(|(_, shortcut, _)| WireChord {
                key: shortcut.chord.key.dom_name(),
                primary: shortcut.chord.primary,
                shift: shortcut.chord.shift,
                alt: shortcut.chord.alt,
                any_shift: shortcut.chord.key.ignores_shift(),
                in_inputs: shortcut.in_inputs,
            })
            .collect()
    }
}

#[derive(Serialize)]
struct WireChord {
    key: String,
    primary: bool,
    shift: bool,
    alt: bool,
    any_shift: bool,
    in_inputs: bool,
}

#[derive(Deserialize)]
struct WirePress {
    key: String,
    primary: bool,
    shift: bool,
    alt: bool,
    editable: bool,
    modal: bool,
}

impl WirePress {
    fn into_press(self) -> Option<KeyPress> {
        Key::from_dom(&self.key).map(|key| KeyPress {
            chord: KeyChord {
                key,
                primary: self.primary,
                shift: self.shift,
                alt: self.alt,
            },
            in_editable: self.editable,
            modal_open: self.modal,
        })
    }
}

const LISTENER: &str = r"
let chords = [];
const editable = (target) => !!target && (target.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName));
document.addEventListener('keydown', (event) => {
  if (event.defaultPrevented || event.isComposing) return;
  const press = {
    key: event.key,
    primary: event.metaKey || event.ctrlKey,
    shift: event.shiftKey,
    alt: event.altKey,
    editable: editable(event.target),
    modal: !!document.querySelector('.app > .layer'),
  };
  if (press.modal) return;
  const key = event.key.toLowerCase();
  const hit = chords.some((chord) => chord.key === key && chord.primary === press.primary && chord.alt === press.alt
    && (chord.any_shift || chord.shift === press.shift) && (chord.in_inputs || !press.editable));
  if (!hit) return;
  event.preventDefault();
  dioxus.send(press);
});
for (;;) { chords = await dioxus.recv(); }
";

/// Installs the document listener and provides the registry; call once at the root.
pub fn use_shortcut_provider() {
    let registry = use_context_provider(|| Signal::new(ShortcutRegistry::default()));
    let listener = use_hook(|| document::eval(LISTENER));

    use_effect(move || {
        let chords = registry.read().wire_chords();
        if let Err(error) = listener.send(chords) {
            tracing::warn!("Could not update the keyboard shortcuts: {error}");
        }
    });

    use_future(move || async move {
        let mut listener = listener;
        loop {
            match listener.recv::<WirePress>().await {
                Ok(wire) => {
                    let Some(press) = wire.into_press() else {
                        continue;
                    };
                    let handler = {
                        let registry = registry.peek();
                        registry.resolve(&press).and_then(|id| registry.handler(id))
                    };
                    if let Some(handler) = handler {
                        handler.call(());
                    }
                }
                Err(error) => {
                    tracing::warn!("The keyboard shortcut listener stopped: {error}");
                    break;
                }
            }
        }
    });
}

/// Runs `handler` when `shortcut` is pressed, for as long as the calling component is mounted.
pub fn use_shortcut(shortcut: Shortcut, mut handler: impl FnMut() + 'static) {
    let mut registry = use_context::<Signal<ShortcutRegistry>>();
    let callback = use_callback(move |()| handler());
    let id = use_hook(|| registry.write().register(shortcut, callback));
    use_drop(move || registry.write().unregister(id));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn press(chord: KeyChord) -> KeyPress {
        KeyPress {
            chord,
            in_editable: false,
            modal_open: false,
        }
    }

    #[test]
    fn dom_key_names_parse() {
        assert_eq!(Key::from_dom("O"), Some(Key::Char('o')));
        assert_eq!(Key::from_dom(","), Some(Key::Char(',')));
        assert_eq!(Key::from_dom(" "), Some(Key::Space));
        assert_eq!(Key::from_dom("ArrowDown"), Some(Key::ArrowDown));
        assert_eq!(Key::from_dom("Shift"), None);
    }

    #[test]
    fn modifiers_must_match() {
        let open = Shortcut::new(KeyChord::primary(Key::Char('o')));
        assert!(matches(&open, &press(KeyChord::primary(Key::Char('o')))));
        assert!(!matches(&open, &press(KeyChord::key(Key::Char('o')))));
        assert!(!matches(
            &open,
            &press(KeyChord::primary(Key::Char('o')).with_shift())
        ));
    }

    #[test]
    fn punctuation_ignores_shift() {
        let help = Shortcut::new(KeyChord::key(Key::Char('?')));
        assert!(matches(
            &help,
            &press(KeyChord::key(Key::Char('?')).with_shift())
        ));
    }

    #[test]
    fn text_fields_and_open_sheets_block_shortcuts_unless_allowed() {
        let help = Shortcut::new(KeyChord::key(Key::Char('?')));
        let typing = KeyPress {
            in_editable: true,
            ..press(KeyChord::key(Key::Char('?')))
        };
        assert!(!matches(&help, &typing));
        let open = Shortcut::new(KeyChord::primary(Key::Char('o'))).in_inputs();
        let in_field = KeyPress {
            in_editable: true,
            ..press(KeyChord::primary(Key::Char('o')))
        };
        assert!(matches(&open, &in_field));
        let behind_sheet = KeyPress {
            modal_open: true,
            ..press(KeyChord::primary(Key::Char('o')))
        };
        assert!(!matches(&open, &behind_sheet));
    }
}
