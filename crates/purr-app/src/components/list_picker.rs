use super::{
    icon::{Icon, IconName, IconSize},
    popover::{OptionRole, Popover, PopoverOption},
    search_field::SearchField,
};
use crate::utils::dom_id::use_dom_id;
use dioxus::{
    core::{Runtime, ScopeId},
    prelude::*,
};

/// Which item of a [`ListPicker`] the keyboard and the pointer point at; there is only ever one.
///
/// Items are named by a key unique in the list, so the pointed item survives the list changing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ListNav {
    list_id: Signal<String>,
    pointed: Signal<Option<String>>,
}

/// A [`ListNav`] for one picker; nothing is pointed at until the pointer or the arrows move.
pub fn use_list_nav() -> ListNav {
    let list_id = use_dom_id("list");
    use_hook(|| {
        Runtime::current().in_scope(ScopeId::ROOT, || ListNav {
            list_id: Signal::new(list_id),
            pointed: Signal::new(None),
        })
    })
}

/// The keys of one line of a [`ListPicker`]: the line itself first, then any choices inside it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NavRow {
    keys: Vec<String>,
}

impl NavRow {
    /// A line that is a single choice.
    pub fn single(key: impl Into<String>) -> Self {
        Self {
            keys: vec![key.into()],
        }
    }

    /// A line with choices inside it, which Left and Right reach.
    pub fn with_inner(key: impl Into<String>, inner: impl IntoIterator<Item = String>) -> Self {
        Self {
            keys: std::iter::once(key.into()).chain(inner).collect(),
        }
    }

    fn lead(&self) -> Option<&String> {
        self.keys.first()
    }

    fn position(&self, key: &str) -> Option<usize> {
        self.keys.iter().position(|found| found == key)
    }
}

/// An arrow key press inside a [`ListPicker`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Motion {
    Up,
    Down,
    Left,
    Right,
}

/// The key the mark moves to from `active`: Up and Down go line to line, wrapping at both ends;
/// Left and Right move inside a line and stop at its ends.
#[must_use]
pub fn move_mark(rows: &[NavRow], active: Option<&str>, motion: Motion) -> Option<String> {
    let line = active.and_then(|key| rows.iter().position(|row| row.position(key).is_some()));
    match motion {
        Motion::Up | Motion::Down => step(line, rows.len(), motion == Motion::Down)
            .and_then(|index| rows.get(index))
            .and_then(NavRow::lead)
            .cloned(),
        Motion::Left | Motion::Right => {
            let row = rows.get(line?)?;
            let at = row.position(active?)?;
            let to = match motion {
                Motion::Right => at + 1,
                Motion::Left => at.checked_sub(1)?,
                Motion::Up | Motion::Down => at,
            };
            row.keys.get(to).cloned()
        }
    }
}

impl ListNav {
    /// The DOM id of the list box.
    #[must_use]
    pub fn list_id(&self) -> String {
        self.list_id.peek().clone()
    }

    /// The DOM id of the item with `key`.
    #[must_use]
    pub fn row_id(&self, key: &str) -> String {
        format!("{}-{key}", self.list_id.peek())
    }

    /// The key of the item to mark among `rows`: the pointed one, else the first line while a
    /// search narrows the list, so Enter picks the best match.
    #[must_use]
    pub fn active(&self, rows: &[NavRow], searching: bool) -> Option<String> {
        let pointed = self.pointed.read();
        pointed
            .as_ref()
            .filter(|key| rows.iter().any(|row| row.position(key).is_some()))
            .or_else(|| rows.first().and_then(NavRow::lead).filter(|_| searching))
            .cloned()
    }

    fn is_pointing(&self) -> bool {
        self.pointed.peek().is_some()
    }

    /// Points at the item with `key`.
    pub fn point(mut self, key: &str) {
        if self.pointed.peek().as_deref() != Some(key) {
            self.pointed.set(Some(key.to_owned()));
        }
    }

    /// Points at nothing.
    pub fn clear(mut self) {
        if self.pointed.peek().is_some() {
            self.pointed.set(None);
        }
    }
}

/// The index an arrow press moves to among `count` lines from `current`, wrapping at both ends.
#[must_use]
pub const fn step(current: Option<usize>, count: usize, down: bool) -> Option<usize> {
    if count == 0 {
        return None;
    }
    Some(match (current, down) {
        (None, true) => 0,
        (None, false) => count - 1,
        (Some(index), true) => (index + 1) % count,
        (Some(index), false) => (index + count - 1) % count,
    })
}

/// A popover (a sheet on phones) for choosing one item of a list, shared by the model and language
/// pickers: a pinned search line with an optional `filter` beside the field, an optional note, the
/// scrolling list box, and a pinned footer.
///
/// `rows` names the lines in the order they show. Up and Down move between lines from the search
/// field, Left and Right between the choices of a line once one is marked, and Enter calls
/// `on_enter` with the marked key. Items mark themselves with `nav.active(..)` and call
/// `nav.point(..)` on pointer moves, so the pointer and the keyboard share one mark.
#[component]
pub fn ListPicker(
    /// The id of the element the popover hangs from.
    #[props(into)]
    anchor: String,
    /// The dialog's name.
    #[props(into)]
    label: String,
    /// The list box's name.
    #[props(into)]
    list_label: String,
    on_close: EventHandler<()>,
    #[props(default)] wide: bool,
    nav: ListNav,
    rows: Vec<NavRow>,
    on_enter: EventHandler<String>,
    /// The search field's placeholder; without one there is no search field.
    #[props(into)]
    search: Option<String>,
    #[props(into, default)] query: String,
    on_query: Option<EventHandler<String>>,
    /// A control beside the search field that narrows the list.
    filter: Option<Element>,
    /// A line under the search field.
    note: Option<Element>,
    /// What shows instead of items, such as a loading or empty state.
    status: Option<Element>,
    footer: Option<Element>,
    children: Element,
) -> Element {
    let searching = !query.trim().is_empty();
    let active = nav.active(&rows, searching);
    let active_id = active.as_deref().map(|key| nav.row_id(key));
    let list_id = nav.list_id();

    let on_key = move |event: KeyboardEvent| {
        let motion = match event.key() {
            Key::Named(NamedKey::ArrowDown) => Motion::Down,
            Key::Named(NamedKey::ArrowUp) => Motion::Up,
            Key::Named(NamedKey::ArrowLeft) if nav.is_pointing() => Motion::Left,
            Key::Named(NamedKey::ArrowRight) if nav.is_pointing() => Motion::Right,
            Key::Named(NamedKey::Enter) => {
                if let Some(key) = &active {
                    event.prevent_default();
                    on_enter.call(key.clone());
                }
                return;
            }
            Key::Named(_) | Key::Character(_) => return,
        };
        let Some(key) = move_mark(&rows, active.as_deref(), motion) else {
            return;
        };
        event.prevent_default();
        nav.point(&key);
        document::eval(&format!(
            "document.getElementById({:?})?.scrollIntoView({{block:'nearest'}})",
            nav.row_id(&key)
        ));
    };

    let field = search.map(|placeholder| {
        rsx! {
            SearchField {
                value: query,
                placeholder: placeholder.clone(),
                label: placeholder,
                controls: list_id.clone(),
                active_descendant: active_id,
                oninput: move |value: String| {
                    nav.clear();
                    if let Some(handler) = on_query {
                        handler.call(value);
                    }
                },
                onkeydown: on_key,
            }
        }
    });
    let header = match (field, filter, note) {
        (None, None, None) => None,
        (field, filter, note) => Some(rsx! {
            if field.is_some() || filter.is_some() {
                div { class: "psearch",
                    {field}
                    {filter}
                }
            }
            if let Some(note) = note {
                p { class: "pnote", role: "note", {note} }
            }
        }),
    };

    rsx! {
        Popover {
            anchor,
            label,
            on_close,
            wide,
            class: "picker",
            header,
            footer,
            {status}
            div {
                class: "plist",
                role: "listbox",
                id: list_id,
                aria_label: list_label,
                onmouseleave: move |_| nav.clear(),
                {children}
            }
        }
    }
}

/// The key of the [`LastUsedOption`] line.
pub const LAST_USED_KEY: &str = "last-used";

/// The "Last used" line that leads a picker editing a default, set apart from the list below it.
#[component]
pub fn LastUsedOption(
    nav: ListNav,
    highlighted: bool,
    selected: bool,
    #[props(into)] subtitle: String,
    onselect: EventHandler<()>,
) -> Element {
    rsx! {
        PopoverOption {
            id: nav.row_id(LAST_USED_KEY),
            title: "Last used",
            subtitle,
            role: OptionRole::Option,
            selected,
            highlighted,
            leading: rsx! {
                if selected {
                    Icon { name: IconName::Check, size: IconSize::Small }
                }
            },
            on_hover: move |()| nav.point(LAST_USED_KEY),
            onselect,
        }
        div { class: "psep", aria_hidden: "true" }
    }
}

/// What a [`ListPicker`] says when nothing is left to list, with one way out.
#[component]
pub fn PickerEmpty(
    #[props(into)] text: String,
    #[props(into)] action: String,
    on_action: EventHandler<()>,
) -> Element {
    rsx! {
        div { class: "pempty", role: "status",
            p { "{text}" }
            button {
                class: "btn plain sm",
                r#type: "button",
                onclick: move |_| on_action.call(()),
                "{action}"
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_first_arrow_press_lands_on_an_end() {
        assert_eq!(step(None, 3, true), Some(0));
        assert_eq!(step(None, 3, false), Some(2));
    }

    #[test]
    fn arrows_wrap_around() {
        assert_eq!(step(Some(2), 3, true), Some(0));
        assert_eq!(step(Some(0), 3, false), Some(2));
        assert_eq!(step(Some(1), 3, true), Some(2));
    }

    #[test]
    fn an_empty_list_has_nowhere_to_go() {
        assert_eq!(step(None, 0, true), None);
        assert_eq!(move_mark(&[], None, Motion::Down), None);
    }

    fn lines() -> Vec<NavRow> {
        vec![
            NavRow::single("a"),
            NavRow::with_inner("b", ["b1".to_owned(), "b2".to_owned()]),
            NavRow::single("c"),
        ]
    }

    #[test]
    fn up_and_down_land_on_the_line_itself_even_from_a_choice_inside_it() {
        assert_eq!(
            move_mark(&lines(), Some("b2"), Motion::Down),
            Some("c".to_owned())
        );
        assert_eq!(
            move_mark(&lines(), Some("b1"), Motion::Up),
            Some("a".to_owned())
        );
        assert_eq!(
            move_mark(&lines(), None, Motion::Down),
            Some("a".to_owned())
        );
    }

    #[test]
    fn left_and_right_walk_a_line_and_stop_at_its_ends() {
        assert_eq!(
            move_mark(&lines(), Some("b"), Motion::Right),
            Some("b1".to_owned())
        );
        assert_eq!(
            move_mark(&lines(), Some("b1"), Motion::Left),
            Some("b".to_owned())
        );
        assert_eq!(move_mark(&lines(), Some("b2"), Motion::Right), None);
        assert_eq!(move_mark(&lines(), Some("b"), Motion::Left), None);
        assert_eq!(move_mark(&lines(), Some("a"), Motion::Right), None);
        assert_eq!(move_mark(&lines(), None, Motion::Right), None);
    }
}
