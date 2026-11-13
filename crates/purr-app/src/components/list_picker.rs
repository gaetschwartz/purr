use super::{
    icon::{Icon, IconName, IconSize},
    popover::{OptionRole, Popover, PopoverOption},
    search_field::SearchField,
};
use crate::utils::dom_id::use_dom_id;
use dioxus::prelude::*;

/// Which row of a [`ListPicker`] the keyboard and the pointer point at; there is only ever one.
///
/// Rows are named by a key unique in the list, so the pointed row survives the list changing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ListNav {
    list_id: Signal<String>,
    pointed: Signal<Option<String>>,
}

/// A [`ListNav`] for one picker; nothing is pointed at until the pointer or the arrows move.
pub fn use_list_nav() -> ListNav {
    let list_id = use_dom_id("list");
    ListNav {
        list_id: use_signal(|| list_id),
        pointed: use_signal(|| None),
    }
}

impl ListNav {
    /// The DOM id of the list box.
    #[must_use]
    pub fn list_id(&self) -> String {
        self.list_id.read().clone()
    }

    /// The DOM id of the row with `key`.
    #[must_use]
    pub fn row_id(&self, key: &str) -> String {
        format!("{}-{key}", self.list_id.read())
    }

    /// The key of the row to mark among `keys`: the pointed one, else the first while a search
    /// narrows the list, so Enter picks the best match.
    #[must_use]
    pub fn active(&self, keys: &[String], searching: bool) -> Option<String> {
        let pointed = self.pointed.read();
        pointed
            .as_ref()
            .filter(|key| keys.contains(key))
            .or_else(|| keys.first().filter(|_| searching))
            .cloned()
    }

    /// Points at the row with `key`.
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

/// The index an arrow press moves to among `count` rows from `current`, wrapping at both ends.
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

/// A popover (a sheet on phones) for choosing one row of a list, shared by the model and language
/// pickers: an optional pinned search field and note, the scrolling list box, and a pinned footer.
///
/// `keys` names the rows in the order they show; arrows move through them from the search field
/// and Enter calls `on_enter` with the marked key. Rows mark themselves with `nav.active(..)` and
/// call `nav.point(..)` on pointer moves, so the pointer and the keyboard share one mark.
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
    keys: Vec<String>,
    on_enter: EventHandler<String>,
    /// The search field's placeholder; without one there is no search field.
    #[props(into)]
    search: Option<String>,
    #[props(into, default)] query: String,
    on_query: Option<EventHandler<String>>,
    /// A line under the search field.
    note: Option<Element>,
    /// What shows instead of rows, such as a loading or empty state.
    status: Option<Element>,
    footer: Option<Element>,
    children: Element,
) -> Element {
    let searching = !query.trim().is_empty();
    let active = nav.active(&keys, searching);
    let active_id = active.as_deref().map(|key| nav.row_id(key));
    let list_id = nav.list_id();

    let on_key = move |event: KeyboardEvent| {
        let down = match event.key() {
            Key::Named(NamedKey::ArrowDown) => true,
            Key::Named(NamedKey::ArrowUp) => false,
            Key::Named(NamedKey::Enter) => {
                if let Some(key) = &active {
                    event.prevent_default();
                    on_enter.call(key.clone());
                }
                return;
            }
            Key::Named(_) | Key::Character(_) => return,
        };
        event.prevent_default();
        let current = active
            .as_ref()
            .and_then(|key| keys.iter().position(|row| row == key));
        let Some(key) = step(current, keys.len(), down).and_then(|index| keys.get(index)) else {
            return;
        };
        nav.point(key);
        document::eval(&format!(
            "document.getElementById({:?})?.scrollIntoView({{block:'nearest'}})",
            nav.row_id(key)
        ));
    };

    let header = match (&search, note) {
        (None, None) => None,
        (search, note) => Some(rsx! {
            if let Some(placeholder) = search {
                SearchField {
                    value: query,
                    placeholder: placeholder.clone(),
                    label: placeholder.clone(),
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

/// The key of the [`LastUsedOption`] row.
pub const LAST_USED_KEY: &str = "last-used";

/// The "Last used" row that leads a picker editing a default, set apart from the list below it.
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

/// A titled run of rows inside a [`ListPicker`].
#[component]
pub fn PickerSection(#[props(into)] title: String, children: Element) -> Element {
    let heading_id = use_dom_id("picker-section");
    rsx! {
        div { role: "group", aria_labelledby: "{heading_id}",
            div { class: "ph", id: "{heading_id}", "{title}" }
            {children}
        }
    }
}

/// What a [`ListPicker`] says when its search leaves nothing, with a way back to the full list.
#[component]
pub fn PickerEmpty(#[props(into)] text: String, on_clear: EventHandler<()>) -> Element {
    rsx! {
        div { class: "pempty", role: "status",
            p { "{text}" }
            button {
                class: "btn plain sm",
                r#type: "button",
                onclick: move |_| on_clear.call(()),
                "Clear search"
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
    }
}
