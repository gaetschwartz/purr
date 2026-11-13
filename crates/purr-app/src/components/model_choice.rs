use super::{
    icon::{Icon, IconName, IconSize},
    list_picker::ListNav,
};
use crate::{
    model_catalog::{
        clusters::ModelCluster,
        copy,
        entry::{ModelEntry, Precision},
        format::{check_percent, percent},
        listing::ModelRow,
        state::ModelState,
    },
    utils::byte_size::ByteSize,
};
use dioxus::prelude::*;

/// The size of a model that would have to be downloaded, or how far its download is.
#[must_use]
pub fn download_size(row: &ModelRow) -> Option<String> {
    match &row.state {
        ModelState::Downloading(progress) => {
            Some(percent(progress).map_or_else(|| "…".to_owned(), |done| format!("{done}%")))
        }
        ModelState::Downloaded
        | ModelState::Unavailable {
            installed: true, ..
        } => None,
        ModelState::NotDownloaded
        | ModelState::Verifying(_)
        | ModelState::Failed(_)
        | ModelState::NeedsRepair(_)
        | ModelState::Unavailable {
            installed: false, ..
        } => Some(ByteSize(row.entry.size_bytes).to_string()),
    }
}

fn facts(entry: &ModelEntry) -> String {
    let language = entry.language.label();
    entry.group.trade_off().map_or_else(
        || language.to_owned(),
        |trade| format!("{language} · {trade}"),
    )
}

fn subtitle(row: &ModelRow) -> String {
    match &row.state {
        ModelState::Unavailable { reason, .. } => copy::unavailable(*reason).1.to_owned(),
        ModelState::Downloading(_) => "Downloading…".to_owned(),
        ModelState::Verifying(progress) => check_percent(progress).map_or_else(
            || "Checking files…".to_owned(),
            |done| format!("Checking files… {done}%"),
        ),
        ModelState::Failed(_) => "Last download failed".to_owned(),
        ModelState::NeedsRepair(_) => "Files damaged".to_owned(),
        ModelState::NotDownloaded | ModelState::Downloaded => facts(&row.entry),
    }
}

fn version_tag(precision: &Precision) -> String {
    match precision {
        Precision::Quantized { bits } => format!("Q{bits}"),
        Precision::Other(raw) => raw.clone(),
        Precision::Full | Precision::Mixed => "Full".to_owned(),
    }
}

/// Picks `row`, or makes it the default when Ctrl or Cmd is held and `on_alt` is given.
fn choose(
    event: &MouseEvent,
    row: &ModelRow,
    onselect: EventHandler<ModelRow>,
    on_alt: Option<EventHandler<ModelRow>>,
) {
    if matches!(row.state, ModelState::Unavailable { .. }) {
        return;
    }
    let modifiers = event.modifiers();
    match on_alt {
        Some(alt) if modifiers.ctrl() || modifiers.meta() => alt.call(row.clone()),
        Some(_) | None => onselect.call(row.clone()),
    }
}

/// One model of the model picker: a check when selected, the name, what it is good at or what is
/// happening to it, and its download size when it is not on this device.
#[component]
pub fn ModelChoice(
    row: ModelRow,
    nav: ListNav,
    selected: bool,
    highlighted: bool,
    hint: Option<String>,
    onselect: EventHandler<ModelRow>,
    on_alt: Option<EventHandler<ModelRow>>,
) -> Element {
    let disabled = matches!(row.state, ModelState::Unavailable { .. });
    let size = download_size(&row);
    let downloading = matches!(row.state, ModelState::Downloading(_));
    let class = if highlighted { "opt kb" } else { "opt" };
    let id = row.entry.id.clone();
    let title = row.entry.label.clone();
    let detail = subtitle(&row);
    rsx! {
        button {
            id: nav.row_id(&id),
            class,
            r#type: "button",
            role: "option",
            title: hint,
            aria_selected: selected,
            aria_disabled: disabled.then_some("true"),
            onmouseenter: move |_| nav.point(&id),
            onclick: move |event: MouseEvent| choose(&event, &row, onselect, on_alt),
            span { class: "ck",
                if selected {
                    Icon { name: IconName::Check, size: IconSize::Small }
                }
            }
            span { class: "t",
                b { "{title}" }
                span { "{detail}" }
            }
            if let Some(size) = size {
                span { class: "r",
                    if downloading {
                        span { class: "sr", "downloaded " }
                    } else {
                        Icon { name: IconName::Download, size: IconSize::Small }
                        span { class: "sr", "download size " }
                    }
                    "{size}"
                }
            }
        }
    }
}

/// A model to download with its other versions as small choices underneath.
#[component]
pub fn ClusterChoice(
    cluster: ModelCluster,
    nav: ListNav,
    selected: String,
    active: Option<String>,
    hint: Option<String>,
    onselect: EventHandler<ModelRow>,
    on_alt: Option<EventHandler<ModelRow>>,
) -> Element {
    let is_active = |id: &str| active.as_deref() == Some(id);
    let ModelCluster { primary, variants } = cluster;
    let name = primary.entry.name.clone();
    rsx! {
        ModelChoice {
            selected: primary.entry.id == selected,
            highlighted: is_active(&primary.entry.id),
            row: primary,
            nav,
            hint: hint.clone(),
            onselect,
            on_alt,
        }
        if !variants.is_empty() {
            div {
                class: "pvars",
                role: "group",
                aria_label: "Other versions of {name}",
                for row in variants {
                    VersionChoice {
                        key: "{row.entry.id}",
                        selected: row.entry.id == selected,
                        highlighted: is_active(&row.entry.id),
                        row,
                        nav,
                        hint: hint.clone(),
                        onselect,
                        on_alt,
                    }
                }
            }
        }
    }
}

#[component]
fn VersionChoice(
    row: ModelRow,
    nav: ListNav,
    selected: bool,
    highlighted: bool,
    hint: Option<String>,
    onselect: EventHandler<ModelRow>,
    on_alt: Option<EventHandler<ModelRow>>,
) -> Element {
    let disabled = matches!(row.state, ModelState::Unavailable { .. });
    let tag = version_tag(&row.entry.precision);
    let size = download_size(&row);
    let class = match (selected, highlighted) {
        (true, true) => "pvar sel kb",
        (true, false) => "pvar sel",
        (false, true) => "pvar kb",
        (false, false) => "pvar",
    };
    let name = size.as_ref().map_or_else(
        || row.entry.label.clone(),
        |size| format!("{}, {size}", row.entry.label),
    );
    let id = row.entry.id.clone();
    rsx! {
        button {
            id: nav.row_id(&id),
            class,
            r#type: "button",
            role: "option",
            title: hint,
            aria_label: name,
            aria_selected: selected,
            aria_disabled: disabled.then_some("true"),
            onmouseenter: move |_| nav.point(&id),
            onclick: move |event: MouseEvent| choose(&event, &row, onselect, on_alt),
            if selected {
                Icon { name: IconName::Check, size: IconSize::Small }
            }
            b { "{tag}" }
            if let Some(size) = size {
                span { "{size}" }
            }
        }
    }
}
