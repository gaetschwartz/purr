use super::{
    banner::{Banner, BannerTone},
    button::{Button, ButtonSize, ButtonVariant},
    chip::{Chip, ChipTone},
    icon::{Icon, IconName, IconSize},
    icon_button::IconButton,
    model_check::ModelCheck,
    model_download::ModelDownload,
    popover::{Popover, PopoverOption},
};
use crate::{
    hooks::use_models::use_models,
    model_catalog::{copy, format::checking, listing::ModelRow, state::ModelState},
    utils::{byte_size::ByteSize, dom_id::use_dom_id},
};
use dioxus::prelude::*;

const COMPACT_BANNER: &str = "margin-top:10px;padding:10px 12px";

/// One model of the list: name, hint, tags and the controls that fit its state.
///
/// `in_use` marks the model transcriptions use; `on_delete` asks the screen to confirm deleting it.
#[component]
pub fn ModelListRow(
    row: ModelRow,
    #[props(default)] in_use: bool,
    on_delete: EventHandler<String>,
) -> Element {
    let models = use_models();
    let mut menu_open = use_signal(|| false);
    let anchor = use_dom_id("model-menu");
    let ModelRow { entry, state } = row;
    let id = entry.id.clone();
    let label = entry.label.clone();
    let size = ByteSize(entry.size_bytes).to_string();
    let downloading_elsewhere =
        models.is_downloading() && !matches!(state, ModelState::Downloading(_));

    let (disabled, hint) = match &state {
        ModelState::Unavailable { reason, .. } => {
            (true, Some(copy::unavailable(*reason).1.to_owned()))
        }
        ModelState::NotDownloaded
        | ModelState::Downloaded
        | ModelState::Downloading(_)
        | ModelState::Verifying(_)
        | ModelState::Failed(_)
        | ModelState::NeedsRepair(_) => (false, entry.hint()),
    };
    let can_verify = !matches!(state, ModelState::NeedsRepair(_));

    let menu = rsx! {
        IconButton {
            icon: IconName::More,
            label: "More actions for {label}",
            id: "{anchor}",
            aria_haspopup: "menu",
            aria_expanded: "{menu_open()}",
            onclick: move |_| menu_open.set(!menu_open()),
        }
        if menu_open() {
            Popover {
                anchor,
                label: "Actions for {label}",
                menu: true,
                on_close: move |()| menu_open.set(false),
                if can_verify {
                    PopoverOption {
                        title: "Check files",
                        leading: rsx! {
                            Icon { name: IconName::Shield, size: IconSize::Small }
                        },
                        onselect: {
                            let id = id.clone();
                            move |()| {
                                menu_open.set(false);
                                models.verify(&id);
                            }
                        },
                    }
                }
                PopoverOption {
                    title: "Delete…",
                    trailing: "{size}",
                    danger: true,
                    leading: rsx! {
                        Icon { name: IconName::Trash, size: IconSize::Small }
                    },
                    onselect: {
                        let id = id.clone();
                        move |()| {
                            menu_open.set(false);
                            on_delete.call(id.clone());
                        }
                    },
                }
            }
        }
    };

    let side = match &state {
        ModelState::NotDownloaded => rsx! {
            Button {
                size: ButtonSize::Small,
                icon: IconName::Download,
                disabled: downloading_elsewhere,
                title: if downloading_elsewhere { "Wait for the current download to finish" },
                aria_label: "Get {label}, {size}",
                onclick: {
                    let id = id.clone();
                    move |_| models.download(&id)
                },
                "Get"
            }
        },
        ModelState::Downloaded => rsx! {
            if in_use {
                Chip { tone: ChipTone::Accent, "In use" }
            }
            span { class: "state-ok", "Downloaded" }
            {menu}
        },
        ModelState::Downloading(_) => rsx! {},
        ModelState::Verifying(_) => rsx! {},
        ModelState::Failed(failure) => {
            let words = copy::failure(failure);
            let tone = if words.is_connection {
                ChipTone::Warn
            } else {
                ChipTone::Bad
            };
            let icon = if words.is_connection {
                IconName::WifiOff
            } else {
                IconName::XCircle
            };
            rsx! {
                Chip { tone, icon,
                    "{words.chip}"
                }
                Button {
                    size: ButtonSize::Small,
                    variant: ButtonVariant::Primary,
                    icon: IconName::Retry,
                    disabled: downloading_elsewhere,
                    aria_label: "Retry downloading {label}",
                    onclick: {
                        let id = id.clone();
                        move |_| models.download(&id)
                    },
                    "Retry"
                }
                Button {
                    size: ButtonSize::Small,
                    variant: ButtonVariant::Plain,
                    aria_label: "Dismiss the failed download of {label}",
                    onclick: {
                        let id = id.clone();
                        move |_| models.dismiss(&id)
                    },
                    "Dismiss"
                }
            }
        }
        ModelState::NeedsRepair(_) => rsx! {
            Button {
                size: ButtonSize::Small,
                variant: ButtonVariant::Primary,
                disabled: downloading_elsewhere,
                aria_label: "Download again to repair {label}",
                onclick: {
                    let id = id.clone();
                    move |_| models.download_again(&id)
                },
                "Download again"
            }
            {menu}
        },
        ModelState::Unavailable { reason, installed } => {
            let (chip, _) = copy::unavailable(*reason);
            rsx! {
                Chip { "{chip}" }
                if *installed {
                    {menu}
                }
            }
        }
    };

    let notice = match &state {
        ModelState::Failed(failure) => {
            let words = copy::failure(failure);
            let tone = if words.is_connection {
                BannerTone::Warn
            } else {
                BannerTone::Bad
            };
            rsx! {
                Banner {
                    tone,
                    title: "{words.title}",
                    role: "alert",
                    style: COMPACT_BANNER,
                    "{words.body}"
                }
            }
        }
        ModelState::NeedsRepair(damage) => rsx! {
            Banner {
                tone: BannerTone::Warn,
                icon: IconName::Warning,
                title: "Integrity check failed",
                role: "alert",
                style: COMPACT_BANNER,
                {copy::damage(damage)}
            }
        },
        ModelState::NotDownloaded
        | ModelState::Downloaded
        | ModelState::Downloading(_)
        | ModelState::Verifying(_)
        | ModelState::Unavailable { .. } => rsx! {},
    };

    let class = if disabled { "mrow disabled" } else { "mrow" };
    rsx! {
        div { class, role: "group", aria_label: "{label}",
            div { style: "min-width:0",
                div { class: "nm", "{entry.label}" }
                if let Some(hint) = hint {
                    div { class: "hint", "{hint}" }
                }
                div { class: "tags",
                    Chip { "{entry.language.label()}" }
                    if let Some(tag) = entry.precision.tag() {
                        Chip { "{tag}" }
                    }
                }
                {notice}
            }
            div { class: "side",
                span { class: "size", "{size}" }
                {side}
            }
            match &state {
                ModelState::Downloading(progress) => rsx! {
                    ModelDownload {
                        progress: *progress,
                        label: label.clone(),
                        on_cancel: move |()| models.cancel(&id),
                    }
                },
                ModelState::Verifying(progress) => rsx! {
                    ModelCheck {
                        progress: *progress,
                        label: label.clone(),
                        text: checking("files", progress),
                    }
                },
                ModelState::NotDownloaded
                | ModelState::Downloaded
                | ModelState::Failed(_)
                | ModelState::NeedsRepair(_)
                | ModelState::Unavailable { .. } => rsx! {},
            }
        }
    }
}
