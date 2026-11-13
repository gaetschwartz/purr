use super::{
    icon::{Icon, IconName, IconSize},
    model_cluster::ModelClusterRow,
};
use crate::{
    hooks::use_disclosure::use_disclosure, model_catalog::clusters::ModelCluster,
    utils::dom_id::use_dom_id,
};
use dioxus::prelude::*;

/// The collapsed "More models" section of the Models screen, for models a newer one replaces.
///
/// `open_by_default` opens it until the user chooses otherwise; `in_use` is the id of the model
/// transcriptions use.
#[component]
pub fn ModelMore(
    clusters: Vec<ModelCluster>,
    open_by_default: bool,
    #[props(default)] in_use: Option<String>,
    on_delete: EventHandler<String>,
) -> Element {
    let heading_id = use_dom_id("models-more");
    let mut disclosure = use_disclosure(open_by_default);
    let open = disclosure.is_open();
    let count = clusters.len();

    rsx! {
        section { aria_labelledby: "{heading_id}",
            div { class: "fam-h",
                h2 { id: "{heading_id}",
                    button {
                        class: "fam-toggle",
                        r#type: "button",
                        aria_expanded: "{open}",
                        onclick: move |_| disclosure.toggle(),
                        Icon {
                            name: if open { IconName::ChevronDown } else { IconName::ChevronRight },
                            size: IconSize::Small,
                        }
                        "More models"
                        span { class: "count", "{count}" }
                    }
                }
            }
            if open {
                div { class: "group surface",
                    for cluster in clusters {
                        ModelClusterRow {
                            key: "{cluster.primary.entry.id}",
                            cluster,
                            in_use: in_use.clone(),
                            on_delete,
                        }
                    }
                }
            }
        }
    }
}
