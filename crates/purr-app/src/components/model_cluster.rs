use super::{
    icon::{Icon, IconName, IconSize},
    model_row::ModelListRow,
};
use crate::{hooks::use_disclosure::use_disclosure, model_catalog::clusters::ModelCluster};
use dioxus::prelude::*;

/// One model of the list with its other versions folded underneath.
///
/// `in_use` is the id of the model transcriptions use; `on_delete` asks the screen to confirm deleting one.
#[component]
pub fn ModelClusterRow(
    cluster: ModelCluster,
    #[props(default)] in_use: Option<String>,
    on_delete: EventHandler<String>,
) -> Element {
    let is_in_use = |id: &str| in_use.as_deref() == Some(id);
    let auto = cluster.variants_need_attention()
        || cluster.variants.iter().any(|row| is_in_use(&row.entry.id));
    let mut disclosure = use_disclosure(auto);
    let open = disclosure.is_open();
    let toggle_label = cluster.variants_label();
    let ModelCluster { primary, variants } = cluster;

    rsx! {
        div { class: "mcluster",
            ModelListRow {
                in_use: is_in_use(&primary.entry.id),
                row: primary,
                on_delete,
            }
            if !variants.is_empty() {
                button {
                    class: "mtoggle",
                    r#type: "button",
                    aria_expanded: "{open}",
                    onclick: move |_| disclosure.toggle(),
                    Icon {
                        name: if open { IconName::ChevronDown } else { IconName::ChevronRight },
                        size: IconSize::Small,
                    }
                    "{toggle_label}"
                }
                if open {
                    div { class: "mvariants",
                        for row in variants {
                            ModelListRow {
                                key: "{row.entry.id}",
                                in_use: is_in_use(&row.entry.id),
                                row,
                                on_delete,
                            }
                        }
                    }
                }
            }
        }
    }
}
