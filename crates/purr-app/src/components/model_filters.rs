use super::{
    search_field::SearchField,
    segmented::{Segment, Segmented},
    select::{Select, SelectOption},
};
use crate::model_catalog::{
    family::ModelGroup,
    listing::{ModelFilter, ModelQuery},
};
use dioxus::prelude::*;

/// The filter row of the Models screen: language and storage filter, size class, and search.
///
/// `groups` are the size classes that exist on this platform; `onchange` gets the whole new query.
#[component]
pub fn ModelFilters(
    query: ModelQuery,
    groups: Vec<ModelGroup>,
    onchange: EventHandler<ModelQuery>,
) -> Element {
    let segments = ModelFilter::ALL
        .into_iter()
        .map(|filter| Segment::new(filter, filter.label()))
        .collect();
    let mut options = vec![SelectOption::new(None, "All sizes")];
    options.extend(
        groups
            .into_iter()
            .map(|group| SelectOption::new(Some(group), group.label())),
    );

    rsx! {
        div { class: "filters",
            Segmented {
                segments,
                value: query.filter,
                label: "Filter models",
                onchange: {
                    let query = query.clone();
                    move |filter| {
                        onchange
                            .call(ModelQuery {
                                filter,
                                ..query.clone()
                            });
                    }
                },
            }
            Select {
                options,
                value: query.group,
                label: "Show one model size",
                onchange: {
                    let query = query.clone();
                    move |group| {
                        onchange
                            .call(ModelQuery {
                                group,
                                ..query.clone()
                            });
                    }
                },
            }
            SearchField {
                value: query.search.clone(),
                placeholder: "Search models",
                label: "Search models",
                oninput: move |search| {
                    onchange
                        .call(ModelQuery {
                            search,
                            ..query.clone()
                        });
                },
            }
        }
    }
}
