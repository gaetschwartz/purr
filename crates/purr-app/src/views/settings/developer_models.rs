use super::{
    changes::report, developer_add_browser::AddBrowserModel, developer_add_native::AddNativeModel,
};
use crate::{
    components::{
        button::{Button, ButtonSize, ButtonVariant},
        group::{Group, Item},
        toast::use_toasts,
    },
    platform::{self, IS_WEB},
    settings_store::use_settings,
};
use dioxus::prelude::*;
use purr_common::{
    developer::model::{BrowserOrigin, DeveloperSource},
    platform::Platform as _,
};

fn describe(source: &DeveloperSource) -> String {
    match source {
        DeveloperSource::Browser(model) => {
            let origin = match &model.origin {
                BrowserOrigin::HuggingFace { repo, revision } => format!("{repo} @ {revision}"),
                BrowserOrigin::LocalFiles => "files from this computer".to_owned(),
            };
            format!(
                "{origin} · encoder {} · decoder {}",
                model.encoder_dtype, model.decoder_dtype
            )
        }
        DeveloperSource::Native(model) => model.path.display().to_string(),
    }
}

/// The models loaded outside the catalog, and the form that adds one.
#[component]
pub fn DeveloperModelsGroup() -> Element {
    let settings = use_settings();
    let toasts = use_toasts();
    let models = settings.with(|settings| settings.developer.models.clone());

    rsx! {
        Group {
            header: "Developer models",
            footer: "Loaded without a manifest entry and without hash checks. They appear in the model picker marked as developer models.",
            for model in models {
                Item {
                    key: "{model.id.as_str()}",
                    title: model.name.clone(),
                    subtitle: describe(&model.source),
                    control: rsx! {
                        Button {
                            size: ButtonSize::Small,
                            variant: ButtonVariant::Danger,
                            onclick: {
                                let id = model.id.as_str().to_owned();
                                move |_| {
                                    let id = id.clone();
                                    spawn(async move {
                                        if IS_WEB && let Err(error) = platform::get_platform().remove_model(&id).await {
                                            toasts.error("Couldn't remove the model's files", error.to_string());
                                        }
                                        report(
                                            toasts,
                                            settings.update(|settings| settings.remove_developer_model(&id)),
                                        );
                                    });
                                }
                            },
                            "Remove"
                        }
                    },
                }
            }
        }
        if IS_WEB {
            AddBrowserModel {}
        } else {
            AddNativeModel {}
        }
    }
}
