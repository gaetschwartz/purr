//! Theme and transparency preferences, applied as attributes on the document root.

use crate::settings_store::{SettingsStore, status::LoadState};
use dioxus::prelude::*;
use purr_common::settings::Theme;

/// The look the user chose.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Appearance {
    pub theme: Theme,
    pub reduce_transparency: bool,
}

impl Default for Appearance {
    fn default() -> Self {
        Self {
            theme: Theme::System,
            reduce_transparency: false,
        }
    }
}

/// What the document root carries for an [`Appearance`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RootAttributes {
    /// `data-theme`: `light`, `dark` or `system`.
    pub data_theme: &'static str,
    /// `data-transparency="reduced"`.
    pub reduced_transparency: bool,
}

/// Maps the chosen appearance onto the root attributes; the stylesheet follows the OS for `system`.
#[must_use]
pub const fn root_attributes(appearance: Appearance) -> RootAttributes {
    let data_theme = match appearance.theme {
        Theme::Light => "light",
        Theme::Dark => "dark",
        Theme::System => "system",
    };
    RootAttributes {
        data_theme,
        reduced_transparency: appearance.reduce_transparency,
    }
}

fn root_script(attributes: RootAttributes) -> String {
    let theme = attributes.data_theme;
    let transparency = if attributes.reduced_transparency {
        "root.setAttribute('data-transparency', 'reduced');"
    } else {
        "root.removeAttribute('data-transparency');"
    };
    format!(
        "const root = document.documentElement; root.lang = 'en'; root.setAttribute('data-theme', '{theme}'); {transparency}"
    )
}

/// Keeps the document root in step with the theme and transparency held by the settings store.
pub fn use_appearance_provider(store: SettingsStore) {
    let appearance = use_memo(move || {
        store.with(|settings| Appearance {
            theme: settings.ui.theme,
            reduce_transparency: settings.ui.reduce_transparency,
        })
    });

    use_effect(move || {
        if store.load_state() == LoadState::Loading {
            return;
        }
        document::eval(&root_script(root_attributes(appearance())));
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_theme_names_itself_on_the_root() {
        for (theme, expected) in [
            (Theme::System, "system"),
            (Theme::Light, "light"),
            (Theme::Dark, "dark"),
        ] {
            let appearance = Appearance {
                theme,
                reduce_transparency: false,
            };
            assert_eq!(root_attributes(appearance).data_theme, expected);
        }
    }

    #[test]
    fn reduced_transparency_reaches_the_root() {
        let appearance = Appearance {
            theme: Theme::Dark,
            reduce_transparency: true,
        };
        assert_eq!(
            root_attributes(appearance),
            RootAttributes {
                data_theme: "dark",
                reduced_transparency: true,
            }
        );
    }
}
