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
    /// The `dark` class the Tailwind `dark:` utilities of the older views key on.
    pub dark_class: bool,
    /// `data-transparency="reduced"`.
    pub reduced_transparency: bool,
}

/// Maps the chosen appearance and the system colour scheme onto the root attributes.
#[must_use]
pub const fn root_attributes(appearance: Appearance, system_dark: bool) -> RootAttributes {
    let (data_theme, dark_class) = match appearance.theme {
        Theme::Light => ("light", false),
        Theme::Dark => ("dark", true),
        Theme::System => ("system", system_dark),
    };
    RootAttributes {
        data_theme,
        dark_class,
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
        "const root = document.documentElement; root.lang = 'en'; root.setAttribute('data-theme', '{theme}'); {transparency} root.classList.toggle('dark', {});",
        attributes.dark_class
    )
}

const SYSTEM_SCHEME_WATCHER: &str = "const query = window.matchMedia('(prefers-color-scheme: dark)'); dioxus.send(query.matches); query.addEventListener('change', (event) => dioxus.send(event.matches)); await new Promise(() => {});";

/// Keeps the document root in step with the theme and transparency held by the settings store.
pub fn use_appearance_provider(store: SettingsStore) {
    let appearance = use_memo(move || {
        store.with(|settings| Appearance {
            theme: settings.ui.theme,
            reduce_transparency: settings.ui.reduce_transparency,
        })
    });
    let mut system_dark = use_signal(|| false);

    use_future(move || async move {
        let mut watcher = document::eval(SYSTEM_SCHEME_WATCHER);
        while let Ok(dark) = watcher.recv::<bool>().await {
            system_dark.set(dark);
        }
    });

    use_effect(move || {
        if store.load_state() == LoadState::Loading {
            return;
        }
        let attributes = root_attributes(appearance(), system_dark());
        document::eval(&root_script(attributes));
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn system_theme_follows_the_os_for_tailwind() {
        let appearance = Appearance::default();
        assert_eq!(
            root_attributes(appearance, true),
            RootAttributes {
                data_theme: "system",
                dark_class: true,
                reduced_transparency: false,
            }
        );
        assert!(!root_attributes(appearance, false).dark_class);
    }

    #[test]
    fn explicit_themes_win_over_the_os() {
        let dark = Appearance {
            theme: Theme::Dark,
            reduce_transparency: true,
        };
        assert_eq!(
            root_attributes(dark, false),
            RootAttributes {
                data_theme: "dark",
                dark_class: true,
                reduced_transparency: true,
            }
        );
        let light = Appearance {
            theme: Theme::Light,
            reduce_transparency: false,
        };
        assert_eq!(
            root_attributes(light, true),
            RootAttributes {
                data_theme: "light",
                dark_class: false,
                reduced_transparency: false,
            }
        );
    }
}
