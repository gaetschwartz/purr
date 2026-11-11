//! Theme and transparency preferences, applied as attributes on the document root.

use crate::{components::toast::Toasts, platform};
use dioxus::prelude::*;
use futures::StreamExt as _;
use purr_common::{
    platform::{Platform as _, PlatformError},
    settings::Theme,
};

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
        "const root = document.documentElement; root.setAttribute('data-theme', '{theme}'); {transparency} root.classList.toggle('dark', {});",
        attributes.dark_class
    )
}

const SYSTEM_SCHEME_WATCHER: &str = "const query = window.matchMedia('(prefers-color-scheme: dark)'); dioxus.send(query.matches); query.addEventListener('change', (event) => dioxus.send(event.matches)); await new Promise(() => {});";

/// Read and change the appearance from any view, from [`use_appearance`].
#[derive(Clone, Copy, PartialEq)]
pub struct AppearanceHandle {
    appearance: Signal<Appearance>,
    saver: Coroutine<Appearance>,
}

impl AppearanceHandle {
    /// The current choice (subscribes the caller).
    #[must_use]
    pub fn get(&self) -> Appearance {
        *self.appearance.read()
    }

    /// The current choice, without subscribing the caller.
    #[must_use]
    pub fn peek(&self) -> Appearance {
        *self.appearance.peek()
    }

    /// Replaces the whole appearance and saves it.
    pub fn set(mut self, appearance: Appearance) {
        self.appearance.set(appearance);
        self.saver.send(appearance);
    }

    /// Switches the theme and saves it.
    pub fn set_theme(self, theme: Theme) {
        self.set(Appearance {
            theme,
            ..self.peek()
        });
    }

    /// Turns the solid surfaces on or off and saves the choice.
    pub fn set_reduce_transparency(self, reduce_transparency: bool) {
        self.set(Appearance {
            reduce_transparency,
            ..self.peek()
        });
    }
}

/// The last of a burst of requests; earlier ones are superseded.
fn newest<T>(first: T, rest: impl IntoIterator<Item = T>) -> T {
    rest.into_iter().last().unwrap_or(first)
}

async fn save(chosen: Appearance) -> Result<(), PlatformError> {
    let platform = platform::get_platform();
    let mut settings = platform.load_settings().await?;
    settings.ui.theme = chosen.theme;
    settings.ui.reduce_transparency = chosen.reduce_transparency;
    platform.save_settings(&settings).await
}

/// Loads the saved appearance, keeps the root attributes in sync and provides the handle.
pub fn use_appearance_provider(toasts: Toasts) -> AppearanceHandle {
    let mut appearance = use_signal(Appearance::default);
    let mut system_dark = use_signal(|| false);

    use_future(move || async move {
        match platform::get_platform().load_settings().await {
            Ok(settings) => appearance.set(Appearance {
                theme: settings.ui.theme,
                reduce_transparency: settings.ui.reduce_transparency,
            }),
            Err(error) => {
                tracing::warn!("Could not load the appearance, using the defaults: {error}");
            }
        }
    });

    use_future(move || async move {
        let mut watcher = document::eval(SYSTEM_SCHEME_WATCHER);
        while let Ok(dark) = watcher.recv::<bool>().await {
            system_dark.set(dark);
        }
    });

    use_effect(move || {
        let attributes = root_attributes(appearance(), system_dark());
        document::eval(&root_script(attributes));
    });

    let saver = use_coroutine(
        move |mut requests: UnboundedReceiver<Appearance>| async move {
            while let Some(first) = requests.next().await {
                let pending = std::iter::from_fn(|| requests.try_recv().ok());
                if let Err(error) = save(newest(first, pending)).await {
                    tracing::error!("Could not save the appearance: {error}");
                    toasts.error("Couldn't save the appearance", error.to_string());
                }
            }
        },
    );

    use_context_provider(|| AppearanceHandle { appearance, saver })
}

/// The appearance handle provided at the root.
pub fn use_appearance() -> AppearanceHandle {
    use_context::<AppearanceHandle>()
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
    fn a_burst_of_saves_keeps_only_the_newest() {
        assert_eq!(newest(1, [2, 3]), 3);
        assert_eq!(newest(1, []), 1);
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
