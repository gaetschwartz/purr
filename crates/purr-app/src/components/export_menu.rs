use super::{
    button::{Button, ButtonVariant},
    icon::{Icon, IconName, IconSize},
    layer::PopoverAlign,
    popover::{Popover, PopoverOption},
    segmented::{Segment, Segmented},
    sheet::Sheet,
    toast::{Toasts, use_toasts},
};
use crate::{
    platform::{
        IS_WEB,
        export::{SaveOutcome, TextFile, copy_to_clipboard, save_text_file},
    },
    settings_store::use_settings,
    shortcuts::{Key, KeyChord, Shortcut, use_shortcut},
    transcribe::export_format::{ExportDocument, export_file_name, render},
    utils::counted::counted,
};
use dioxus::{core::spawn_forever, prelude::*};
use purr_common::{export_format::ExportFormat, platform::TranscriptSegment};
use std::rc::Rc;

const PREVIEW_CHARS: usize = 1400;

/// What a transcript export describes.
#[derive(Debug, Clone, PartialEq)]
pub struct ExportSource {
    pub file_name: String,
    pub model: String,
    /// The Whisper code of the spoken language, when known.
    pub language: Option<String>,
    pub duration_seconds: f64,
    pub segments: Rc<Vec<TranscriptSegment>>,
}

impl ExportSource {
    fn render(&self, format: ExportFormat) -> Result<String, String> {
        render(
            format,
            &ExportDocument {
                file_name: &self.file_name,
                model: &self.model,
                language: self.language.as_deref(),
                duration_seconds: self.duration_seconds,
                segments: &self.segments,
            },
        )
        .map_err(|error| format!("the {} could not be written: {error}", format.short_name()))
    }
}

fn save(source: &ExportSource, format: ExportFormat, toasts: Toasts) {
    let rendered = source.render(format);
    let file_name = export_file_name(format, &source.file_name);
    spawn_forever(async move {
        let saved = match rendered {
            Ok(content) => save_text_file(TextFile {
                file_name: &file_name,
                content_type: format.content_type(),
                content: &content,
            })
            .await
            .map_err(|error| error.to_string()),
            Err(error) => Err(error),
        };
        match saved {
            Ok(SaveOutcome::Saved) => {
                let detail = if IS_WEB {
                    "Your browser put it in your downloads."
                } else {
                    "The file is where you chose to save it."
                };
                toasts.success(format!("Saved {file_name}"), detail);
            }
            Ok(SaveOutcome::Cancelled) => {}
            Err(error) => {
                tracing::error!("Could not export {file_name}: {error}");
                toasts.error(format!("Couldn't save {file_name}"), error);
            }
        }
    });
}

/// Puts `format` of `source` on the clipboard and says how it went.
pub fn copy_export(source: &ExportSource, format: ExportFormat, toasts: Toasts) {
    let count = source.segments.len();
    let copying = source.render(format).map(|text| copy_to_clipboard(&text));
    spawn_forever(async move {
        let copied = match copying {
            Ok(copy) => copy.await.map_err(|error| error.to_string()),
            Err(error) => Err(error),
        };
        match copied {
            Ok(()) => {
                let what = match format {
                    ExportFormat::Txt => {
                        format!("{}, no timestamps.", counted(count, "segment", "segments"))
                    }
                    ExportFormat::Srt | ExportFormat::Vtt | ExportFormat::Json => {
                        format!("{} text is on the clipboard.", format.short_name())
                    }
                };
                toasts.success("Transcript copied", what);
            }
            Err(error) => {
                toasts.error(
                    "Couldn't copy the transcript",
                    format!("{error}. Use Export to save it as a file instead."),
                );
            }
        }
    });
}

/// The Export button: a menu of formats, then a sheet with a preview, Copy and Save.
#[component]
pub fn ExportMenu(
    /// The DOM id of the button, which the menu hangs from.
    #[props(into)]
    id: String,
    source: ExportSource,
) -> Element {
    let settings = use_settings();
    let preferred = settings.with(|settings| settings.ui.export_format);
    let mut menu_open = use_signal(|| false);
    let mut sheet = use_signal(|| None::<ExportFormat>);
    let can_export = !source.segments.is_empty();
    use_shortcut(
        Shortcut::new(KeyChord::primary(Key::Char('e')), "Export the transcript").in_inputs(),
        move || {
            if can_export {
                menu_open.set(false);
                sheet.set(Some(preferred));
            }
        },
    );

    rsx! {
        Button {
            variant: ButtonVariant::Primary,
            icon: IconName::Share,
            id: id.clone(),
            aria_haspopup: "menu",
            aria_expanded: menu_open(),
            disabled: source.segments.is_empty(),
            onclick: move |_| menu_open.toggle(),
            "Export"
        }
        if menu_open() {
            Popover {
                anchor: id,
                label: "Export format",
                menu: true,
                align: PopoverAlign::End,
                on_close: move |()| menu_open.set(false),
                div { class: "plist", role: "menu", aria_label: "Export format",
                    for format in ExportFormat::ALL {
                        PopoverOption {
                            key: "{format.extension()}",
                            title: format.label(),
                            subtitle: (format == preferred).then_some("Your default".to_owned()),
                            trailing: format.short_name(),
                            leading: rsx! {
                                Icon { name: if format == preferred { IconName::Check } else { IconName::Doc }, size: IconSize::Small }
                            },
                            onselect: move |()| {
                                menu_open.set(false);
                                sheet.set(Some(format));
                            },
                        }
                    }
                }
            }
        }
        if let Some(format) = sheet() {
            ExportSheet {
                source,
                format,
                on_format: move |next| sheet.set(Some(next)),
                on_close: move |()| sheet.set(None),
            }
        }
    }
}

/// The export sheet: a format switch, a preview, Copy and Save. Render it only while open.
#[component]
pub fn ExportSheet(
    source: ExportSource,
    format: ExportFormat,
    on_format: EventHandler<ExportFormat>,
    on_close: EventHandler<()>,
) -> Element {
    let toasts = use_toasts();
    let rendered = source.render(format);
    let preview = match &rendered {
        Ok(text) if text.chars().count() > PREVIEW_CHARS => {
            format!(
                "{}\n…",
                text.chars().take(PREVIEW_CHARS).collect::<String>()
            )
        }
        Ok(text) => text.clone(),
        Err(error) => format!("Nothing to preview: {error}"),
    };
    let file_name = export_file_name(format, &source.file_name);
    let copy_source = source.clone();
    rsx! {
        Sheet {
            title: "Export transcript",
            on_close,
            actions: rsx! {
                Button {
                    icon: IconName::Copy,
                    onclick: move |_| copy_export(&copy_source, format, toasts),
                    "Copy"
                }
                Button {
                    variant: ButtonVariant::Primary,
                    icon: IconName::Download,
                    disabled: rendered.is_err(),
                    onclick: move |_| {
                        on_close.call(());
                        save(&source, format, toasts);
                    },
                    "Save {file_name}"
                }
            },
            Segmented {
                label: "Format",
                full: true,
                value: format,
                segments: ExportFormat::ALL.into_iter().map(|format| Segment::new(format, format.short_name())).collect::<Vec<_>>(),
                onchange: move |next| on_format.call(next),
            }
            p { class: "foot", "{format.description()}" }
            pre { class: "preview", tabindex: "0", aria_label: "Preview", "{preview}" }
        }
    }
}
