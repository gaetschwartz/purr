use dioxus::prelude::*;

/// The glyphs of the design system, drawn on a 24 x 24 grid with a 1.7 stroke.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IconName {
    Wave,
    Clock,
    Cube,
    Cog,
    Logs,
    Lock,
    Shield,
    Upload,
    Download,
    ChevronDown,
    ChevronUp,
    ChevronRight,
    UpDown,
    Check,
    Close,
    Search,
    Copy,
    Share,
    Play,
    Pause,
    Trash,
    Pencil,
    Globe,
    Cpu,
    Bolt,
    Sun,
    Moon,
    Auto,
    Info,
    Warning,
    XCircle,
    CheckCircle,
    WifiOff,
    Retry,
    More,
    Keyboard,
    Doc,
    Text,
    List,
    Minus,
    Plus,
    Music,
    Stop,
    Window,
    Desktop,
    Gauge,
    Speaker,
    Translate,
    Thermometer,
    Hash,
    Stack,
    File,
    Reset,
    Back,
    Cat,
}

impl IconName {
    /// The SVG children of the glyph.
    #[must_use]
    pub const fn paths(self) -> &'static str {
        match self {
            Self::Wave => r#"<path d="M4 10.5v3M8 7.5v9M12 4.5v15M16 8v8M20 10.5v3"/>"#,
            Self::Clock => r#"<circle cx="12" cy="12" r="8.5"/><path d="M12 7.5V12l3 2"/>"#,
            Self::Cube => {
                r#"<path d="M12 3.5l7.5 4.2v8.6L12 20.5l-7.5-4.2V7.7z"/><path d="M4.7 7.8L12 12l7.3-4.2M12 12v8.3"/>"#
            }
            Self::Cog => {
                r#"<path d="M10.4 5.6L10.6 3.3H13.4L13.6 5.6L15.4 6.3L17.2 4.9L19.1 6.8L17.7 8.6L18.4 10.4L20.7 10.6V13.4L18.4 13.6L17.7 15.4L19.1 17.2L17.2 19.1L15.4 17.7L13.6 18.4L13.4 20.7H10.6L10.4 18.4L8.6 17.7L6.8 19.1L4.9 17.2L6.3 15.4L5.6 13.6L3.3 13.4V10.6L5.6 10.4L6.3 8.6L4.9 6.8L6.8 4.9L8.6 6.3z"/><circle cx="12" cy="12" r="2.8"/>"#
            }
            Self::Logs => {
                r#"<rect x="5" y="3.5" width="14" height="17" rx="3"/><path d="M9 8.5h6M9 12h6M9 15.5h3.5"/>"#
            }
            Self::Lock => {
                r#"<rect x="5" y="10.5" width="14" height="10" rx="3"/><path d="M8.5 10.5V8a3.5 3.5 0 0 1 7 0v2.5"/>"#
            }
            Self::Shield => {
                r#"<path d="M12 3.5l7 2.6v5.4c0 4.3-2.9 7.6-7 9-4.1-1.4-7-4.7-7-9V6.1z"/><path d="M9 12l2.2 2.2L15.3 10"/>"#
            }
            Self::Upload => {
                r#"<path d="M12 15V4.5M7.5 9L12 4.5 16.5 9M5 14.5V17a2.5 2.5 0 0 0 2.5 2.5h9A2.5 2.5 0 0 0 19 17v-2.5"/>"#
            }
            Self::Download => {
                r#"<path d="M12 4.5V15M7.5 10.5L12 15l4.5-4.5M5 14.5V17a2.5 2.5 0 0 0 2.5 2.5h9A2.5 2.5 0 0 0 19 17v-2.5"/>"#
            }
            Self::ChevronDown => r#"<path d="M7 10l5 5 5-5"/>"#,
            Self::ChevronUp => r#"<path d="M7 14l5-5 5 5"/>"#,
            Self::ChevronRight => r#"<path d="M10 7l5 5-5 5"/>"#,
            Self::UpDown => r#"<path d="M8.5 9.5L12 6l3.5 3.5M8.5 14.5L12 18l3.5-3.5"/>"#,
            Self::Check => r#"<path d="M5.5 12.5l4 4 9-9"/>"#,
            Self::Close => r#"<path d="M7 7l10 10M17 7L7 17"/>"#,
            Self::Search => r#"<circle cx="11" cy="11" r="6.5"/><path d="M16 16l4 4"/>"#,
            Self::Copy => {
                r#"<rect x="8.5" y="8.5" width="11" height="11" rx="2.5"/><path d="M15.5 8.5V6.5a2 2 0 0 0-2-2h-7a2 2 0 0 0-2 2v7a2 2 0 0 0 2 2h2"/>"#
            }
            Self::Share => {
                r#"<path d="M12 3.5v11M8 7.5l4-4 4 4M8.5 10.5H7a2.5 2.5 0 0 0-2.5 2.5v5A2.5 2.5 0 0 0 7 20.5h10a2.5 2.5 0 0 0 2.5-2.5v-5a2.5 2.5 0 0 0-2.5-2.5h-1.5"/>"#
            }
            Self::Play => {
                r#"<path d="M8.5 5.8v12.4a.8.8 0 0 0 1.2.7l10-6.2a.8.8 0 0 0 0-1.4l-10-6.2a.8.8 0 0 0-1.2.7z"/>"#
            }
            Self::Pause => {
                r#"<rect x="6.5" y="5.5" width="4" height="13" rx="1.2"/><rect x="13.5" y="5.5" width="4" height="13" rx="1.2"/>"#
            }
            Self::Trash => {
                r#"<path d="M4.5 7h15M9.5 7V5.5A1.5 1.5 0 0 1 11 4h2a1.5 1.5 0 0 1 1.5 1.5V7M6.5 7l.8 11.2a2 2 0 0 0 2 1.8h5.4a2 2 0 0 0 2-1.8L17.5 7"/>"#
            }
            Self::Pencil => r#"<path d="M15.5 5.5l3 3L9 18H6v-3zM13.5 7.5l3 3"/>"#,
            Self::Globe => {
                r#"<circle cx="12" cy="12" r="8.5"/><path d="M3.5 12h17M12 3.5c2.4 2.6 3.5 5.4 3.5 8.5s-1.1 5.9-3.5 8.5c-2.4-2.6-3.5-5.4-3.5-8.5S9.6 6.1 12 3.5z"/>"#
            }
            Self::Cpu => {
                r#"<rect x="7" y="7" width="10" height="10" rx="2"/><rect x="10" y="10" width="4" height="4" rx=".8"/><path d="M9.5 4v3M14.5 4v3M9.5 17v3M14.5 17v3M4 9.5h3M4 14.5h3M17 9.5h3M17 14.5h3"/>"#
            }
            Self::Bolt => r#"<path d="M13 3.5L5.5 13.5H12l-1 7 7.5-10H12z"/>"#,
            Self::Sun => {
                r#"<circle cx="12" cy="12" r="3.8"/><path d="M12 3.5v1.8M12 18.7v1.8M3.5 12h1.8M18.7 12h1.8M6 6l1.3 1.3M16.7 16.7L18 18M6 18l1.3-1.3M16.7 7.3L18 6"/>"#
            }
            Self::Moon => r#"<path d="M19 14.5A7.5 7.5 0 1 1 9.5 5a6 6 0 0 0 9.5 9.5z"/>"#,
            Self::Auto => {
                r#"<circle cx="12" cy="12" r="8"/><path d="M12 4a8 8 0 0 1 0 16z" fill="currentColor"/>"#
            }
            Self::Info => r#"<circle cx="12" cy="12" r="8.5"/><path d="M12 11v5M12 8h.01"/>"#,
            Self::Warning => {
                r#"<path d="M10.3 5.2a2 2 0 0 1 3.4 0l6.6 11.5a2 2 0 0 1-1.7 3H5.4a2 2 0 0 1-1.7-3z"/><path d="M12 10v3.8M12 16.8h.01"/>"#
            }
            Self::XCircle => {
                r#"<circle cx="12" cy="12" r="8.5"/><path d="M9.5 9.5l5 5M14.5 9.5l-5 5"/>"#
            }
            Self::CheckCircle => {
                r#"<circle cx="12" cy="12" r="8.5"/><path d="M8.5 12.3l2.4 2.4 4.6-4.9"/>"#
            }
            Self::WifiOff => {
                r#"<path d="M4 9.2a12 12 0 0 1 3.6-2.2M12.8 6.1A12 12 0 0 1 20 9.2M7 12.6a7.5 7.5 0 0 1 3.5-1.9M15.4 11.3a7.5 7.5 0 0 1 1.6 1.3M9.8 16a3.2 3.2 0 0 1 4.4 0M4 4l16 16M12 19.3h.01"/>"#
            }
            Self::Retry => r#"<path d="M19.5 12a7.5 7.5 0 1 1-2.2-5.3M19.5 4.5v4h-4"/>"#,
            Self::More => {
                r#"<circle cx="6.5" cy="12" r="1.2" fill="currentColor"/><circle cx="12" cy="12" r="1.2" fill="currentColor"/><circle cx="17.5" cy="12" r="1.2" fill="currentColor"/>"#
            }
            Self::Keyboard => {
                r#"<rect x="3.5" y="6.5" width="17" height="11" rx="2.5"/><path d="M7 10h.01M10 10h.01M13 10h.01M16.5 10h.5M8 14h8"/>"#
            }
            Self::Doc => {
                r#"<path d="M7 3.5h6.5L18 8v10a2.5 2.5 0 0 1-2.5 2.5h-8.5A2.5 2.5 0 0 1 4.5 18V6A2.5 2.5 0 0 1 7 3.5z"/><path d="M13.5 3.5V8H18M8.5 12.5h7M8.5 16h5"/>"#
            }
            Self::Text => r#"<path d="M5 7h14M5 11.5h14M5 16h9"/>"#,
            Self::List => r#"<path d="M10 7h10M10 12h10M10 17h10M5 7h.01M5 12h.01M5 17h.01"/>"#,
            Self::Minus => r#"<path d="M5 12h14"/>"#,
            Self::Plus => r#"<path d="M12 5v14M5 12h14"/>"#,
            Self::Music => {
                r#"<path d="M9 17.5V6.2l10-2v11.3"/><circle cx="6.5" cy="17.5" r="2.5"/><circle cx="16.5" cy="15.5" r="2.5"/>"#
            }
            Self::Stop => r#"<rect x="7" y="7" width="10" height="10" rx="2.2"/>"#,
            Self::Window => {
                r#"<rect x="3.5" y="5" width="17" height="14" rx="2.5"/><path d="M3.5 9h17M6.5 7h.01M8.5 7h.01"/>"#
            }
            Self::Desktop => {
                r#"<rect x="3.5" y="4.5" width="17" height="11.5" rx="2.2"/><path d="M9 20h6M12 16v4"/>"#
            }
            Self::Gauge => r#"<path d="M4.5 16a7.5 7.5 0 1 1 15 0"/><path d="M12 16l3.5-4.5"/>"#,
            Self::Speaker => {
                r#"<path d="M5 9.5h3l4-3.5v12l-4-3.5H5z"/><path d="M15.5 9a4 4 0 0 1 0 6M18 6.5a7.5 7.5 0 0 1 0 11"/>"#
            }
            Self::Translate => {
                r#"<path d="M4 6h9M8.5 4v2M6 6c.6 3 2.6 5.4 5.5 6.6M11 6c-.8 3.6-3.2 6-6.5 7.3M12.5 20l3.5-8 3.5 8M13.6 17.5h4.8"/>"#
            }
            Self::Thermometer => {
                r#"<path d="M10 13.5V5.5a2 2 0 0 1 4 0v8a4 4 0 1 1-4 0z"/><path d="M12 9v6.5"/>"#
            }
            Self::Hash => r#"<path d="M9 4l-2 16M17 4l-2 16M4.5 9h15M4 15h15"/>"#,
            Self::Stack => {
                r#"<path d="M12 4l8 4-8 4-8-4z"/><path d="M4 12l8 4 8-4M4 16l8 4 8-4"/>"#
            }
            Self::File => {
                r#"<path d="M7 3.5h6.5L18 8v10a2.5 2.5 0 0 1-2.5 2.5h-8.5A2.5 2.5 0 0 1 4.5 18V6A2.5 2.5 0 0 1 7 3.5z"/><path d="M13.5 3.5V8H18"/>"#
            }
            Self::Reset => r#"<path d="M4.5 12a7.5 7.5 0 1 0 2.2-5.3M4.5 4.5v4h4"/>"#,
            Self::Back => r#"<path d="M14.5 6.5L9 12l5.5 5.5"/>"#,
            Self::Cat => {
                r#"<path d="M5 18V9.2L7.6 5.5l2.6 3.3h3.6l2.6-3.3L19 9.2V18a2.5 2.5 0 0 1-2.5 2.5h-9A2.5 2.5 0 0 1 5 18z"/><path d="M9.2 13.2v2.6M12 11.8v5.4M14.8 13.2v2.6"/>"#
            }
        }
    }
}

/// Rendered size of an icon: 16, 20, 28 or 40 px.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum IconSize {
    Small,
    #[default]
    Medium,
    Large,
    ExtraLarge,
}

impl IconSize {
    const fn class(self) -> &'static str {
        match self {
            Self::Small => "i sm",
            Self::Medium => "i",
            Self::Large => "i lg",
            Self::ExtraLarge => "i xl",
        }
    }
}

/// An inline SVG glyph, hidden from assistive technology unless it is given a `label`.
#[component]
pub fn Icon(
    name: IconName,
    #[props(default)] size: IconSize,
    #[props(into, default)] class: String,
    #[props(into)] label: Option<String>,
) -> Element {
    let classes = if class.is_empty() {
        size.class().to_owned()
    } else {
        format!("{} {class}", size.class())
    };
    match label {
        Some(label) => rsx! {
            svg {
                class: classes,
                view_box: "0 0 24 24",
                role: "img",
                "aria-label": label,
                dangerous_inner_html: name.paths(),
            }
        },
        None => rsx! {
            svg {
                class: classes,
                view_box: "0 0 24 24",
                "aria-hidden": "true",
                dangerous_inner_html: name.paths(),
            }
        },
    }
}
