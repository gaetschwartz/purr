//! The browser's WebGPU adapter as plain rows.

use crate::hooks::use_models::GpuSupport;

/// The status badge at the end of a row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Badge {
    Available,
    Unavailable,
}

impl Badge {
    /// The words on the badge.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Available => "Available",
            Self::Unavailable => "Unavailable",
        }
    }
}

/// One line of the panel: a backend, the device it uses or why it cannot run, and its badge.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SystemRow {
    pub backend: &'static str,
    pub name: String,
    pub detail: Option<String>,
    pub badge: Badge,
}

/// The rows for the browser's WebGPU adapter.
#[must_use]
pub fn browser_rows(gpu: &GpuSupport) -> Vec<SystemRow> {
    let (name, detail, badge) = match gpu {
        GpuSupport::Checking => ("Checking this browser…".to_owned(), None, Badge::Unavailable),
        GpuSupport::Unavailable => (
            "No WebGPU adapter found".to_owned(),
            Some(
                "Purr uses the CPU here, which is slower. Use a current Chrome, Edge or another browser with WebGPU enabled."
                    .to_owned(),
            ),
            Badge::Unavailable,
        ),
        GpuSupport::Available { name, shader_f16 } => (
            name.clone()
                .unwrap_or_else(|| "WebGPU adapter".to_owned()),
            Some(if *shader_f16 {
                "16-bit floats supported".to_owned()
            } else {
                "No 16-bit floats: the Large v3 Turbo model is unavailable".to_owned()
            }),
            Badge::Available,
        ),
    };
    vec![SystemRow {
        backend: "WebGPU",
        name,
        detail,
        badge,
    }]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(backend: &'static str, name: &str, detail: Option<&str>, badge: Badge) -> SystemRow {
        SystemRow {
            backend,
            name: name.to_owned(),
            detail: detail.map(str::to_owned),
            badge,
        }
    }

    #[test]
    fn the_browser_reports_its_adapter_and_what_it_lacks() {
        let with = |shader_f16| {
            browser_rows(&GpuSupport::Available {
                name: Some("Some Adapter".to_owned()),
                shader_f16,
            })
        };
        assert_eq!(
            with(true),
            [row(
                "WebGPU",
                "Some Adapter",
                Some("16-bit floats supported"),
                Badge::Available
            )]
        );
        assert_eq!(
            with(false),
            [row(
                "WebGPU",
                "Some Adapter",
                Some("No 16-bit floats: the Large v3 Turbo model is unavailable"),
                Badge::Available
            )]
        );
    }

    #[test]
    fn a_browser_without_webgpu_cannot_transcribe() {
        let rows = browser_rows(&GpuSupport::Unavailable);
        assert_eq!(
            rows.iter().map(|row| row.badge).collect::<Vec<_>>(),
            [Badge::Unavailable]
        );
        assert_eq!(
            rows.first().map(|row| row.name.as_str()),
            Some("No WebGPU adapter found")
        );
    }
}
