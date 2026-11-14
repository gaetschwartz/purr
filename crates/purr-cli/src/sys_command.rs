//! `purr sys`: the report is the command's result, on stdout in every mode.

use crate::engine_status;
use crate::resultln;
use crate::runtime;
use crate::sys_report;
use crate::theme::{self, Role, Themed as _};
use miette::IntoDiagnostic as _;
use purr_core::SystemInfo;
use purr_core::accel::backend::AccelBackend;
use purr_core::accel::engine_report;
use purr_core::dev::FeatureStatus;
use std::borrow::Cow;

fn feature_status(
    backend: AccelBackend,
    feature: &FeatureStatus,
) -> theme::Painted<Cow<'static, str>> {
    match feature {
        FeatureStatus::Disabled => Role::Caution.paint(Cow::Borrowed("Not built")),
        FeatureStatus::RuntimeLibraryMissing { library, .. } => {
            let hint = backend
                .runtime_library_hint()
                .map(|hint| format!("; {hint}"))
                .unwrap_or_default();
            Role::Negative.paint(Cow::Owned(format!(
                "Built, but the runtime library {library} cannot be loaded{hint}"
            )))
        }
        FeatureStatus::EnabledButNotAvailable => {
            Role::Negative.paint(Cow::Borrowed("Built, but no usable device"))
        }
        FeatureStatus::Available(_) => Role::Positive.paint(Cow::Borrowed("Available")),
    }
}

pub fn handle(json: bool) -> miette::Result<()> {
    let reports = engine_report::reports(&crate::engines::all(), &[]);
    let sys = SystemInfo::of(&reports);

    if json {
        resultln!("{}", sys_report::json(&sys, &reports).into_diagnostic()?)?;
        return Ok(());
    }

    resultln!("{} {}", crate::APP_NAME, crate::build_info::version())?;
    resultln!()?;
    resultln!(
        "{}",
        "Acceleration backends (default priority order, best first):".heading()
    )?;
    resultln!()?;
    let longest = AccelBackend::DEFAULT_PRIORITY
        .iter()
        .map(|b| b.pretty_name().len())
        .max()
        .unwrap_or(0);

    for backend in AccelBackend::DEFAULT_PRIORITY {
        let status = sys
            .backends
            .get(backend)
            .unwrap_or(&FeatureStatus::Disabled);
        resultln!(
            "  * {:longest$} - {}",
            backend.pretty_name(),
            feature_status(*backend, status)
        )?;
        if let FeatureStatus::RuntimeLibraryMissing { searched, .. } = status {
            resultln!("    {}", runtime::searched_summary(searched).muted())?;
        }
    }
    resultln!()?;

    if sys_report::is_union(&reports) {
        resultln!("{}", "Engines:".heading())?;
        resultln!()?;
        for report in &reports {
            for line in engine_status::lines(report) {
                resultln!("{line}")?;
            }
        }
        resultln!()?;
    }

    resultln!(
        "{}",
        "Devices (pick one with --device INDEX, or --device BACKEND/TYPE/ID):".heading()
    )?;
    resultln!()?;

    let blocks = sys_report::device_blocks(&sys, &reports);
    if blocks.is_empty() {
        resultln!("{}", "No devices found.".warning())?;
        resultln!()?;
    } else {
        for block in blocks {
            resultln!("{}", block.headline.name())?;
            for line in block.details {
                resultln!("{line}")?;
            }
            resultln!()?;
        }
    }

    resultln!("{}", "CPU:".heading())?;
    resultln!()?;
    let mut lines = sys_report::cpu_lines(&sys.cpu).into_iter();
    if let Some(headline) = lines.next() {
        resultln!("{}", headline.name())?;
    }
    for line in lines {
        resultln!("{line}")?;
    }

    Ok(())
}
