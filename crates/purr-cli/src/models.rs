//! `models` subcommand: download, list, delete, inspect and verify Whisper models

use crate::cli::ModelCommands;
use crate::color::Stream;
use crate::global;
use crate::theme::{ProgressPart, Themed as _, progress_style};
use indicatif::{HumanBytes, HumanDuration, ProgressBar, ProgressStyle};
use miette::IntoDiagnostic as _;
use purr_core::ModelManager;
use purr_core::model::registry::ModelSpec;
use purr_core::model::status::{ModelFileStatus, ModelIntegrity};
use purr_core::model::vad_or_none;
use std::fmt::Write as _;
use tracing::debug;

fn download_bar_template(colors: bool) -> String {
    format!(
        "{{spinner{}}} [{{elapsed_precise}}] [{{wide_bar{}}}] {{bytes}}/{{total_bytes}} ({{bytes_per_sec}}, {{eta}})",
        progress_style(ProgressPart::Spinner, colors),
        progress_style(ProgressPart::Bar, colors)
    )
}

fn download_spinner_template(colors: bool) -> String {
    format!(
        "{{spinner{}}} [{{elapsed_precise}}] {{bytes}} downloaded... {{msg}}",
        progress_style(ProgressPart::Spinner, colors)
    )
}

fn verify_spinner_template(colors: bool) -> String {
    format!(
        "{{spinner{}}} [{{elapsed_precise}}] {{msg}}",
        progress_style(ProgressPart::Spinner, colors)
    )
}

fn stderr_colors() -> bool {
    global::colors().enabled(Stream::Stderr)
}

/// Handle model management subcommands
pub async fn handle_model_command(command: &ModelCommands) -> miette::Result<()> {
    let model_manager = ModelManager::new()?;

    match command {
        ModelCommands::Download { model, force } => {
            let spec = model_manager.model(model)?;
            download(&model_manager, spec, *force).await?;
            ensure_vad(&model_manager).await;
            Ok(())
        }
        ModelCommands::List { available } => {
            if *available {
                list_available(&model_manager).await
            } else {
                list_downloaded(&model_manager).await
            }
        }
        ModelCommands::Delete { model } => {
            let spec = model_manager.model(model)?;
            delete(&model_manager, spec).await
        }
        ModelCommands::Info { model } => {
            let spec = model_manager.model(model)?;
            info(&model_manager, spec).await
        }
        ModelCommands::Verify { model } => verify(&model_manager, model.as_deref()).await,
    }
}

/// Fetches the voice activity model that goes with the Whisper models; a failure is only logged.
pub async fn ensure_vad(manager: &ModelManager) {
    let fetched = manager.ensure_vad_model(None).await.map(Some);
    if let Some(path) = vad_or_none(fetched) {
        debug!("The voice activity model is at {}", path.display());
    }
}

async fn download(manager: &ModelManager, spec: &ModelSpec, force: bool) -> miette::Result<()> {
    if !force && manager.is_model_downloaded(spec).await? {
        println!(
            "{} Model {} is already downloaded.",
            "Info:".info(),
            spec.id
        );
        return Ok(());
    }

    println!(
        "{} Downloading model: {} ({}, {})",
        "Info:".info(),
        spec.id,
        spec.description,
        HumanBytes(spec.file.size_bytes)
    );

    let colors = stderr_colors();
    let bar_style = ProgressStyle::with_template(&download_bar_template(colors))
        .into_diagnostic()?
        .progress_chars("#>-");
    let spinner_style =
        ProgressStyle::with_template(&download_spinner_template(colors)).into_diagnostic()?;
    let progress_bar = ProgressBar::new(0);
    progress_bar.set_style(bar_style);

    manager
        .download_model_with_progress(spec, None, |downloaded, total| {
            if let Some(total) = total {
                if progress_bar.length().unwrap_or(0) != total {
                    progress_bar.set_length(total);
                }
            } else {
                progress_bar.set_style(spinner_style.clone());
            }
            progress_bar.set_position(downloaded);
        })
        .await?;

    let elapsed = progress_bar.elapsed();
    let per_sec = progress_bar
        .length()
        .map(|len| len as f64 / elapsed.as_secs_f64());
    progress_bar.finish_and_clear();

    let average = per_sec.map_or_else(String::new, |per_sec| {
        format!(" ({}/s avg)", HumanBytes(per_sec as u64).quantity())
    });
    println!(
        "{} Model downloaded and verified in {:#}{average}.",
        "Success:".success(),
        HumanDuration(elapsed).quantity(),
    );
    Ok(())
}

async fn list_downloaded(manager: &ModelManager) -> miette::Result<()> {
    let local = manager.local_models().await?;

    if local.is_empty() {
        println!("{} No models downloaded yet.", "Info:".info());
        println!(
            "Use {} to download a model.",
            format_args!("{} models download base", env!("CARGO_PKG_NAME")).command()
        );
        return Ok(());
    }

    println!("{} Downloaded models:", "Info:".info());
    println!();

    for (spec, status) in &local {
        let path = manager.get_model_path(spec);
        match status {
            ModelFileStatus::Present => println!(
                "  {} - {} ({})",
                spec.id.name(),
                spec.description.muted(),
                HumanBytes(spec.file.size_bytes).quantity()
            ),
            ModelFileStatus::SizeMismatch { expected, actual } => println!(
                "  {} - {} (has {}, expected {}; download it again)",
                spec.id.negative(),
                "incomplete".negative(),
                HumanBytes(*actual).quantity(),
                HumanBytes(*expected).quantity()
            ),
            ModelFileStatus::Missing => {}
        }

        if global::verbosity().is_verbose() {
            println!("    Path: {}", path.display().path());
        }
    }

    println!();
    println!(
        "XDG data directory: {}",
        manager.models_dir().display().path()
    );
    Ok(())
}

async fn list_available(manager: &ModelManager) -> miette::Result<()> {
    println!("{}", "Available Whisper Models:".heading());
    println!();

    let registry = manager.registry();
    let id_width = registry
        .models()
        .iter()
        .map(|model| model.id.len())
        .max()
        .unwrap_or_default();

    for group in registry.groups() {
        println!(
            "  {} - {}",
            group.family.to_string().name(),
            group.family.summary()
        );

        for spec in group.models {
            let mut notes = Vec::new();
            if spec.id == registry.default_model().id {
                notes.push("recommended".to_owned());
            }
            for feature in &spec.features {
                notes.push(feature.summary().to_owned());
            }
            match manager.file_status(spec).await? {
                ModelFileStatus::Present => notes.push("downloaded".positive().to_string()),
                ModelFileStatus::SizeMismatch { .. } => {
                    notes.push("incomplete".negative().to_string());
                }
                ModelFileStatus::Missing => {}
            }

            let mut line = format!(
                "    {}  {:<12}  {:<5}  {:>11}",
                format!("{:<id_width$}", spec.id).path(),
                spec.language.to_string(),
                spec.quantization.to_string(),
                HumanBytes(spec.file.size_bytes).to_string().quantity(),
            );
            if !notes.is_empty() {
                write!(line, "  {}", notes.join(", ").muted()).into_diagnostic()?;
            }
            println!("{line}");
        }
        println!();
    }

    println!(
        "{}{}",
        "Usage: ".muted(),
        format_args!("{} models download <model>", env!("CARGO_PKG_NAME")).command()
    );
    println!(
        "{}{}",
        "Example: ".muted(),
        format_args!(
            "{} models download {}",
            env!("CARGO_PKG_NAME"),
            registry.default_model().id
        )
        .command()
    );
    Ok(())
}

async fn delete(manager: &ModelManager, spec: &ModelSpec) -> miette::Result<()> {
    match manager.file_status(spec).await? {
        ModelFileStatus::Missing => {
            println!(
                "{} Model {} is not downloaded.",
                "Warning:".warning(),
                spec.id
            );
            return Ok(());
        }
        ModelFileStatus::Present | ModelFileStatus::SizeMismatch { .. } => {}
    }

    manager.delete_model(spec).await?;

    println!(
        "{} Model {} deleted successfully.",
        "Success:".success(),
        spec.id
    );
    Ok(())
}

async fn info(manager: &ModelManager, spec: &ModelSpec) -> miette::Result<()> {
    println!("{} Model Information", "Info:".info());
    println!();
    println!("Name: {}", spec.id.name());
    println!("Description: {}", spec.description);
    println!("Family: {}", spec.family);
    println!("Language: {}", spec.language);
    println!("Quantization: {}", spec.quantization);
    if spec.features.is_empty() {
        println!("Features: none");
    } else {
        let features: Vec<String> = spec.features.iter().map(ToString::to_string).collect();
        println!("Features: {}", features.join(", "));
    }
    println!("Filename: {}", spec.filename.path());
    println!(
        "Size: {} ({} bytes)",
        HumanBytes(spec.file.size_bytes).to_string().quantity(),
        spec.file.size_bytes
    );
    println!("SHA-256: {}", spec.file.sha256);
    println!("Source: {} @ {}", spec.source.repo, spec.source.revision);
    println!("URL: {}", spec.url());

    let path = manager.get_model_path(spec);
    match manager.file_status(spec).await? {
        ModelFileStatus::Present => {
            println!("Downloaded: {}", "yes".positive());
            println!("Path: {}", path.display().path());
        }
        ModelFileStatus::SizeMismatch { expected, actual } => {
            println!(
                "Downloaded: {} (file has {}, expected {})",
                "incomplete".negative(),
                HumanBytes(actual),
                HumanBytes(expected)
            );
            println!("Path: {}", path.display().path());
        }
        ModelFileStatus::Missing => println!("Downloaded: {}", "no".negative()),
    }
    Ok(())
}

async fn verify(manager: &ModelManager, model: Option<&str>) -> miette::Result<()> {
    let specs: Vec<&ModelSpec> = if let Some(id) = model {
        vec![manager.model(id)?]
    } else {
        manager
            .local_models()
            .await?
            .into_iter()
            .map(|(spec, _)| spec)
            .collect()
    };

    if specs.is_empty() {
        println!("{} No models downloaded yet.", "Info:".info());
        return Ok(());
    }

    let spinner_style = ProgressStyle::with_template(&verify_spinner_template(stderr_colors()))
        .into_diagnostic()?;
    let mut failed = 0usize;

    for spec in specs {
        let spinner = ProgressBar::new_spinner();
        spinner.set_style(spinner_style.clone());
        spinner.set_message(format!(
            "Verifying {} ({})",
            spec.id,
            HumanBytes(spec.file.size_bytes)
        ));
        spinner.enable_steady_tick(std::time::Duration::from_millis(100));
        let integrity = manager.verify_model(spec).await;
        spinner.finish_and_clear();

        match integrity? {
            ModelIntegrity::Verified => {
                println!("{} {}", "ok     ".success(), spec.id);
            }
            ModelIntegrity::Missing => {
                println!("{} {}", "missing".warning(), spec.id);
            }
            ModelIntegrity::SizeMismatch { expected, actual } => {
                failed += 1;
                println!(
                    "{} {}: {} bytes, expected {} bytes",
                    "corrupt".error(),
                    spec.id,
                    actual,
                    expected
                );
            }
            ModelIntegrity::ChecksumMismatch { expected, actual } => {
                failed += 1;
                println!(
                    "{} {}: SHA-256 {actual}, expected {expected}",
                    "corrupt".error(),
                    spec.id,
                );
            }
        }
    }

    if failed > 0 {
        return Err(miette::miette!(
            help = "Download the affected models again with 'models download <model> --force'",
            "{failed} model(s) failed verification"
        ));
    }
    Ok(())
}
