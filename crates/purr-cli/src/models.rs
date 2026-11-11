//! `models` subcommand: download, list, delete and inspect Whisper models

use crate::cli::ModelCommands;
use crate::fmt::Verbosity;
use indicatif::{HumanBytes, HumanDuration, ProgressBar, ProgressStyle};
use miette::IntoDiagnostic as _;
use owo_colors::OwoColorize as _;
use purr_core::ModelManager;
use purr_core::model::registry::ModelSpec;
use purr_core::model::status::ModelFileStatus;
use std::fmt::Write as _;

const DOWNLOAD_BAR_TEMPLATE: &str = "{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec}, {eta})";
const DOWNLOAD_SPINNER_TEMPLATE: &str =
    "{spinner:.green} [{elapsed_precise}] {bytes} downloaded... {msg}";

/// Handle model management subcommands
pub async fn handle_model_command(
    command: ModelCommands,
    verbosity: Verbosity,
) -> miette::Result<()> {
    let model_manager = ModelManager::new()?;

    match command {
        ModelCommands::Download { model, force } => {
            let spec = model_manager.model(&model)?;
            download(&model_manager, spec, force).await
        }
        ModelCommands::List { available } => {
            if available {
                list_available(&model_manager).await
            } else {
                list_downloaded(&model_manager, verbosity).await
            }
        }
        ModelCommands::Delete { model } => {
            let spec = model_manager.model(&model)?;
            delete(&model_manager, spec).await
        }
        ModelCommands::Info { model } => {
            let spec = model_manager.model(&model)?;
            info(&model_manager, spec).await
        }
    }
}

async fn download(manager: &ModelManager, spec: &ModelSpec, force: bool) -> miette::Result<()> {
    if !force && manager.is_model_downloaded(spec).await? {
        println!(
            "{} Model {} is already downloaded.",
            "Info:".blue().bold(),
            spec.id
        );
        return Ok(());
    }

    println!(
        "{} Downloading model: {} ({}, {})",
        "Info:".blue().bold(),
        spec.id,
        spec.description,
        HumanBytes(spec.size_bytes)
    );

    let bar_style = ProgressStyle::with_template(DOWNLOAD_BAR_TEMPLATE)
        .into_diagnostic()?
        .progress_chars("#>-");
    let spinner_style =
        ProgressStyle::with_template(DOWNLOAD_SPINNER_TEMPLATE).into_diagnostic()?;
    let progress_bar = ProgressBar::new(0);
    progress_bar.set_style(bar_style);

    manager
        .download_model_with_progress(spec, |downloaded, total| {
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
        format!(
            " ({}{} avg)",
            HumanBytes(per_sec as u64).cyan(),
            "/s".cyan()
        )
    });
    println!(
        "{} Model downloaded and verified in {:#}{average}.",
        "Success:".green().bold(),
        HumanDuration(elapsed).cyan(),
    );
    Ok(())
}

async fn list_downloaded(manager: &ModelManager, verbosity: Verbosity) -> miette::Result<()> {
    let local = manager.local_models().await?;

    if local.is_empty() {
        println!("{} No models downloaded yet.", "Info:".blue().bold());
        println!(
            "Use {}{} models download <model> to download a model.",
            env!("CARGO_PKG_NAME").cyan(),
            " models download base".cyan()
        );
        return Ok(());
    }

    println!("{} Downloaded models:", "Info:".blue().bold());
    println!();

    for (spec, status) in &local {
        let path = manager.get_model_path(spec);
        match status {
            ModelFileStatus::Present => println!(
                "  {} - {} ({})",
                spec.id.green(),
                spec.description.dimmed(),
                HumanBytes(spec.size_bytes).yellow()
            ),
            ModelFileStatus::SizeMismatch { expected, actual } => println!(
                "  {} - {} (has {}, expected {}; download it again)",
                spec.id.red(),
                "incomplete".red(),
                HumanBytes(*actual).yellow(),
                HumanBytes(*expected).yellow()
            ),
            ModelFileStatus::Missing => {}
        }

        if verbosity.is_verbose() {
            println!("    Path: {}", path.display().to_string().dimmed());
        }
    }

    println!();
    println!(
        "XDG data directory: {}",
        manager.models_dir().display().to_string().dimmed()
    );
    Ok(())
}

async fn list_available(manager: &ModelManager) -> miette::Result<()> {
    println!("{}", "Available Whisper Models:".blue().bold());
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
            group.family.to_string().green().bold(),
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
                ModelFileStatus::Present => notes.push("downloaded".green().to_string()),
                ModelFileStatus::SizeMismatch { .. } => notes.push("incomplete".red().to_string()),
                ModelFileStatus::Missing => {}
            }

            let mut line = format!(
                "    {}  {:<12}  {:<5}  {:>11}",
                format!("{:<id_width$}", spec.id).cyan(),
                spec.language.to_string(),
                spec.quantization.to_string(),
                HumanBytes(spec.size_bytes).to_string().yellow(),
            );
            if !notes.is_empty() {
                write!(line, "  {}", notes.join(", ").dimmed()).into_diagnostic()?;
            }
            println!("{line}");
        }
        println!();
    }

    println!(
        "{}{}{}",
        "Usage: ".dimmed(),
        env!("CARGO_PKG_NAME").cyan().dimmed(),
        " models download <model>".cyan().dimmed()
    );
    println!(
        "{}{}{}",
        "Example: ".dimmed(),
        env!("CARGO_PKG_NAME").cyan().dimmed(),
        format_args!(" models download {}", registry.default_model().id)
            .cyan()
            .dimmed()
    );
    Ok(())
}

async fn delete(manager: &ModelManager, spec: &ModelSpec) -> miette::Result<()> {
    if matches!(manager.file_status(spec).await?, ModelFileStatus::Missing) {
        println!(
            "{} Model {} is not downloaded.",
            "Warning:".yellow().bold(),
            spec.id
        );
        return Ok(());
    }

    manager.delete_model(spec).await?;

    println!(
        "{} Model {} deleted successfully.",
        "Success:".green().bold(),
        spec.id
    );
    Ok(())
}

async fn info(manager: &ModelManager, spec: &ModelSpec) -> miette::Result<()> {
    println!("{} Model Information", "Info:".blue().bold());
    println!();
    println!("Name: {}", spec.id.green().bold());
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
    println!("Filename: {}", spec.filename.yellow());
    println!(
        "Size: {} ({} bytes)",
        HumanBytes(spec.size_bytes).to_string().yellow(),
        spec.size_bytes
    );
    println!("SHA-256: {}", spec.sha256);
    println!("Source: {} @ {}", spec.source.repo, spec.source.revision);
    println!("URL: {}", spec.url());

    let path = manager.get_model_path(spec);
    match manager.file_status(spec).await? {
        ModelFileStatus::Present => {
            println!("Downloaded: {}", "yes".green());
            println!("Path: {}", path.display());
        }
        ModelFileStatus::SizeMismatch { expected, actual } => {
            println!(
                "Downloaded: {} (file has {}, expected {})",
                "incomplete".red(),
                HumanBytes(actual),
                HumanBytes(expected)
            );
            println!("Path: {}", path.display());
        }
        ModelFileStatus::Missing => println!("Downloaded: {}", "no".red()),
    }
    Ok(())
}
