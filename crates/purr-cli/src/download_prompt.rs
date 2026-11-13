//! Offers to download a model when a transcription finds none.

use crate::models;
use crate::output;
use crate::output::sink::stdin_is_terminal;
use crate::theme::Themed as _;
use crate::{status, statusln};
use indicatif::{HumanBytes, HumanDuration};
use miette::IntoDiagnostic as _;
use purr_core::ModelManager;
use purr_core::math::ByteSpeed;
use purr_core::model::registry::ModelSpec;
use std::io;
use std::path::PathBuf;
use tracing::error;

/// Whether anyone can see and answer the question.
pub fn can_ask() -> bool {
    output::is_interactive() && stdin_is_terminal()
}

pub async fn prompt_for_model_download(
    model_manager: &ModelManager,
    model: Option<&ModelSpec>,
) -> miette::Result<Option<PathBuf>> {
    let program = crate::APP_NAME;
    if !can_ask() {
        let wanted = model.unwrap_or_else(|| model_manager.registry().default_model());
        return Err(if model.is_some() {
            miette::miette!(
                "Model {} is not downloaded. Download it with: {program} models download {}",
                wanted.id.name(),
                wanted.id
            )
        } else {
            miette::miette!(
                "No Whisper model found. Download one with: {program} models download {}",
                wanted.id
            )
        });
    }

    statusln!();
    if let Some(model) = model {
        statusln!(
            "{} Model {} is not downloaded!",
            "Notice:".warning(),
            model.id.name()
        );
    } else {
        statusln!("{} No Whisper model found!", "Notice:".warning());
        statusln!("To transcribe audio, you need to download a Whisper model first.");
        statusln!();
        statusln!(
            "The {} model is recommended for most users:",
            model_manager.registry().default_model().id.name()
        );
    }
    let model = model.unwrap_or_else(|| model_manager.registry().default_model());
    statusln!("  • {}", model.description.muted());
    statusln!("  • Size: {}", HumanBytes(model.file.size_bytes).quantity());
    statusln!(
        "  • Download time: {}",
        HumanDuration(model.estimated_download_time(64u32 * ByteSpeed::MIBPS)).quantity()
    );
    statusln!();

    status!(
        "Would you like to download the {} model now? [Y/n]: ",
        model.id
    );

    let mut input = String::new();
    io::stdin().read_line(&mut input).into_diagnostic()?;
    let input = input.trim().to_lowercase();

    let should_download = input.is_empty() || input == "y" || input == "yes";

    statusln!();
    if should_download {
        statusln!("{} Downloading {} model...", "Info:".info(), model.id);

        match model_manager.download_model(model, None).await {
            Ok(model_path) => {
                models::ensure_vad(model_manager).await;
                statusln!(
                    "{} Model downloaded successfully to: {}",
                    "Success:".success(),
                    model_path.display().path()
                );
                statusln!();
                Ok(Some(model_path))
            }
            Err(e) => {
                error!("Failed to download model: {}", e);
                Err(e.into())
            }
        }
    } else {
        statusln!("Model download cancelled. You can download a model later with:");
        statusln!(
            "  {}",
            format_args!("{program} models download <model>").command()
        );
        Ok(None)
    }
}
