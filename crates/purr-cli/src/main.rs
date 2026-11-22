//! Whisper UI CLI - Audio transcription command-line interface
#![deny(clippy::print_stdout, clippy::print_stderr, clippy::dbg_macro)]

mod bar;
mod batch;
mod build_info;
mod cli;
mod color;
mod download_prompt;
mod duration_arg;
mod engine_status;
mod engines;
mod engines_command;
mod error;
mod fatal;
mod global;
mod inputs;
mod interrupt;
mod json;
mod language_arg;
mod logging;
mod models;
mod outcome;
mod output;
mod output_plan;
mod profile_text;
mod progress;
mod render;
mod runtime;
mod scope;
mod stats;
mod sys_command;
mod sys_report;
mod template;
mod theme;
mod verbosity;
mod vt;
mod writer;

use crate::download_prompt::{Prompting, prompt_for_model_download};
use crate::error::{CliError, CliResult};
use crate::inputs::BatchFile;
use crate::language_arg::LanguageVerdict;
use crate::theme::Themed as _;
use cli::{ASCII_ART, Cli, Commands};
use purr_common::devices::DeviceOrder;
use purr_core::accel::engine_report::{EngineReport, reports};
use purr_core::accel::selector::EngineDevices;
use purr_core::engine::Engine;
use purr_core::{
    ModelManager,
    config::{ModelRef, RunConfig},
    model::vad_or_none,
};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use tracing::debug;

const APP_NAME: &str = env!("CARGO_BIN_NAME");

#[tokio::main]
async fn main() -> ExitCode {
    match main_impl().await {
        Ok(code) => code,
        Err(error) => {
            logging::record_fatal(&error);
            errorln!("{}", fatal::render(error));
            ExitCode::FAILURE
        }
    }
}

async fn main_impl() -> CliResult<ExitCode> {
    let cli = global::init(Cli::from_env());

    logging::init(cli)?;
    debug!("Command line arguments: {:?}", cli);
    let engines = engines::all();
    for engine in &engines {
        engine.install_logging();
    }

    if let Some(kind) = cli.throw_error {
        return Err(kind.raise());
    }

    if let Some(command) = &cli.command {
        handle_command(command).await?;
        return Ok(ExitCode::SUCCESS);
    }

    if cli.inputs.is_empty() && cli.batch_file.is_none() {
        statusln!("{ASCII_ART}\n");
        return Err(CliError::NoMediaInput);
    }

    let inputs = inputs::expand(&cli.inputs, cli.batch_file.as_ref())?;
    let plan = output_plan::plan(
        &inputs,
        cli.output,
        cli.output_file.as_deref(),
        cli.merge,
        cli.skip_existing,
    )?;
    let prompting = match cli.batch_file {
        Some(BatchFile::Stdin) => Prompting::Never,
        Some(BatchFile::Path(_)) | None => Prompting::Allowed,
    };

    batch::run(cli, &engines, inputs, plan, prompting).await
}

/// Handle subcommands
async fn handle_command(command: &Commands) -> CliResult<()> {
    match command {
        Commands::Models { command } => models::handle_model_command(command).await,
        Commands::Sys { json } => sys_command::handle(*json),
        Commands::Engines { json } => engines_command::handle(*json),
        Commands::Runtime { command } => runtime::handle_runtime_command(command),
    }
}

/// `--device` resolved against the devices of the engine that runs `model`.
fn device_order_for(
    cli: &Cli,
    engines: &[Arc<dyn Engine>],
    model: &ModelRef,
) -> CliResult<DeviceOrder> {
    let found = reports(engines, &[]);
    let nothing = [];
    let own = found
        .iter()
        .find(|report| report.engine == model.engine)
        .map_or(
            EngineDevices {
                engine: model.engine,
                devices: &nothing,
            },
            EngineReport::listed,
        );
    let others: Vec<EngineDevices<'_>> = found
        .iter()
        .filter(|report| report.engine != model.engine)
        .map(EngineReport::listed)
        .collect();
    Ok(cli.device_order_among(own, &others)?)
}

async fn setup_config(
    cli: &Cli,
    engines: &[Arc<dyn Engine>],
    prompting: Prompting,
) -> CliResult<(RunConfig, ModelRef)> {
    let model = setup_model(cli, prompting).await?;

    let mut config = cli.run_config();
    if let Some(spec) = ModelManager::new()?.listed(&model.path) {
        match language_arg::check(cli.language.as_ref(), &spec.id.to_string(), &spec.profile)? {
            LanguageVerdict::Applies => {}
            LanguageVerdict::HintIgnored => {
                statusln!(
                    "{} {} finds the language itself; --language is ignored.",
                    "Warning:".warning(),
                    spec.family.label()
                );
                config = config.with_opt_language(None);
            }
        }
    }
    if cli.device.is_some() {
        config = config.with_devices(device_order_for(cli, engines, &model)?);
    }

    if config.vad_model.is_none() {
        let installed = ModelManager::new()?.vad_model_path().await;
        config = config.with_vad_model(vad_or_none(installed));
    }

    Ok((config, model))
}

async fn setup_model(cli: &Cli, prompting: Prompting) -> CliResult<ModelRef> {
    let model_manager = ModelManager::new()?;
    let path: PathBuf;
    if let Some(ref model_string) = cli.model {
        let model_path = Path::new(model_string);
        if model_path.is_absolute() {
            if !model_path.exists() {
                return Err(CliError::ModelFileNotFound {
                    path: model_path.to_path_buf(),
                });
            }
            path = model_path.to_path_buf();
        } else {
            let model_path = std::env::current_dir()
                .map_err(|source| CliError::CurrentDir { source })?
                .join(model_path);
            if model_path.exists() {
                path = model_path;
            } else {
                let model = model_manager.model(model_string)?;
                if !engines::built().contains(&model.engine) {
                    return Err(CliError::EngineNotBuilt {
                        model: model.id.to_string(),
                        engine: model.engine.label().to_string(),
                    });
                }
                if model_manager.is_model_downloaded(model).await? {
                    path = model_manager.get_model_path(model);
                } else if let Some(model_path) =
                    prompt_for_model_download(&model_manager, Some(model), prompting).await?
                {
                    path = model_path;
                } else {
                    return Err(CliError::NoModelAvailable);
                }
            }
        }
    } else if let Some(model) = model_manager
        .find_first_available_model(&engines::built())
        .await?
    {
        path = model;
    } else if let Some(model_path) =
        prompt_for_model_download(&model_manager, None, prompting).await?
    {
        path = model_path;
    } else {
        return Err(CliError::NoModelAvailable);
    }
    Ok(model_manager.reference(path).await?)
}
