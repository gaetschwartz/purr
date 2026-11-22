use crate::cli::{Cli, OutputFormat};
use crate::download_prompt::Prompting;
use crate::error::{CliError, CliResult};
use crate::inputs::{InputForm, InputItem, InputSource, Inputs};
use crate::interrupt::Interrupt;
use crate::outcome::{Entry, Outcome, Summary};
use crate::output_plan::{OutputPlan, Sink};
use crate::render::{self, Layout, Rendering};
use crate::theme::Themed as _;
use crate::writer::PartFile;
use crate::{errorln, output, progress, result, setup_config, stats, statusln};
use futures::StreamExt as _;
use indicatif::{ProgressBar, ProgressFinish, ProgressStyle};
use purr_common::platform::ChunkKind;
use purr_core::accel::report::AccelReport;
use purr_core::error::Error as CoreError;
use purr_core::pipeline::events::{
    EndReason, StreamEvent, StreamingChunk, Transcription, TranscriptionStats,
};
use purr_core::pipeline::model::LoadedModel;
use purr_core::pipeline::result::{SyncTranscriptionResult, TranscriptionSegment};
use purr_core::pipeline::run::Input;
use purr_core::{
    AudioProcessor, SourceLength,
    config::{RunConfig, Windowing},
    engine::Engine,
};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{debug, info};

enum Destination {
    Stdout,
    Merged,
    File(PathBuf),
}

struct Job {
    item: InputItem,
    destination: Destination,
    length: SourceLength,
    skipped: bool,
}

struct Context<'a> {
    cli: &'a Cli,
    run: RunConfig,
    show_bar: bool,
    interrupt: Interrupt,
}

struct Finished {
    outcome: Outcome,
    audio_seconds: f64,
}

pub async fn run(
    cli: &Cli,
    engines: &[Arc<dyn Engine>],
    inputs: Inputs,
    plan: OutputPlan,
    prompting: Prompting,
) -> CliResult<ExitCode> {
    let Inputs { form, items } = inputs;

    if let (InputForm::Single, Some(item)) = (form, items.first()) {
        ensure_exists(item)?;
        announce(item, None);
    }

    let (config, model) = setup_config(cli, engines, prompting).await?;
    if let Some(lang) = &config.language {
        info!("Language: {lang}");
        statusln!("{} {lang}", "Language:".info());
    }

    let OutputPlan { sink, skipped } = plan;
    let sources: Vec<Option<PathBuf>> = items
        .iter()
        .enumerate()
        .map(|(position, item)| match &item.source {
            InputSource::File(path) if !skipped.contains(&position) => Some(path.clone()),
            InputSource::File(_) | InputSource::Url(_) => None,
        })
        .collect();
    let probing = tokio::task::spawn_blocking(move || probe_lengths(&sources));
    let mut loaded = LoadedModel::load(&model, &config, engines).await?;
    let lengths = if probing.is_finished() {
        probing.await.map_err(|source| CoreError::TaskJoin {
            task: "input probing",
            source,
        })?
    } else {
        debug!("Probing was still running when the model was ready");
        vec![SourceLength::Unknown; items.len()]
    };

    let destinations: Vec<Destination> = match &sink {
        Sink::PerInput(paths) => paths.iter().cloned().map(Destination::File).collect(),
        Sink::Stdout { .. } => items.iter().map(|_| Destination::Stdout).collect(),
        Sink::Merged(_) => items.iter().map(|_| Destination::Merged).collect(),
    };
    let jobs: Vec<Job> = items
        .into_iter()
        .enumerate()
        .zip(destinations)
        .zip(lengths)
        .map(|(((position, item), destination), length)| Job {
            skipped: skipped.contains(&position),
            item,
            destination,
            length,
        })
        .collect();

    let interrupt = cli
        .interrupt_after
        .map_or_else(Interrupt::install, |millis| {
            Interrupt::after(Duration::from_millis(millis))
        });
    let context = Context {
        cli,
        run: if cli.no_stream {
            debug!("Transcribing without streaming");
            config.with_windowing(Windowing::Whole)
        } else {
            debug!("Streaming transcription");
            config
        },
        show_bar: cli.progress && output::is_interactive(),
        interrupt,
    };
    let mut destination = Output::new(cli, context.show_bar, sink);

    let started = Instant::now();
    let total = jobs.len();
    let mut entries = Vec::with_capacity(total);
    let mut audio_seconds = 0.0;
    let mut halted = false;
    for job in jobs {
        let display = job.item.display.clone();
        let outcome = if halted || context.interrupt.is_requested() {
            halted = true;
            Outcome::NotStarted
        } else if job.skipped {
            statusln!("{} {display} (output exists)", "Skipping:".warning());
            Outcome::Skipped
        } else {
            if matches!(form, InputForm::Batch) {
                announce(&job.item, Some(total));
            }
            let finished = run_input(&context, &mut loaded, &mut destination, job).await;
            audio_seconds += finished.audio_seconds;
            finished.outcome
        };
        if cli.fail_fast && matches!(outcome, Outcome::Failed { .. }) {
            halted = true;
        }
        entries.push(Entry { display, outcome });
    }

    let summary = Summary::new(entries, audio_seconds, started.elapsed());
    let merged = destination.finish(!summary.cut_short());
    let summary = match &merged {
        Ok(Some(Merged::Unfinished(part))) => summary.with_merged_part(part.clone()),
        Ok(Some(Merged::Completed(_)) | None) | Err(_) => summary,
    };
    let exit = match form {
        InputForm::Single => {
            let summary = summary.into_result()?;
            if summary.cut_short() {
                errorln!("{}", summary.render());
            }
            summary.exit_code()
        }
        InputForm::Batch => {
            errorln!("{}", summary.render());
            summary.exit_code()
        }
    };
    if let Some(Merged::Completed(path)) = merged?
        && cli.verbosity.is_verbose()
    {
        statusln!(
            "{} Output written to: {}",
            "Success:".success(),
            path.display().path()
        );
    }
    Ok(exit)
}

fn probe_lengths(sources: &[Option<PathBuf>]) -> Vec<SourceLength> {
    let started = Instant::now();
    let lengths: Vec<SourceLength> = sources
        .iter()
        .map(|source| source.as_deref().map_or(SourceLength::Unknown, probe_file))
        .collect();
    debug!("Probed {} inputs in {:?}", lengths.len(), started.elapsed());
    lengths
}

fn probe_file(path: &Path) -> SourceLength {
    match AudioProcessor::probe_file_length(path) {
        Ok(length) => length,
        Err(error) => {
            debug!("Could not probe {}: {error}", path.display());
            SourceLength::Unknown
        }
    }
}

fn ensure_exists(item: &InputItem) -> CliResult<()> {
    match &item.source {
        InputSource::File(path) if !path.exists() => {
            Err(CliError::AudioFileNotFound { path: path.clone() })
        }
        InputSource::File(_) | InputSource::Url(_) => Ok(()),
    }
}

fn announce(item: &InputItem, total: Option<usize>) {
    let kind = match item.source {
        InputSource::File(_) => "file",
        InputSource::Url(_) => "URL",
    };
    info!("Input: {} ({kind})", item.display);
    let position = total.map_or_else(String::new, |total| format!("[{}/{total}] ", item.index));
    statusln!(
        "{position}{} {} ({kind})",
        "Transcribing:".info(),
        item.display.path()
    );
}

async fn run_input(
    context: &Context<'_>,
    loaded: &mut LoadedModel,
    output: &mut Output,
    job: Job,
) -> Finished {
    debug!("Input {}: length {:?}", job.item.index, job.length);
    output.begin_input(job.destination);
    match attempt(context, loaded, output, &job.item).await {
        Ok(Course::Ended {
            ended,
            audio_seconds,
        }) => Finished {
            outcome: Outcome::Done { ended },
            audio_seconds,
        },
        Ok(Course::Interrupted { audio_seconds }) => Finished {
            outcome: Outcome::Interrupted {
                part: output.unfinished(),
            },
            audio_seconds,
        },
        Ok(Course::Unopened) => Finished {
            outcome: Outcome::NotStarted,
            audio_seconds: 0.0,
        },
        Err(error) => Finished {
            outcome: Outcome::Failed {
                part: output.unfinished(),
                error,
            },
            audio_seconds: 0.0,
        },
    }
}

enum Course {
    Ended {
        ended: EndReason,
        audio_seconds: f64,
    },
    Interrupted {
        audio_seconds: f64,
    },
    Unopened,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Halt {
    Stop,
    Cancel,
}

impl Halt {
    const fn for_run(length: SourceLength, no_stream: bool) -> Self {
        match (length, no_stream) {
            (SourceLength::Unknown, _)
            | (SourceLength::Exact(_) | SourceLength::Estimated(_), true) => Self::Stop,
            (SourceLength::Exact(_) | SourceLength::Estimated(_), false) => Self::Cancel,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Verdict {
    Ended(EndReason),
    Interrupted,
}

fn verdict(
    asked: Option<Halt>,
    ended: Option<EndReason>,
    length: SourceLength,
) -> CliResult<Verdict> {
    match (asked, ended) {
        (Some(Halt::Cancel), _) => Ok(Verdict::Interrupted),
        (Some(Halt::Stop), Some(EndReason::Stopped)) => Ok(match length {
            SourceLength::Unknown => Verdict::Ended(EndReason::Stopped),
            SourceLength::Exact(_) | SourceLength::Estimated(_) => Verdict::Interrupted,
        }),
        (
            Some(Halt::Stop) | None,
            Some(reason @ (EndReason::SourceEnded | EndReason::DurationReached)),
        ) => Ok(Verdict::Ended(reason)),
        (None, Some(EndReason::Stopped)) | (Some(Halt::Stop) | None, None) => {
            Err(CliError::EndedUnexplained)
        }
    }
}

struct Pumped {
    audio_seconds: f64,
    verdict: Verdict,
}

async fn attempt(
    context: &Context<'_>,
    loaded: &mut LoadedModel,
    output: &mut Output,
    item: &InputItem,
) -> CliResult<Course> {
    ensure_exists(item)?;
    let input = match &item.source {
        InputSource::File(path) => Input::File(path.clone()),
        InputSource::Url(url) => Input::Url {
            url: url.clone(),
            config: context.cli.input_config(),
        },
    };
    let mut transcription = tokio::select! {
        opened = loaded.transcribe(input, &context.run) => opened?,
        () = context.interrupt.requested() => return Ok(Course::Unopened),
    };
    let length = transcription.length();
    let no_stream = context.cli.no_stream;
    if no_stream && matches!(length, SourceLength::Unknown) {
        statusln!(
            "{} {} has no known length; nothing is written until it ends",
            "Warning:".warning(),
            item.display.path()
        );
    }
    let halt = Halt::for_run(length, no_stream);
    let Pumped {
        audio_seconds,
        verdict,
    } = if no_stream {
        whole(context, &mut transcription, output, item, halt).await?
    } else {
        streamed(context, &mut transcription, output, item, halt).await?
    };
    match verdict {
        Verdict::Interrupted => Ok(Course::Interrupted { audio_seconds }),
        Verdict::Ended(ended) => {
            if let Some(path) = output.complete_input()?
                && context.cli.verbosity.is_verbose()
            {
                let what = if no_stream {
                    "Output"
                } else {
                    "Streaming output"
                };
                statusln!(
                    "{} {what} written to: {}",
                    "Success:".success(),
                    path.display().path()
                );
            }
            Ok(Course::Ended {
                ended,
                audio_seconds,
            })
        }
    }
}

async fn pump(
    transcription: &mut Transcription,
    interrupt: &Interrupt,
    halt: Halt,
    mut on_event: impl FnMut(StreamEvent) -> CliResult<()>,
) -> CliResult<Option<Halt>> {
    let mut asked = false;
    let mut complete = false;
    loop {
        let event = tokio::select! {
            event = transcription.next() => event,
            () = interrupt.requested(), if !asked && !complete => {
                asked = true;
                match halt {
                    Halt::Stop => transcription.stop(),
                    Halt::Cancel => transcription.cancel(),
                }
                continue;
            }
        };
        let Some(event) = event else {
            return Ok(asked.then_some(halt));
        };
        let event = event?;
        if let StreamEvent::Chunk(chunk) = &event
            && chunk.is_final
        {
            complete = true;
        }
        on_event(event)?;
    }
}

async fn streamed(
    context: &Context<'_>,
    transcription: &mut Transcription,
    output: &mut Output,
    item: &InputItem,
    halt: Halt,
) -> CliResult<Pumped> {
    let bar = if context.show_bar {
        Some(progress_line_bar()?)
    } else {
        None
    };
    let mut chunks = 0usize;
    let mut received = 0.0;

    let asked = pump(transcription, &context.interrupt, halt, |event| {
        let chunk = match event {
            StreamEvent::Progress(progress) => {
                received = progress.received;
                if let Some(bar) = &bar {
                    bar.set_message(progress::line(&progress).muted().to_string());
                }
                return Ok(());
            }
            StreamEvent::Chunk(chunk) => chunk,
        };
        match chunk.kind {
            ChunkKind::Committed => chunks += 1,
            ChunkKind::Tentative => return Ok(()),
        }
        output.chunk(item, &chunk)?;
        if let Some(ref chunk_stats) = chunk.final_stats
            && context.cli.verbosity.is_verbose()
        {
            stats::show("Streaming Transcription Statistics:", chunk_stats, None);
        }
        Ok(())
    })
    .await?;
    drop(bar);

    output.stream_end()?;
    debug!("Processed {chunks} chunks");
    let verdict = verdict(asked, transcription.end_reason(), transcription.length())?;
    if chunks == 0 && matches!(verdict, Verdict::Ended(_)) {
        return Err(CliError::NothingTranscribed);
    }
    Ok(Pumped {
        audio_seconds: received,
        verdict,
    })
}

async fn whole(
    context: &Context<'_>,
    transcription: &mut Transcription,
    output: &mut Output,
    item: &InputItem,
    halt: Halt,
) -> CliResult<Pumped> {
    let mut collected = Collected::default();
    let asked = pump(transcription, &context.interrupt, halt, |event| {
        collected.absorb(event);
        Ok(())
    })
    .await?;
    let verdict = verdict(asked, transcription.end_reason(), transcription.length())?;
    let received = collected.received;
    let result = collected.into_result(transcription.accel().clone());
    match verdict {
        Verdict::Ended(_) => output.document(item, &result)?,
        Verdict::Interrupted if !result.segments.is_empty() => output.document(item, &result)?,
        Verdict::Interrupted => {}
    }
    if context.cli.verbosity.is_verbose() {
        stats::show(
            "Transcription Statistics:",
            &result.stats,
            Some(result.accel.selected.pretty_name()),
        );
    }
    Ok(Pumped {
        audio_seconds: if result.audio_duration > 0.0 {
            f64::from(result.audio_duration)
        } else {
            received
        },
        verdict,
    })
}

#[derive(Default)]
struct Collected {
    segments: Vec<TranscriptionSegment>,
    final_stats: Option<TranscriptionStats>,
    received: f64,
}

impl Collected {
    fn absorb(&mut self, event: StreamEvent) {
        let chunk = match event {
            StreamEvent::Progress(progress) => {
                self.received = progress.received;
                return;
            }
            StreamEvent::Chunk(chunk) => chunk,
        };
        match chunk.kind {
            ChunkKind::Committed => {}
            ChunkKind::Tentative => return,
        }
        if chunk.final_stats.is_some() {
            self.final_stats = chunk.final_stats;
        }
        if !chunk.text.is_empty() {
            self.segments.push(TranscriptionSegment {
                text: chunk.text,
                start: chunk.start,
                end: chunk.end,
                words: chunk.words,
            });
        }
    }

    fn into_result(self, accel: AccelReport) -> SyncTranscriptionResult {
        let text = self
            .segments
            .iter()
            .map(|segment| segment.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        let (processing_time, audio_duration) = self.final_stats.map_or((0.0, 0.0), |stats| {
            (stats.processing_time, stats.audio_duration)
        });
        let stats = TranscriptionStats {
            processing_time,
            audio_duration,
            segment_count: self.segments.len(),
            word_count: text.split_whitespace().count(),
        };
        SyncTranscriptionResult {
            text,
            segments: self.segments,
            processing_time,
            audio_duration,
            stats,
            accel,
        }
    }
}

fn progress_line_bar() -> CliResult<ProgressBar> {
    let style = ProgressStyle::with_template("{msg}")
        .map_err(|source| CliError::ProgressTemplate { source })?;
    Ok(output::add_bar(
        ProgressBar::new_spinner()
            .with_style(style)
            .with_finish(ProgressFinish::AndClear),
    ))
}

struct FileSlot {
    path: PathBuf,
    part: Option<PartFile>,
}

impl FileSlot {
    const fn new(path: PathBuf) -> Self {
        Self { path, part: None }
    }

    fn append(&mut self, text: &str) -> CliResult<()> {
        let part = match self.part {
            Some(ref mut part) => part,
            None => self.part.insert(PartFile::create(&self.path)?),
        };
        part.append(text)?;
        Ok(())
    }

    fn part_path(&self) -> Option<&Path> {
        self.part.as_ref().map(PartFile::part_path)
    }

    fn complete(self) -> CliResult<PathBuf> {
        let part = match self.part {
            Some(part) => part,
            None => PartFile::create(&self.path)?,
        };
        Ok(part.complete()?)
    }
}

#[derive(Debug, PartialEq, Eq)]
enum Merged {
    Completed(PathBuf),
    Unfinished(PathBuf),
}

enum Target {
    Stdout,
    PerInput,
    Merged,
}

struct Output {
    target: Target,
    slot: Option<FileSlot>,
    rendering: Rendering,
    layout: Layout,
    headers: bool,
    tag_input: bool,
    document_end: &'static str,
    begun: bool,
    printed: bool,
}

impl Output {
    fn new(cli: &Cli, show_bar: bool, sink: Sink) -> Self {
        let format = cli.output;
        let text_like = match format {
            OutputFormat::Text | OutputFormat::Txt => true,
            OutputFormat::Json | OutputFormat::Srt => false,
        };
        let (target, slot, shared, layout) = match sink {
            Sink::Stdout { batch } => {
                let layout = if batch && format == OutputFormat::Json {
                    Layout::File
                } else {
                    Layout::Stdout {
                        one_chunk_per_line: show_bar,
                    }
                };
                (Target::Stdout, None, batch, layout)
            }
            Sink::PerInput(_) => (Target::PerInput, None, false, Layout::File),
            Sink::Merged(path) => (
                Target::Merged,
                Some(FileSlot::new(path)),
                true,
                Layout::File,
            ),
        };
        Self {
            target,
            slot,
            rendering: Rendering {
                format,
                timestamps: cli.timestamps,
                word_timestamps: cli.word_timestamps,
            },
            layout,
            headers: shared && text_like,
            tag_input: shared,
            document_end: if shared { "\n" } else { "" },
            begun: false,
            printed: false,
        }
    }

    fn begin_input(&mut self, destination: Destination) {
        self.begun = false;
        match destination {
            Destination::File(path) => self.slot = Some(FileSlot::new(path)),
            Destination::Stdout | Destination::Merged => {}
        }
    }

    fn unfinished(&self) -> Option<PathBuf> {
        match self.target {
            Target::PerInput => self
                .slot
                .as_ref()
                .and_then(|slot| slot.part.as_ref())
                .map(|part| part.part_path().to_path_buf()),
            Target::Stdout | Target::Merged => None,
        }
    }

    fn tag<'a>(&self, item: &'a InputItem) -> Option<&'a str> {
        self.tag_input.then_some(item.display.as_str())
    }

    fn write(&mut self, text: &str) -> CliResult<()> {
        if text.is_empty() {
            return Ok(());
        }
        match &mut self.slot {
            Some(slot) => slot.append(text),
            None => Ok(result!("{text}")?),
        }
    }

    fn begin(&mut self, item: &InputItem) -> CliResult<()> {
        if self.begun {
            return Ok(());
        }
        self.begun = true;
        if self.headers {
            if self.printed {
                self.write("\n")?;
            }
            self.write(&render::header(&item.display))?;
        }
        self.printed = true;
        Ok(())
    }

    fn chunk(&mut self, item: &InputItem, chunk: &StreamingChunk) -> CliResult<()> {
        let Some(piece) = render::streaming_piece(&self.rendering, chunk, self.tag(item))? else {
            return Ok(());
        };
        self.begin(item)?;
        self.write(&piece)?;
        self.write(render::terminator(&self.rendering, chunk, self.layout))
    }

    fn stream_end(&mut self) -> CliResult<()> {
        self.write(render::stream_end(self.layout))
    }

    fn document(&mut self, item: &InputItem, result: &SyncTranscriptionResult) -> CliResult<()> {
        let piece = render::document_piece(&self.rendering, result, self.tag(item))?;
        self.begin(item)?;
        self.write(&piece)?;
        self.write(self.document_end)
    }

    fn complete_input(&mut self) -> CliResult<Option<PathBuf>> {
        match self.target {
            Target::PerInput => self.slot.take().map(FileSlot::complete).transpose(),
            Target::Stdout | Target::Merged => Ok(None),
        }
    }

    fn finish(&mut self, complete: bool) -> CliResult<Option<Merged>> {
        match self.target {
            Target::Merged => {
                let Some(slot) = self.slot.take() else {
                    return Ok(None);
                };
                let Some(part) = slot.part_path().map(Path::to_path_buf) else {
                    return Ok(None);
                };
                if complete {
                    Ok(Some(Merged::Completed(slot.complete()?)))
                } else {
                    Ok(Some(Merged::Unfinished(part)))
                }
            }
            Target::Stdout | Target::PerInput => Ok(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser as _;
    use pretty_assertions::assert_eq;
    use tempfile::TempDir;

    fn item(index: usize, display: &str) -> InputItem {
        InputItem {
            index,
            source: InputSource::File(PathBuf::from(display)),
            display: display.to_owned(),
        }
    }

    fn chunk(text: &str, index: usize) -> StreamingChunk {
        StreamingChunk {
            text: text.to_owned(),
            start: 0.0,
            end: 1.0,
            is_final: false,
            chunk_index: index,
            final_stats: None,
            language: None,
            kind: ChunkKind::Committed,
            words: Vec::new(),
        }
    }

    #[test]
    fn merged_text_has_a_header_per_input_and_a_blank_line_between() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("all.txt");
        let cli = Cli::try_parse_from(["purr", "a.wav"]).unwrap();
        let mut output = Output::new(&cli, false, Sink::Merged(path.clone()));
        for (index, name, text) in [(1, "a.wav", "Hello"), (2, "b.wav", "world")] {
            output.begin_input(Destination::Merged);
            output.chunk(&item(index, name), &chunk(text, 0)).unwrap();
            output.stream_end().unwrap();
            assert_eq!(output.complete_input().unwrap(), None);
        }
        assert_eq!(
            output.finish(true).unwrap(),
            Some(Merged::Completed(path.clone()))
        );
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "==> a.wav <==\nHello\n\n==> b.wav <==\nworld\n"
        );
        assert!(!dir.path().join("all.txt.part").exists());
    }

    #[test]
    fn a_merged_output_stays_a_part_file_when_the_run_is_cut_short() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("all.txt");
        let cli = Cli::try_parse_from(["purr", "a.wav"]).unwrap();
        let mut output = Output::new(&cli, false, Sink::Merged(path.clone()));
        output.begin_input(Destination::Merged);
        output.chunk(&item(1, "a.wav"), &chunk("Hello", 0)).unwrap();
        output.stream_end().unwrap();
        let part = dir.path().join("all.txt.part");
        assert_eq!(
            output.finish(false).unwrap(),
            Some(Merged::Unfinished(part.clone()))
        );
        assert!(!path.exists());
        assert_eq!(
            std::fs::read_to_string(&part).unwrap(),
            "==> a.wav <==\nHello\n"
        );
    }

    #[test]
    fn a_merged_output_that_wrote_nothing_has_nothing_to_finish() {
        let dir = TempDir::new().unwrap();
        let cli = Cli::try_parse_from(["purr", "a.wav"]).unwrap();
        let mut output = Output::new(&cli, false, Sink::Merged(dir.path().join("all.txt")));
        assert_eq!(output.finish(false).unwrap(), None);
        assert_eq!(output.finish(true).unwrap(), None);
    }

    #[test]
    fn merged_json_lines_name_their_input_and_have_no_headers() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("all.jsonl");
        let cli = Cli::try_parse_from(["purr", "a.wav", "-o", "json"]).unwrap();
        let mut output = Output::new(&cli, false, Sink::Merged(path.clone()));
        for (index, name) in [(1, "a.wav"), (2, "b.wav")] {
            output.begin_input(Destination::Merged);
            output.chunk(&item(index, name), &chunk("Hi", 0)).unwrap();
            output.stream_end().unwrap();
        }
        output.finish(true).unwrap();
        let written = std::fs::read_to_string(&path).unwrap();
        let inputs: Vec<String> = written
            .lines()
            .map(|line| {
                let value: serde_json::Value = serde_json::from_str(line).unwrap();
                value["input"].as_str().unwrap().to_owned()
            })
            .collect();
        assert_eq!(inputs, ["a.wav", "b.wav"]);
    }

    #[test]
    fn a_completed_input_replaces_its_part_file_with_the_final_file() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("a.txt");
        let cli = Cli::try_parse_from(["purr", "a.wav", "-f", "x_{stem}.txt"]).unwrap();
        let mut output = Output::new(&cli, false, Sink::PerInput(vec![path.clone()]));
        output.begin_input(Destination::File(path.clone()));
        output.chunk(&item(1, "a.wav"), &chunk("Hello", 0)).unwrap();
        assert_eq!(
            std::fs::read_to_string(dir.path().join("a.txt.part")).unwrap(),
            "Hello\n"
        );
        assert_eq!(output.complete_input().unwrap(), Some(path.clone()));
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "Hello\n");
        assert!(!dir.path().join("a.txt.part").exists());
    }

    #[test]
    fn an_abandoned_input_leaves_its_part_file_and_names_it() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("a.txt");
        let cli = Cli::try_parse_from(["purr", "a.wav", "-f", "x_{stem}.txt"]).unwrap();
        let mut output = Output::new(&cli, false, Sink::PerInput(vec![path.clone()]));
        output.begin_input(Destination::File(path.clone()));
        output.chunk(&item(1, "a.wav"), &chunk("Hello", 0)).unwrap();
        assert_eq!(output.unfinished(), Some(dir.path().join("a.txt.part")));
        assert_eq!(
            std::fs::read_to_string(dir.path().join("a.txt.part")).unwrap(),
            "Hello\n"
        );
        assert!(!path.exists());
    }

    #[test]
    fn an_input_that_wrote_nothing_has_no_part_file_to_leave() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("a.txt");
        let cli = Cli::try_parse_from(["purr", "a.wav", "-f", "x_{stem}.txt"]).unwrap();
        let mut output = Output::new(&cli, false, Sink::PerInput(vec![path.clone()]));
        output.begin_input(Destination::File(path));
        assert_eq!(output.unfinished(), None);
    }

    const KNOWN: SourceLength = SourceLength::Exact(Duration::from_secs(60));

    #[test]
    fn a_stream_of_unknown_length_is_stopped_and_a_known_one_is_cancelled() {
        assert_eq!(Halt::for_run(SourceLength::Unknown, false), Halt::Stop);
        assert_eq!(Halt::for_run(KNOWN, false), Halt::Cancel);
        assert_eq!(
            Halt::for_run(SourceLength::Estimated(Duration::from_secs(5)), false),
            Halt::Cancel
        );
    }

    #[test]
    fn a_stop_before_the_end_of_a_known_length_is_interrupted_and_of_an_unknown_one_is_done() {
        assert_eq!(
            verdict(
                Some(Halt::Stop),
                Some(EndReason::Stopped),
                SourceLength::Unknown
            )
            .unwrap(),
            Verdict::Ended(EndReason::Stopped)
        );
        assert_eq!(
            verdict(Some(Halt::Stop), Some(EndReason::Stopped), KNOWN).unwrap(),
            Verdict::Interrupted
        );
        assert_eq!(
            verdict(Some(Halt::Cancel), None, KNOWN).unwrap(),
            Verdict::Interrupted
        );
    }

    #[test]
    fn a_source_that_ended_first_is_done_whatever_was_asked() {
        for asked in [None, Some(Halt::Stop)] {
            for reason in [EndReason::SourceEnded, EndReason::DurationReached] {
                assert_eq!(
                    verdict(asked, Some(reason), KNOWN).unwrap(),
                    Verdict::Ended(reason)
                );
            }
        }
    }

    #[test]
    fn an_end_nobody_explains_is_an_error() {
        assert!(matches!(
            verdict(None, None, KNOWN),
            Err(CliError::EndedUnexplained)
        ));
        assert!(matches!(
            verdict(Some(Halt::Stop), None, KNOWN),
            Err(CliError::EndedUnexplained)
        ));
    }

    #[test]
    fn a_missing_local_file_is_that_inputs_failure() {
        assert!(matches!(
            ensure_exists(&item(1, "definitely-not-here.wav")),
            Err(CliError::AudioFileNotFound { .. })
        ));
    }
}
