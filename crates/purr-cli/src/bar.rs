//! The progress rows: a rule that follows the audio, a total rule in a batch, and a status row.

use crate::error::{CliError, CliResult};
use crate::inputs::{InputForm, InputItem};
use crate::outcome::Outcome;
use crate::progress::{BatchStatus, Status, StatusLength, status_text};
use crate::theme::{ProgressPart, progress_style};
use crate::{color::Stream, global, output};
use console::Term;
use indicatif::style::ProgressTracker;
use indicatif::{ProgressBar, ProgressState, ProgressStyle};
use parking_lot::Mutex;
use purr_core::SourceLength;
use purr_core::pipeline::events::StreamProgress;
use std::fmt;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::watch;
use tracing::debug;

const DONE: char = '━';
const HALF: char = '╸';
const REST: char = '─';
const MIN_RULE: usize = 10;
const MAX_RULE: usize = 100;
const SWEEP: Duration = Duration::from_millis(1500);
const SEGMENT: usize = 6;
const STEADY_TICK: Duration = Duration::from_millis(100);
const SPINNER: [&str; 11] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏", "✔"];
const STARTING: &str = "Starting";

/// The full cells of a rule filled to `fraction`, and whether a half cell follows them.
#[must_use]
pub fn rule_fill(fraction: f64, width: usize) -> (usize, bool) {
    let cells = fraction.clamp(0.0, 1.0) * width as f64;
    let full = (cells.floor() as usize).min(width);
    let half = full < width && cells - full as f64 >= 0.5;
    (full, half)
}

/// Where a segment of `segment` cells starts on a rule of `width` cells, sweeping to the far end and
/// back again, every `SWEEP` in each direction.
#[must_use]
pub fn bounce_position(elapsed: Duration, width: usize, segment: usize) -> usize {
    let span = width.saturating_sub(segment);
    let phase = (elapsed.as_secs_f64() / SWEEP.as_secs_f64()) % 2.0;
    let along = if phase <= 1.0 { phase } else { 2.0 - phase };
    ((along * span as f64).round() as usize).min(span)
}

/// How far a batch has come, where each input counts for its weight.
#[must_use]
pub fn total_fraction(
    done_weight: f64,
    current_weight: f64,
    current_fraction: f64,
    total_weight: f64,
) -> f64 {
    if total_weight > 0.0 {
        (current_weight.mul_add(current_fraction, done_weight) / total_weight).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn filled_rule(fraction: f64, width: usize) -> String {
    let (full, half) = rule_fill(fraction, width);
    let rest = width - full - usize::from(half);
    let mut rule = String::with_capacity(width * DONE.len_utf8());
    rule.extend(std::iter::repeat_n(DONE, full));
    if half {
        rule.push(HALF);
    }
    rule.extend(std::iter::repeat_n(REST, rest));
    rule
}

fn bouncing_rule(elapsed: Duration, width: usize) -> String {
    let segment = SEGMENT.min(width / 2);
    let start = bounce_position(elapsed, width, segment);
    let mut rule = String::with_capacity(width * DONE.len_utf8());
    rule.extend(std::iter::repeat_n(REST, start));
    rule.extend(std::iter::repeat_n(DONE, segment));
    rule.extend(std::iter::repeat_n(REST, width - start - segment));
    rule
}

type Elapsed = Arc<dyn Fn() -> Duration + Send + Sync>;
type Columns = Arc<dyn Fn() -> usize + Send + Sync>;
type Place = Box<dyn Fn(ProgressBar) -> ProgressBar>;

/// Where the rows are drawn and what they read the time and the width from.
pub struct Surface {
    place: Place,
    columns: Columns,
    elapsed: Elapsed,
    steady: Option<Duration>,
    colors: bool,
}

impl Surface {
    /// The terminal, through the output module so that text lines never land inside a row.
    pub fn terminal() -> Self {
        let started = Instant::now();
        Self {
            place: Box::new(output::add_bar),
            columns: Arc::new(|| usize::from(Term::stdout().size().1)),
            elapsed: Arc::new(move || started.elapsed()),
            steady: Some(STEADY_TICK),
            colors: global::colors().enabled(Stream::Stdout),
        }
    }
}

pub struct Plan<'a> {
    pub items: &'a [InputItem],
    pub skipped: &'a [usize],
    pub form: InputForm,
    pub probes: watch::Receiver<Vec<SourceLength>>,
    pub duration: Option<Duration>,
}

struct Item {
    name: String,
    length: SourceLength,
    settled: bool,
    skipped: bool,
    finished: bool,
}

#[derive(Default, Clone, Copy)]
struct Live {
    transcribed: f64,
    received: f64,
    source_ended: bool,
}

#[derive(Default)]
struct Counts {
    done: usize,
    failed: usize,
    skipped: usize,
}

struct Totals {
    done_weight: f64,
    current_weight: f64,
    current_fraction: f64,
    total_weight: f64,
    weighted: bool,
}

struct Model {
    form: InputForm,
    items: Vec<Item>,
    current: Option<usize>,
    live: Live,
    counts: Counts,
    audio_done: f64,
    duration: Option<f64>,
    probes: Option<watch::Receiver<Vec<SourceLength>>>,
}

impl Model {
    fn new(plan: Plan<'_>) -> Self {
        Self {
            form: plan.form,
            items: plan
                .items
                .iter()
                .enumerate()
                .map(|(position, item)| Item {
                    name: item.display.clone(),
                    length: SourceLength::Unknown,
                    settled: false,
                    skipped: plan.skipped.contains(&position),
                    finished: false,
                })
                .collect(),
            current: None,
            live: Live::default(),
            counts: Counts::default(),
            audio_done: 0.0,
            duration: plan.duration.map(|duration| duration.as_secs_f64()),
            probes: Some(plan.probes),
        }
    }

    fn sync_probes(&mut self) {
        let Some(probes) = &self.probes else {
            return;
        };
        let probed = probes.borrow();
        for (item, length) in self.items.iter_mut().zip(probed.iter()) {
            if !item.settled {
                item.length = *length;
            }
        }
    }

    fn is_running(&self, position: usize) -> bool {
        self.current == Some(position)
            && self.items.get(position).is_some_and(|item| !item.finished)
    }

    fn length_of(&self, position: usize) -> Option<f64> {
        if self.current == Some(position) && self.live.source_ended {
            return Some(self.live.received);
        }
        match self.items.get(position)?.length {
            SourceLength::Exact(length) | SourceLength::Estimated(length) => {
                Some(length.as_secs_f64())
            }
            SourceLength::Unknown => self.duration,
        }
    }

    fn shown_length(&self, position: usize) -> Option<StatusLength> {
        if self.current == Some(position) && self.live.source_ended {
            return Some(StatusLength {
                seconds: self.live.received,
                estimated: false,
            });
        }
        match self.items.get(position)?.length {
            SourceLength::Exact(length) => Some(StatusLength {
                seconds: length.as_secs_f64(),
                estimated: false,
            }),
            SourceLength::Estimated(length) => Some(StatusLength {
                seconds: length.as_secs_f64(),
                estimated: true,
            }),
            SourceLength::Unknown => None,
        }
    }

    fn current_fraction(&self) -> Option<f64> {
        let length = self
            .length_of(self.current?)
            .filter(|length| *length > 0.0)?;
        Some((self.live.transcribed / length).clamp(0.0, 1.0))
    }

    fn totals(&self) -> Totals {
        let counted: Vec<(usize, &Item)> = self
            .items
            .iter()
            .enumerate()
            .filter(|(_, item)| !item.skipped)
            .collect();
        let lengths: Option<Vec<f64>> = counted
            .iter()
            .map(|(position, _)| self.length_of(*position))
            .collect();
        let weighted = lengths.is_some();
        let weights = lengths.unwrap_or_else(|| vec![1.0; counted.len()]);
        let current_fraction = self.current_fraction().unwrap_or(0.0);
        let mut totals = Totals {
            done_weight: 0.0,
            current_weight: 0.0,
            current_fraction,
            total_weight: 0.0,
            weighted,
        };
        for ((position, item), weight) in counted.into_iter().zip(weights) {
            totals.total_weight += weight;
            if item.finished {
                totals.done_weight += weight;
            } else if self.current == Some(position) {
                totals.current_weight = weight;
            }
        }
        totals
    }

    fn speed(&self, elapsed: f64) -> Option<f64> {
        let running = self
            .current
            .is_some_and(|position| self.is_running(position));
        let processed = self.audio_done + if running { self.live.transcribed } else { 0.0 };
        (processed > 0.0 && elapsed > 0.0).then(|| processed / elapsed)
    }

    fn total_rule(&self, width: usize) -> String {
        let totals = self.totals();
        filled_rule(
            total_fraction(
                totals.done_weight,
                totals.current_weight,
                totals.current_fraction,
                totals.total_weight,
            ),
            width,
        )
    }

    fn current_rule(&self, width: usize, elapsed: Duration) -> String {
        match (self.current, self.current_fraction()) {
            (None, _) => filled_rule(0.0, width),
            (Some(_), Some(fraction)) => filled_rule(fraction, width),
            (Some(_), None) => bouncing_rule(elapsed, width),
        }
    }

    fn status_row(&self, elapsed: Duration, width: usize) -> String {
        let Some((position, item)) = self
            .current
            .and_then(|position| Some((position, self.items.get(position)?)))
        else {
            return STARTING.to_owned();
        };
        let seconds = elapsed.as_secs_f64();
        let speed = self.speed(seconds);
        let totals = self.totals();
        let eta = match (totals.weighted, speed) {
            (true, Some(speed)) => {
                let remaining = totals.current_weight.mul_add(
                    -totals.current_fraction,
                    totals.total_weight - totals.done_weight,
                );
                (remaining > 0.0).then(|| remaining / speed)
            }
            (true, None) | (false, _) => None,
        };
        let batch = match self.form {
            InputForm::Single => None,
            InputForm::Batch => Some(BatchStatus {
                index: position + 1,
                total: self.items.len(),
                done: self.counts.done,
                failed: self.counts.failed,
                skipped: self.counts.skipped,
            }),
        };
        status_text(&Status {
            name: &item.name,
            position: self.live.transcribed,
            length: self.shown_length(position),
            batch,
            speed,
            elapsed: seconds,
            eta,
            width,
        })
    }

    fn opened(&mut self, length: SourceLength) {
        let Some(item) = self
            .current
            .and_then(|position| self.items.get_mut(position))
        else {
            debug!("An input opened while none was running");
            return;
        };
        item.length = length;
        item.settled = true;
    }

    fn finish(&mut self, position: usize, outcome: &Outcome) {
        let was_running = self.is_running(position);
        let Some(item) = self.items.get_mut(position) else {
            debug!("Input {position} finished but is not in the batch");
            return;
        };
        if item.finished {
            return;
        }
        item.finished = true;
        match outcome {
            Outcome::Done { .. } => {
                self.counts.done += 1;
                if was_running {
                    self.live.transcribed = self.live.transcribed.max(self.live.received);
                    self.live.source_ended = true;
                    match Duration::try_from_secs_f64(self.live.received) {
                        Ok(received) => {
                            item.length = SourceLength::Exact(received);
                            item.settled = true;
                        }
                        Err(error) => debug!("The audio received is not a duration: {error}"),
                    }
                }
            }
            Outcome::Failed { .. } => self.counts.failed += 1,
            Outcome::Skipped => self.counts.skipped += 1,
            Outcome::Interrupted { .. } | Outcome::NotStarted => {}
        }
        if was_running {
            self.audio_done += self.live.transcribed;
        }
    }
}

#[derive(Clone, Copy)]
enum Row {
    Total,
    Current,
    Status,
}

struct Shared {
    model: Mutex<Model>,
    columns: Columns,
    elapsed: Elapsed,
}

impl Shared {
    fn render(&self, row: Row) -> String {
        let columns = (self.columns)();
        let elapsed = (self.elapsed)();
        let mut model = self.model.lock();
        model.sync_probes();
        match row {
            Row::Total => model.total_rule(columns.clamp(MIN_RULE, MAX_RULE)),
            Row::Current => model.current_rule(columns.clamp(MIN_RULE, MAX_RULE), elapsed),
            Row::Status => model.status_row(elapsed, columns.saturating_sub(2)),
        }
    }
}

#[derive(Clone)]
struct RowKey {
    shared: Arc<Shared>,
    row: Row,
}

impl ProgressTracker for RowKey {
    fn clone_box(&self) -> Box<dyn ProgressTracker> {
        Box::new(self.clone())
    }

    fn tick(&mut self, _: &ProgressState, _: Instant) {}

    fn reset(&mut self, _: &ProgressState, _: Instant) {}

    fn write(&self, _: &ProgressState, w: &mut dyn fmt::Write) {
        if let Err(error) = w.write_str(&self.shared.render(self.row)) {
            debug!("A progress row could not be drawn: {error}");
        }
    }
}

pub struct Bars {
    shared: Arc<Shared>,
    rows: Vec<ProgressBar>,
}

impl Bars {
    pub fn new(plan: Plan<'_>, surface: Surface) -> CliResult<Self> {
        let layout: &[Row] = match plan.form {
            InputForm::Single => &[Row::Current, Row::Status],
            InputForm::Batch => &[Row::Total, Row::Current, Row::Status],
        };
        let shared = Arc::new(Shared {
            columns: surface.columns,
            elapsed: surface.elapsed,
            model: Mutex::new(Model::new(plan)),
        });
        let spinner = format!(
            "{{spinner{}}} {{status}}",
            progress_style(ProgressPart::Spinner, surface.colors)
        );
        let rows = layout
            .iter()
            .map(|row| {
                let (template, key) = match row {
                    Row::Total | Row::Current => ("{rule}", "rule"),
                    Row::Status => (spinner.as_str(), "status"),
                };
                let style = ProgressStyle::with_template(template)
                    .map_err(|source| CliError::ProgressTemplate { source })?
                    .tick_strings(&SPINNER)
                    .with_key(
                        key,
                        RowKey {
                            shared: Arc::clone(&shared),
                            row: *row,
                        },
                    );
                let bar = (surface.place)(ProgressBar::no_length().with_style(style));
                if let Some(interval) = surface.steady {
                    bar.enable_steady_tick(interval);
                }
                Ok(bar)
            })
            .collect::<CliResult<Vec<_>>>()?;
        Ok(Self { shared, rows })
    }

    pub fn start_input(&self, position: usize) {
        {
            let mut model = self.shared.model.lock();
            model.current = Some(position);
            model.live = Live::default();
        }
        self.redraw();
    }

    pub fn opened(&self, length: SourceLength) {
        self.shared.model.lock().opened(length);
        self.redraw();
    }

    pub fn update(&self, progress: &StreamProgress) {
        {
            let mut model = self.shared.model.lock();
            model.live = Live {
                transcribed: progress.frame.transcribed_to,
                received: progress.received,
                source_ended: progress.source_ended,
            };
        }
        self.redraw();
    }

    pub fn finish_input(&self, position: usize, outcome: &Outcome) {
        self.shared.model.lock().finish(position, outcome);
        self.redraw();
    }

    pub fn finish(self) {
        drop(self);
    }

    fn redraw(&self) {
        for row in &self.rows {
            row.force_draw();
        }
    }
}

impl Drop for Bars {
    fn drop(&mut self) {
        self.shared.model.lock().probes = None;
        for row in &self.rows {
            row.finish_and_clear();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use indicatif::{InMemoryTerm, MultiProgress, ProgressDrawTarget};
    use pretty_assertions::assert_eq;
    use purr_common::progress::{ProgressFrame, ProgressPhase};
    use purr_core::pipeline::events::EndReason;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    const COLUMNS: u16 = 90;
    const WIDTH: usize = 90;

    #[test]
    fn a_rule_fills_whole_cells_and_a_half_cell() {
        assert_eq!(rule_fill(0.0, 10), (0, false));
        assert_eq!(rule_fill(0.5, 10), (5, false));
        assert_eq!(rule_fill(0.55, 10), (5, true));
        assert_eq!(rule_fill(1.0, 10), (10, false));
        assert_eq!(rule_fill(0.99, 10), (9, true));
    }

    #[test]
    fn a_rule_fraction_beyond_the_ends_is_clamped() {
        assert_eq!(rule_fill(1.7, 10), (10, false));
        assert_eq!(rule_fill(-0.3, 10), (0, false));
    }

    #[test]
    fn the_segment_sweeps_to_the_far_end_and_back() {
        let at = |millis| bounce_position(Duration::from_millis(millis), 20, 4);
        assert_eq!(at(0), 0);
        assert_eq!(at(750), 8);
        assert_eq!(at(1500), 16);
        assert_eq!(at(2250), 8);
        assert_eq!(at(3000), 0);
        assert_eq!(at(3750), 8);
    }

    #[test]
    fn the_segment_stays_on_a_rule_no_wider_than_itself() {
        assert_eq!(bounce_position(Duration::from_millis(900), 4, 4), 0);
        assert_eq!(bounce_position(Duration::from_millis(900), 3, 4), 0);
    }

    #[test]
    fn a_batch_total_weights_each_input_by_its_audio() {
        let order = |actual: f64, expected: f64| actual.total_cmp(&expected);
        assert_eq!(
            order(total_fraction(60.0, 30.0, 0.5, 180.0), 75.0 / 180.0),
            std::cmp::Ordering::Equal
        );
        assert_eq!(
            order(total_fraction(0.0, 0.0, 0.0, 0.0), 0.0),
            std::cmp::Ordering::Equal
        );
        assert_eq!(
            order(total_fraction(180.0, 0.0, 0.0, 180.0), 1.0),
            std::cmp::Ordering::Equal
        );
    }

    struct Fixture {
        term: InMemoryTerm,
        elapsed: Arc<AtomicU64>,
        probes: watch::Sender<Vec<SourceLength>>,
        bars: Bars,
    }

    impl Fixture {
        fn new(
            names: &[&str],
            lengths: &[SourceLength],
            skipped: &[usize],
            form: InputForm,
            duration: Option<Duration>,
        ) -> Self {
            let items: Vec<InputItem> = names
                .iter()
                .enumerate()
                .map(|(position, name)| InputItem {
                    index: position + 1,
                    source: crate::inputs::InputSource::File(PathBuf::from(name)),
                    display: (*name).to_owned(),
                })
                .collect();
            let term = InMemoryTerm::new(10, COLUMNS);
            let multi = MultiProgress::with_draw_target(ProgressDrawTarget::term_like(Box::new(
                term.clone(),
            )));
            let elapsed = Arc::new(AtomicU64::new(0));
            let clock = Arc::clone(&elapsed);
            let (probes, receiver) = watch::channel(lengths.to_vec());
            let surface = Surface {
                place: Box::new(move |bar| multi.add(bar)),
                columns: Arc::new(|| usize::from(COLUMNS)),
                elapsed: Arc::new(move || Duration::from_millis(clock.load(Ordering::Relaxed))),
                steady: None,
                colors: false,
            };
            let bars = Bars::new(
                Plan {
                    items: &items,
                    skipped,
                    form,
                    probes: receiver,
                    duration,
                },
                surface,
            )
            .unwrap();
            Self {
                term,
                elapsed,
                probes,
                bars,
            }
        }

        fn at(&self, millis: u64) {
            self.elapsed.store(millis, Ordering::Relaxed);
        }

        fn rows(&self) -> String {
            self.bars.redraw();
            self.term.contents()
        }
    }

    fn secs(seconds: u64) -> SourceLength {
        SourceLength::Exact(Duration::from_secs(seconds))
    }

    fn progress(transcribed_to: f64, received: f64, source_ended: bool) -> StreamProgress {
        StreamProgress {
            frame: ProgressFrame {
                transcribed_to,
                decoding: None,
                phase: ProgressPhase::Transcribing,
            },
            received,
            source_ended,
        }
    }

    fn done() -> Outcome {
        Outcome::Done {
            ended: EndReason::SourceEnded,
        }
    }

    fn rule(full: usize, half: bool) -> String {
        filled_rule_for_test(full, half, WIDTH)
    }

    fn filled_rule_for_test(full: usize, half: bool, width: usize) -> String {
        let rest = width - full - usize::from(half);
        format!(
            "{}{}{}",
            DONE.to_string().repeat(full),
            if half {
                HALF.to_string()
            } else {
                String::new()
            },
            REST.to_string().repeat(rest)
        )
    }

    #[test]
    fn a_single_input_at_its_start_has_an_empty_rule() {
        let fixture = Fixture::new(&["jfk.wav"], &[secs(27)], &[], InputForm::Single, None);
        fixture.bars.start_input(0);
        assert_eq!(
            fixture.rows(),
            format!("{}\n⠋ jfk.wav  0:00 / 0:27  0:00", rule(0, false))
        );
    }

    #[test]
    fn a_single_input_half_way_fills_half_the_rule_and_shows_speed_and_eta() {
        let fixture = Fixture::new(&["jfk.wav"], &[secs(27)], &[], InputForm::Single, None);
        fixture.bars.start_input(0);
        fixture.at(4_000);
        fixture.bars.update(&progress(13.5, 20.0, false));
        assert_eq!(
            fixture.rows(),
            format!(
                "{}\n⠋ jfk.wav  0:13 / 0:27  3.4x  0:04  ETA 0:04",
                rule(45, false)
            )
        );
    }

    #[test]
    fn a_single_input_that_is_done_has_a_full_rule_and_the_audio_it_received() {
        let fixture = Fixture::new(&["jfk.wav"], &[secs(27)], &[], InputForm::Single, None);
        fixture.bars.start_input(0);
        fixture.at(9_000);
        fixture.bars.update(&progress(27.0, 27.0, true));
        fixture.bars.finish_input(0, &done());
        assert_eq!(
            fixture.rows(),
            format!("{}\n⠋ jfk.wav  0:27 / 0:27  3.0x  0:09", rule(90, false))
        );
    }

    #[test]
    fn an_estimated_length_is_clamped_at_a_full_bar_until_the_source_ends() {
        let fixture = Fixture::new(
            &["a.mp3"],
            &[SourceLength::Estimated(Duration::from_secs(10))],
            &[],
            InputForm::Single,
            None,
        );
        fixture.bars.start_input(0);
        fixture.bars.update(&progress(12.0, 14.0, false));
        assert_eq!(
            fixture.rows(),
            format!("{}\n⠋ a.mp3  0:12 / ~0:10  0:00", rule(90, false))
        );
        fixture.bars.update(&progress(12.0, 24.0, true));
        assert_eq!(
            fixture.rows(),
            format!("{}\n⠋ a.mp3  0:12 / 0:24  0:00", rule(45, false))
        );
    }

    #[test]
    fn an_unknown_length_bounces_a_segment_and_shows_no_total() {
        let fixture = Fixture::new(
            &["radio"],
            &[SourceLength::Unknown],
            &[],
            InputForm::Single,
            None,
        );
        fixture.bars.start_input(0);
        fixture.at(750);
        fixture.bars.update(&progress(42.0, 50.0, false));
        let segment = 6;
        let start = 42;
        assert_eq!(
            fixture.rows(),
            format!(
                "{}{}{}\n⠋ radio  0:42  56.0x  0:00",
                REST.to_string().repeat(start),
                DONE.to_string().repeat(segment),
                REST.to_string().repeat(WIDTH - start - segment)
            )
        );
    }

    #[test]
    fn an_unknown_length_with_a_duration_fills_toward_the_duration() {
        let fixture = Fixture::new(
            &["radio"],
            &[SourceLength::Unknown],
            &[],
            InputForm::Single,
            Some(Duration::from_secs(60)),
        );
        fixture.bars.start_input(0);
        fixture.at(10_000);
        fixture.bars.update(&progress(30.0, 35.0, false));
        assert_eq!(
            fixture.rows(),
            format!("{}\n⠋ radio  0:30  3.0x  0:10  ETA 0:10", rule(45, false))
        );
    }

    #[test]
    fn a_batch_with_two_of_three_done_draws_three_rows() {
        let fixture = Fixture::new(
            &["a.wav", "b.wav", "c.wav"],
            &[secs(60), secs(30), secs(90)],
            &[],
            InputForm::Batch,
            None,
        );
        for (position, seconds) in [(0, 60.0), (1, 30.0)] {
            fixture.bars.start_input(position);
            fixture.bars.update(&progress(seconds, seconds, true));
            fixture.bars.finish_input(position, &done());
        }
        fixture.bars.start_input(2);
        fixture.at(30_000);
        fixture.bars.update(&progress(45.0, 50.0, false));
        assert_eq!(
            fixture.rows(),
            format!(
                "{}\n{}\n⠋ [3/3] c.wav  0:45 / 1:30  2 done, 0 failed, 0 skipped  4.5x  0:30  ETA 0:10",
                rule(67, true),
                rule(45, false),
            )
        );
    }

    #[test]
    fn a_batch_without_every_length_counts_inputs_and_shows_no_eta() {
        let fixture = Fixture::new(
            &["a.wav", "b.wav"],
            &[secs(60), SourceLength::Unknown],
            &[],
            InputForm::Batch,
            None,
        );
        fixture.bars.start_input(0);
        fixture.bars.update(&progress(60.0, 60.0, true));
        fixture.bars.finish_input(0, &done());
        fixture.bars.start_input(1);
        fixture.at(10_000);
        fixture.bars.update(&progress(20.0, 25.0, false));
        let rows = fixture.rows();
        let lines: Vec<&str> = rows.lines().collect();
        assert_eq!(lines.first().copied(), Some(rule(45, false).as_str()));
        assert_eq!(
            lines.last().copied(),
            Some("⠋ [2/2] b.wav  0:20  1 done, 0 failed, 0 skipped  8.0x  0:10")
        );
    }

    #[test]
    fn probed_lengths_that_arrive_late_switch_the_total_to_audio_weights() {
        let fixture = Fixture::new(
            &["a.wav", "b.wav"],
            &[SourceLength::Unknown, SourceLength::Unknown],
            &[],
            InputForm::Batch,
            None,
        );
        fixture.bars.start_input(0);
        fixture.bars.update(&progress(10.0, 12.0, false));
        assert_eq!(fixture.rows().lines().next(), Some(rule(0, false).as_str()));
        fixture.probes.send_replace(vec![secs(20), secs(60)]);
        assert_eq!(
            fixture.rows().lines().next(),
            Some(rule(11, false).as_str())
        );
    }

    #[test]
    fn skipped_inputs_are_counted_and_leave_the_total() {
        let fixture = Fixture::new(
            &["a.wav", "b.wav"],
            &[secs(60), secs(60)],
            &[0],
            InputForm::Batch,
            None,
        );
        fixture.bars.finish_input(0, &Outcome::Skipped);
        fixture.bars.start_input(1);
        fixture.at(10_000);
        fixture.bars.update(&progress(30.0, 35.0, false));
        assert_eq!(
            fixture.rows(),
            format!(
                "{}\n{}\n⠋ [2/2] b.wav  0:30 / 1:00  0 done, 0 failed, 1 skipped  3.0x  0:10  ETA 0:10",
                rule(45, false),
                rule(45, false),
            )
        );
    }

    #[test]
    fn finishing_the_rows_clears_them() {
        let fixture = Fixture::new(&["jfk.wav"], &[secs(27)], &[], InputForm::Single, None);
        fixture.bars.start_input(0);
        let term = fixture.term.clone();
        fixture.bars.finish();
        assert_eq!(term.contents(), "");
    }
}
