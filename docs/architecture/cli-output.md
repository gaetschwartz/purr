# CLI output

Everything `purr-cli` prints goes through one output layer (`output/`), on a named channel. The
channel decides the stream, the stream decides the colours, and the theme decides the look.
`purr-core` and `purr-common` produce no styled text and keep logging through `tracing`.

## Channels

| Channel | Used for | Goes to |
| --- | --- | --- |
| Result | what the command produces: the transcript (text, SRT, JSON), `purr sys`, `models list`, `runtime status` | stdout, always |
| Status | messages to a person: the banner, `Transcribing:` / `Language:` lines, progress bars, notices, the download prompt, the statistics block | stdout in interactive mode, dropped in plain mode |
| Error | the error that ends the run (`Error: ...`, exit code 1) | stderr, always |
| Trace | diagnostic lines a flag asks for (`--trace-beams`) | stderr, in both modes |
| Log | the `tracing` events, whisper and ggml text included | stderr with `-v`, nothing otherwise |

`channel::route(channel, mode, verbosity)` is the one place that maps a channel to a stream (or to
nothing); it is a pure function with a test over the whole matrix. A command whose product is a
report (`purr sys`, `models list`) prints it as its result in both modes, so `purr sys --json`
always leaves exactly one JSON document on stdout.

## Plain and interactive mode (`output/mode.rs`)

The mode is decided once, in `global::init`: plain when `--plain` is given or stdout is not a
terminal, interactive otherwise. In plain mode stdout carries only the result (the transcript, with
no banner, status line, progress bar or statistics) and nothing asks a question: a missing model is
an error that names the `purr models download` command instead of a prompt. `--plain` does not
change colours; `--color` and the environment still decide those.

In interactive mode `--progress` draws one line under the transcript, and the transcript is then
written one chunk per line so the bar never overwrites a partial line.

`--help`, `--version` and usage errors are written by clap, outside the layer.

## Using the layer

```rust
use crate::{resultln, statusln};

resultln!("{}", report_line)?;      // fallible: a failed write fails the command
statusln!("{} {}", "Info:".info(), text);
errorln!("{}: {}", "Error".error(), error);
traceln!("{line}");
```

`result!`, `resultln!`, `status!`, `statusln!`, `errorln!` and `traceln!` take format arguments.
The text is formatted after the channel has chosen its stream, inside `scope::on_stderr` when that
is stderr, so styled values get the right colour decision without the call site knowing. A
channel that routes nowhere does not even format.

`Output` (`output/mod.rs`) holds the mode, the verbosity, a `Sink` and the one `MultiProgress`.
Every write suspends the progress bars first, so lines never land inside a bar. Bars are created
with `output::add_bar(...)`: drawn on stdout in interactive mode, invisible in plain mode, so
callers do not branch on the mode. `Sink` is a trait; `ProcessSink` writes to the process streams
and flushes each write, a test sink keeps the text in memory.

### The lint guard

`#![deny(clippy::print_stdout, clippy::print_stderr, clippy::dbg_macro)]` at the top of `main.rs`
bans `println!`, `eprintln!`, `print!`, `eprint!` and `dbg!` in the crate (tests are exempt through
`allow-print-in-tests` in `clippy.toml`). `clippy.toml` also lists `std::io::stdout` and
`std::io::stderr` under `disallowed-methods`; `output/sink.rs` is the only module that allows
them, and it also owns "is this stream a terminal" and the Windows console handle.

## Diagnostic log (`logging/`)

`tracing` events reach two independent layers, built in `logging::subscriber`:

- Terminal: nothing by default. `-v` shows info and above from `purr` and `purr-core` (whisper and
  ggml lines included) as `info: ...` on stderr; `-vv` and `-vvv` show everything, narrowed by
  `RUST_LOG`. Lines go through the Log channel, so they respect the progress bars and the stderr
  colour decision.
- Log file: `--log-file <PATH>` appends plain text, never colour, at `--log-level
  error|warn|info|debug|trace` (default `info`, only valid with `--log-file`). The level is
  independent of `-v`. Lines carry a UTC RFC 3339 timestamp, the level and the target. The file is
  opened before anything else runs; failing to open it is a typed `LogFileError`, reported as the
  error line with exit code 1.

The file is opened for appending and each event is one write, so nothing is buffered: there is
nothing to flush on success, error, `process::exit` or Ctrl-C (the CLI installs no handler). The
error that ends the run is recorded in the file on target `purr::fatal`, which the terminal layers
skip because the Error channel already shows it.

User-facing text is never a `tracing` event: it uses a channel. Events are for the log.

## Colour decision (`color.rs`)

`--color auto|always|never` is a global flag (before or after the subcommand). The decision is
taken once per stream, stdout and stderr separately, in this order:

1. `--color always` / `--color never` wins over everything, including `NO_COLOR`.
2. `NO_COLOR` set to a non-empty value: off.
3. `CLICOLOR_FORCE` set to a value other than empty or `0`: on, even when piped.
4. `CLICOLOR=0`: off.
5. Otherwise: on when the stream is a terminal and `TERM` is not `dumb`.

This is the order `anstream` (and so clap) uses, plus the explicit flag on top. On Windows the
console's escape sequence processing is switched on (`vt.rs`) for terminal streams that are on.
`purr --help` and usage errors are coloured by the same choice: `Cli::from_env` reads `--color`
ahead of the full parse.

## Global arguments (`global.rs`)

`main` calls `global::init(Cli::from_env())` once; it returns the `&'static Cli` that the
handlers use. `global::args()` is `None` before that (early errors, tests); `verbosity()`,
`colors()` and `mode()` then answer with their defaults (normal, `--color auto`, plain unless
stdout is a terminal). The global flags are `--color`, `--plain`, `--log-file`, `--log-level` and
`-v`.

## Roles (`theme.rs`)

| Role | Look | Used for |
| --- | --- | --- |
| `Heading` | bold blue | section headings, statistics titles, `--help` headers |
| `Info` | bold blue | `Info:` tags, `info` log label |
| `Success` | bold green | `Success:` tags, `ok` |
| `Warning` | bold yellow | `Warning:` / `Notice:` tags, `missing`, `warn` log label |
| `Error` | bold red | `Error:` line, `corrupt`, `error` log label |
| `Debug` | bold green | `debug` log label |
| `Trace` | bold magenta | `trace` log label |
| `Name` | green | model ids, device and processor names |
| `Positive` | green | downloaded, available, yes |
| `Negative` | red | incomplete, no, unusable |
| `Caution` | yellow | not built |
| `Quantity` | yellow | sizes and durations |
| `Muted` | dim | descriptions, notes, the progress line |
| `Path` | cyan | files, directories, model file names |
| `Command` | bold cyan | commands the user can type, `--help` literals |
| `Placeholder` | italic cyan | `--help` placeholders |

Only base colours, bold, dim and italic are used so the output reads on dark and light terminals.

- `use crate::theme::Themed as _;` then `"Info:".info()`, `size.quantity()`, `path.display().path()`.
  A styled value prints as the exact plain text when colours are off for its stream.
- The terminal log formatter styles with the colour setting of the writer it writes to
  (`.with_ansi(stderr decision)`), so log messages themselves must stay plain. tracing escapes ESC
  in message fields, so pre-styled text there shows up as literal `\x1b[...`.
- JSON, SRT and transcript text are never styled.
- `scope::on_stderr` still exists for the output layer; call sites do not use it.

To add a role, add a variant to `Role`, give it a `Look` in `Role::look`, add a `Themed` method if
call sites need one, and extend the escape sequence table in the `theme` tests.
