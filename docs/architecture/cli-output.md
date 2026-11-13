# CLI output styling

All colour in `purr-cli` goes through one theme. Nothing outside `theme.rs` names a colour, and
`purr-core` produces no styled text.

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
handlers use. `global::args()` is `None` before that (early errors, tests); `verbosity()` and
`colors()` then answer with their defaults (normal, `--color auto`).

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

## Using it

- `use crate::theme::Themed as _;` then `"Info:".info()`, `size.quantity()`, `path.display().path()`.
  A styled value prints as the exact plain text when colours are off for its stream.
- Text is for stdout unless formatted inside `scope::on_stderr(|| ...)`, which makes the stderr
  decision apply. It is a closure so it cannot be held across an `.await`.
- The log formatter is the exception: it styles with the colour setting of the writer it writes
  to (`.with_ansi(stderr decision)`), so log messages themselves must stay plain. tracing escapes
  ESC in message fields, so pre-styled text there shows up as literal `\x1b[...`.
- JSON, SRT and transcript text are never styled.

## Adding a role

Add a variant to `Role`, give it a `Look` in `Role::look`, add a `Themed` method if call sites
need one, and extend the escape sequence table in the `theme` tests.
