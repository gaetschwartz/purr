# CLI batch runs

`purr` transcribes any number of media files (audio or video) and URLs in one run. The model loads
once; inputs run one after another; each input gets its own output unless asked to merge.

## Inputs (`inputs.rs`)

- An argument is a URL (`Url::parse` with scheme `http` or `https`), a wildcard, or a path. A
  wildcard is what the `glob` crate says is one (`Pattern::escape(text) != text`), unless a file of
  that exact name exists. The CLI expands wildcards itself, so `dir\*.wav` works in `cmd` and
  PowerShell; `**` recurses. Matches are sorted and never include directories.
- `-a, --batch-file FILE` adds one input per line (`-` reads standard input; blank lines and lines
  starting with `#` are ignored; a line may be a wildcard).
- A directory is an error that suggests a wildcard. No file-type list decides what is audio:
  every named or matched file goes to FFmpeg, and one it cannot read fails as that input.
- The run is in *batch form* when more than one argument is given or any argument is a wildcard or
  a batch file, even if that expands to one input; one plain path or URL is the *single form* and
  behaves as it always has.

## Outputs (`template.rs`, `output_plan.rs`, `writer.rs`, `render.rs`)

- `-o` picks the format, `-f` the destination. In `-f`, `{stem}`, `{name}`, `{dir}`, `{ext}` and
  `{index}` stand for parts of each input; an existing directory means `<dir>/<stem>.<ext>`.
  Unknown placeholders and two inputs resolving to one path are errors before any work starts.
- Batch form without `-f` writes to stdout: text with a `==> name <==` header per input, JSON as
  JSON Lines with an `input` field; SRT needs `-f`. `-f` without placeholders needs `--merge`,
  which writes every input to one file (not SRT).
- Output is written as it is produced, to `<final path>.part`, and renamed when the input
  completes: the source ended, `--duration` ended it, or the user stopped a source of unknown
  length. A failed input, an input interrupted before a known end, or a killed process leaves its
  `.part` with everything written so far. A merged file is renamed unless Ctrl-C or `--fail-fast`
  left inputs unfinished. `--skip-existing` looks at final paths only.

## Outcomes and exit codes (`outcome.rs`, `batch.rs`)

Each input ends as done, skipped, failed, interrupted or not started. Batch form ends with a
summary on stderr (counts, failures with their errors, unfinished `.part` files, audio time and
speed). Exit code: 0 when nothing failed and nothing was left incomplete or unstarted; 1 when any
input failed; 130 when Ctrl-C cut the run short with unfinished or unstarted inputs. A failed input
does not stop the batch unless `--fail-fast` is given.

## Stopping (`interrupt.rs`)

The first Ctrl-C stops gracefully: a source of unknown length is stopped (decode what arrived,
finish) and counts as done; a source of known length is cancelled and keeps its `.part`; no further
input starts; outputs are flushed and the summary printed. The second Ctrl-C exits with 130 at
once. `--duration TIME` ends an input after that much audio but only when its length is unknown
(streams), so files and streams can be mixed in one command.

## Core pieces (`purr-core`)

- `LoadedModel` (`pipeline/model.rs`) holds one decoder across transcriptions and gets it back
  when a run ends in any way. `Transcription` reports its `SourceLength`, how it ended
  (`EndReason`) and can be stopped.
- `SourceLength` (`Exact`, `Estimated`, `Unknown`) comes from FFmpeg when the source opens: the
  container duration for files, the header or `Content-Length` and bit rate for URLs.
- URL sources stream through a bounded buffer with a connect timeout and a timeout on each wait
  for the network (so `--http-timeout` means seconds without data); a failed or stalled download
  surfaces as an error, and dropping the stream closes the download. Decoded audio and the
  session's pending samples are bounded too.
- A recorded source (a file, or a URL of known length) decides its window boundaries from the
  audio alone, only once a full window is pending or the source has ended, so a run gives the same
  split whenever the audio arrives. A live source (a URL of unknown length) decodes growing windows
  as audio arrives, so its split depends on arrival timing and can differ between runs; the text
  stays the same up to punctuation.

## Progress (`bar.rs`, `progress.rs`)

Shown in interactive mode unless `--no-progress`: a rule that follows the audio and a status row;
a batch adds a total rule above, weighted by audio length once every length is known (local files
are probed while the model loads) and by input count until then. A source of unknown length shows
a bouncing segment, or fills toward `--duration` when one is set.
