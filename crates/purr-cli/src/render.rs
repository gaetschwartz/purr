use crate::cli::OutputFormat;
use crate::error::CliError;
use crate::json;
use purr_common::platform::ChunkKind;
use purr_core::pipeline::{events::StreamingChunk, result::SyncTranscriptionResult};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rendering {
    pub format: OutputFormat,
    pub timestamps: bool,
    pub word_timestamps: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layout {
    File,
    Stdout { one_chunk_per_line: bool },
}

pub fn streaming_piece(
    r: &Rendering,
    chunk: &StreamingChunk,
    input: Option<&str>,
) -> Result<Option<String>, CliError> {
    match chunk.kind {
        ChunkKind::Committed => {}
        ChunkKind::Tentative => return Ok(None),
    }
    if chunk.text.is_empty() && r.format != OutputFormat::Json {
        return Ok(None);
    }
    let piece = match r.format {
        OutputFormat::Text if r.timestamps => {
            format!("[{:.2}s -> {:.2}s] {}", chunk.start, chunk.end, chunk.text)
        }
        OutputFormat::Text | OutputFormat::Txt => chunk.text.clone(),
        OutputFormat::Json => serde_json::to_string(&json::chunk(chunk, r.word_timestamps, input))
            .map_err(|source| CliError::SerializeJson { source })?,
        OutputFormat::Srt => format!(
            "{}\n{} --> {}\n{}\n",
            chunk.chunk_index + 1,
            format_srt_time(chunk.start),
            format_srt_time(chunk.end),
            chunk.text
        ),
    };
    Ok(Some(piece))
}

pub fn terminator(r: &Rendering, chunk: &StreamingChunk, layout: Layout) -> &'static str {
    match layout {
        Layout::File => match r.format {
            OutputFormat::Txt => "",
            OutputFormat::Text | OutputFormat::Json | OutputFormat::Srt => "\n",
        },
        Layout::Stdout { one_chunk_per_line } => match r.format {
            OutputFormat::Json => "\n",
            OutputFormat::Text | OutputFormat::Txt | OutputFormat::Srt => {
                if chunk.text.is_empty() || chunk.text.ends_with('\n') {
                    ""
                } else if r.format == OutputFormat::Srt || one_chunk_per_line {
                    "\n"
                } else {
                    " "
                }
            }
        },
    }
}

pub fn stream_end(layout: Layout) -> &'static str {
    match layout {
        Layout::File => "",
        Layout::Stdout { .. } => "\n",
    }
}

pub fn document_piece(
    r: &Rendering,
    result: &SyncTranscriptionResult,
    input: Option<&str>,
) -> Result<String, CliError> {
    let piece = match r.format {
        OutputFormat::Text if r.timestamps => result
            .segments
            .iter()
            .map(|segment| {
                format!(
                    "[{:.2}s -> {:.2}s] {}",
                    segment.start, segment.end, segment.text
                )
            })
            .collect::<Vec<_>>()
            .join("\n"),
        OutputFormat::Text | OutputFormat::Txt => result.text.clone(),
        OutputFormat::Json => {
            let document = json::document(result, r.word_timestamps, input);
            match input {
                Some(_) => serde_json::to_string(&document),
                None => serde_json::to_string_pretty(&document),
            }
            .map_err(|source| CliError::SerializeJson { source })?
        }
        OutputFormat::Srt => result
            .segments
            .iter()
            .enumerate()
            .map(|(i, segment)| {
                format!(
                    "{}\n{} --> {}\n{}\n",
                    i + 1,
                    format_srt_time(segment.start),
                    format_srt_time(segment.end),
                    segment.text
                )
            })
            .collect::<Vec<_>>()
            .join("\n"),
    };
    Ok(piece)
}

pub fn header(display: &str) -> String {
    format!("==> {display} <==\n")
}

fn format_srt_time(seconds: f64) -> String {
    let hours = (seconds / 3600.0) as u32;
    let minutes = ((seconds % 3600.0) / 60.0) as u32;
    let secs = (seconds % 60.0) as u32;
    let millis = ((seconds % 1.0) * 1000.0) as u32;

    format!("{hours:02}:{minutes:02}:{secs:02},{millis:03}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;
    use purr_common::accel::AccelBackend;
    use purr_common::devices::DeviceKey;
    use purr_common::platform::ChunkKind;
    use purr_common::word::Word;
    use purr_core::accel::{backend::AccelBackend as CoreBackend, report::AccelReport};
    use purr_core::pipeline::{events::TranscriptionStats, result::TranscriptionSegment};

    fn chunk(text: &str, start: f64, end: f64, index: usize) -> StreamingChunk {
        StreamingChunk {
            text: text.to_owned(),
            start,
            end,
            is_final: false,
            chunk_index: index,
            final_stats: None,
            language: None,
            kind: ChunkKind::Committed,
            words: Vec::new(),
        }
    }

    fn rendering(format: OutputFormat, timestamps: bool) -> Rendering {
        Rendering {
            format,
            timestamps,
            word_timestamps: false,
        }
    }

    fn stream(r: &Rendering, chunks: &[StreamingChunk], layout: Layout) -> String {
        let mut out = String::new();
        for chunk in chunks {
            if let Some(piece) = streaming_piece(r, chunk, None).unwrap() {
                out.push_str(&piece);
                out.push_str(terminator(r, chunk, layout));
            }
        }
        out.push_str(stream_end(layout));
        out
    }

    const INLINE: Layout = Layout::Stdout {
        one_chunk_per_line: false,
    };
    const LINES: Layout = Layout::Stdout {
        one_chunk_per_line: true,
    };

    fn two_chunks() -> Vec<StreamingChunk> {
        vec![
            chunk("Hello", 0.0, 1.0, 0),
            chunk("", 1.0, 1.5, 1),
            chunk("world.", 1.5, 2.5, 2),
        ]
    }

    #[test]
    fn text_streams_to_stdout_separated_by_spaces_and_ends_with_a_newline() {
        let r = rendering(OutputFormat::Text, false);
        assert_eq!(stream(&r, &two_chunks(), INLINE), "Hello world. \n");
    }

    #[test]
    fn text_streams_one_chunk_per_line_when_a_bar_is_drawn() {
        let r = rendering(OutputFormat::Text, false);
        assert_eq!(stream(&r, &two_chunks(), LINES), "Hello\nworld.\n\n");
    }

    #[test]
    fn text_with_timestamps_prefixes_each_chunk() {
        let r = rendering(OutputFormat::Text, true);
        assert_eq!(
            stream(&r, &two_chunks(), INLINE),
            "[0.00s -> 1.00s] Hello [1.50s -> 2.50s] world. \n"
        );
    }

    #[test]
    fn text_to_a_file_puts_each_chunk_on_its_own_line() {
        let r = rendering(OutputFormat::Text, false);
        assert_eq!(stream(&r, &two_chunks(), Layout::File), "Hello\nworld.\n");
    }

    #[test]
    fn text_chunk_ending_in_a_newline_gets_no_separator_on_stdout() {
        let r = rendering(OutputFormat::Text, false);
        let chunks = [chunk("line\n", 0.0, 1.0, 0), chunk("next", 1.0, 2.0, 1)];
        assert_eq!(stream(&r, &chunks, INLINE), "line\nnext \n");
    }

    #[test]
    fn txt_streams_to_stdout_like_text() {
        let r = rendering(OutputFormat::Txt, false);
        assert_eq!(stream(&r, &two_chunks(), INLINE), "Hello world. \n");
    }

    #[test]
    fn txt_ignores_timestamps() {
        let r = rendering(OutputFormat::Txt, true);
        assert_eq!(stream(&r, &two_chunks(), INLINE), "Hello world. \n");
    }

    #[test]
    fn txt_to_a_file_is_raw_without_separators() {
        let r = rendering(OutputFormat::Txt, false);
        assert_eq!(stream(&r, &two_chunks(), Layout::File), "Helloworld.");
    }

    #[test]
    fn json_streams_one_compact_line_per_chunk_even_when_the_text_is_empty() {
        let r = rendering(OutputFormat::Json, false);
        assert_eq!(
            stream(&r, &two_chunks(), INLINE),
            concat!(
                r#"{"text":"Hello","start":0.0,"end":1.0,"is_final":false,"chunk_index":0,"final_stats":null,"language":null}"#,
                "\n",
                r#"{"text":"","start":1.0,"end":1.5,"is_final":false,"chunk_index":1,"final_stats":null,"language":null}"#,
                "\n",
                r#"{"text":"world.","start":1.5,"end":2.5,"is_final":false,"chunk_index":2,"final_stats":null,"language":null}"#,
                "\n",
                "\n",
            )
        );
    }

    #[test]
    fn json_to_a_file_is_json_lines() {
        let r = rendering(OutputFormat::Json, false);
        assert_eq!(
            stream(&r, &[chunk("Hi", 0.0, 1.0, 0)], Layout::File),
            concat!(
                r#"{"text":"Hi","start":0.0,"end":1.0,"is_final":false,"chunk_index":0,"final_stats":null,"language":null}"#,
                "\n",
            )
        );
    }

    #[test]
    fn json_line_carries_input_only_when_given() {
        let r = rendering(OutputFormat::Json, false);
        let c = chunk("Hi", 0.0, 1.0, 0);
        assert_eq!(
            streaming_piece(&r, &c, None).unwrap(),
            Some(
                r#"{"text":"Hi","start":0.0,"end":1.0,"is_final":false,"chunk_index":0,"final_stats":null,"language":null}"#
                    .to_owned()
            )
        );
        assert_eq!(
            streaming_piece(&r, &c, Some("a.wav")).unwrap(),
            Some(
                r#"{"input":"a.wav","text":"Hi","start":0.0,"end":1.0,"is_final":false,"chunk_index":0,"final_stats":null,"language":null}"#
                    .to_owned()
            )
        );
    }

    #[test]
    fn input_is_ignored_by_the_other_formats() {
        let c = chunk("Hi", 0.0, 1.0, 0);
        for format in [OutputFormat::Text, OutputFormat::Srt, OutputFormat::Txt] {
            let r = rendering(format, false);
            assert_eq!(
                streaming_piece(&r, &c, Some("a.wav")).unwrap(),
                streaming_piece(&r, &c, None).unwrap()
            );
        }
    }

    #[test]
    fn json_words_appear_only_when_word_timestamps_are_asked_for() {
        let mut c = chunk("ask not", 3.0, 5.0, 0);
        c.words = vec![
            Word {
                range: 0..3,
                start_cs: Some(349),
                end_cs: Some(424),
                score: Some(191),
            },
            Word {
                range: 4..7,
                start_cs: None,
                end_cs: None,
                score: None,
            },
        ];
        let without = rendering(OutputFormat::Json, false);
        let with = Rendering {
            word_timestamps: true,
            ..without
        };
        assert_eq!(
            streaming_piece(&without, &c, None).unwrap(),
            Some(
                r#"{"text":"ask not","start":3.0,"end":5.0,"is_final":false,"chunk_index":0,"final_stats":null,"language":null}"#
                    .to_owned()
            )
        );
        assert_eq!(
            streaming_piece(&with, &c, None).unwrap(),
            Some(
                r#"{"text":"ask not","start":3.0,"end":5.0,"is_final":false,"chunk_index":0,"final_stats":null,"language":null,"words":[{"text":"ask","start":3.49,"end":4.24,"score":0.749},{"text":"not","start":null,"end":null,"score":null}]}"#
                    .to_owned()
            )
        );
    }

    #[test]
    fn srt_streams_numbered_blocks_followed_by_a_blank_line() {
        let r = rendering(OutputFormat::Srt, false);
        let chunks = [chunk("Hi", 0.0, 1.5, 0), chunk("There", 3661.25, 3662.0, 1)];
        assert_eq!(
            stream(&r, &chunks, INLINE),
            concat!(
                "1\n00:00:00,000 --> 00:00:01,500\nHi\n",
                "\n",
                "2\n01:01:01,250 --> 01:01:02,000\nThere\n",
                "\n",
                "\n",
            )
        );
    }

    #[test]
    fn srt_to_a_file_is_the_same_blocks() {
        let r = rendering(OutputFormat::Srt, false);
        assert_eq!(
            stream(&r, &[chunk("Hi", 0.0, 1.5, 0)], Layout::File),
            "1\n00:00:00,000 --> 00:00:01,500\nHi\n\n"
        );
    }

    #[test]
    fn srt_block_is_one_line_per_chunk_regardless_of_the_bar() {
        let r = rendering(OutputFormat::Srt, false);
        let chunks = [chunk("Hi", 0.0, 1.5, 0)];
        assert_eq!(stream(&r, &chunks, INLINE), stream(&r, &chunks, LINES));
    }

    #[test]
    fn empty_text_is_skipped_unless_the_format_is_json() {
        let empty = chunk("", 0.0, 1.0, 0);
        for format in [OutputFormat::Text, OutputFormat::Srt, OutputFormat::Txt] {
            let r = rendering(format, false);
            assert_eq!(streaming_piece(&r, &empty, None).unwrap(), None);
        }
    }

    #[test]
    fn tentative_chunks_are_skipped() {
        let mut tentative = chunk("maybe", 0.0, 1.0, 0);
        tentative.kind = ChunkKind::Tentative;
        for format in [
            OutputFormat::Text,
            OutputFormat::Json,
            OutputFormat::Srt,
            OutputFormat::Txt,
        ] {
            let r = rendering(format, false);
            assert_eq!(streaming_piece(&r, &tentative, None).unwrap(), None);
        }
    }

    #[test]
    fn header_is_arrow_name_arrow_newline() {
        assert_eq!(header("a.wav"), "==> a.wav <==\n");
    }

    fn document() -> SyncTranscriptionResult {
        SyncTranscriptionResult {
            text: "Hi there".to_owned(),
            segments: vec![
                TranscriptionSegment {
                    text: "Hi".to_owned(),
                    start: 0.0,
                    end: 1.5,
                    words: Vec::new(),
                },
                TranscriptionSegment {
                    text: "there".to_owned(),
                    start: 1.5,
                    end: 3.0,
                    words: Vec::new(),
                },
            ],
            processing_time: 2.0,
            audio_duration: 3.0,
            stats: TranscriptionStats {
                processing_time: 2.0,
                audio_duration: 3.0,
                segment_count: 2,
                word_count: 2,
            },
            accel: AccelReport {
                selected: CoreBackend::Cpu,
                key: DeviceKey {
                    backend: AccelBackend::Cpu,
                    name: "cpu".to_owned(),
                    occurrence: 0,
                },
                device: None,
                attempts: Vec::new(),
            },
        }
    }

    #[test]
    fn document_text_is_the_joined_text() {
        let r = rendering(OutputFormat::Text, false);
        assert_eq!(document_piece(&r, &document(), None).unwrap(), "Hi there");
    }

    #[test]
    fn document_text_with_timestamps_is_one_line_per_segment() {
        let r = rendering(OutputFormat::Text, true);
        assert_eq!(
            document_piece(&r, &document(), None).unwrap(),
            "[0.00s -> 1.50s] Hi\n[1.50s -> 3.00s] there"
        );
    }

    #[test]
    fn document_txt_is_the_joined_text_whatever_the_timestamps() {
        let r = rendering(OutputFormat::Txt, true);
        assert_eq!(document_piece(&r, &document(), None).unwrap(), "Hi there");
    }

    #[test]
    fn document_srt_numbers_the_segments() {
        let r = rendering(OutputFormat::Srt, false);
        assert_eq!(
            document_piece(&r, &document(), None).unwrap(),
            "1\n00:00:00,000 --> 00:00:01,500\nHi\n\n2\n00:00:01,500 --> 00:00:03,000\nthere\n"
        );
    }

    #[test]
    fn document_piece_json_is_pretty_for_single_and_compact_when_input_is_given() {
        let r = rendering(OutputFormat::Json, false);
        assert_eq!(
            document_piece(&r, &document(), None).unwrap(),
            r#"{
  "text": "Hi there",
  "segments": [
    {
      "text": "Hi",
      "start": 0.0,
      "end": 1.5
    },
    {
      "text": "there",
      "start": 1.5,
      "end": 3.0
    }
  ],
  "processing_time": 2.0,
  "audio_duration": 3.0,
  "stats": {
    "processing_time": 2.0,
    "audio_duration": 3.0,
    "segment_count": 2,
    "word_count": 2
  },
  "accel": {
    "selected": "cpu",
    "key": {
      "backend": "Cpu",
      "name": "cpu",
      "occurrence": 0
    },
    "device": null,
    "attempts": []
  }
}"#
        );
        assert_eq!(
            document_piece(&r, &document(), Some("a.wav")).unwrap(),
            r#"{"input":"a.wav","text":"Hi there","segments":[{"text":"Hi","start":0.0,"end":1.5},{"text":"there","start":1.5,"end":3.0}],"processing_time":2.0,"audio_duration":3.0,"stats":{"processing_time":2.0,"audio_duration":3.0,"segment_count":2,"word_count":2},"accel":{"selected":"cpu","key":{"backend":"Cpu","name":"cpu","occurrence":0},"device":null,"attempts":[]}}"#
        );
    }
}
