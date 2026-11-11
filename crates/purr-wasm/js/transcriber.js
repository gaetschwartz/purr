import { WhisperTextStreamer } from "@huggingface/transformers";
import { emit } from "./events.js";
import { detectLanguage } from "./language.js";

const SAMPLE_RATE = 16000;
const CHUNK_LENGTH_S = 30;
const STRIDE_LENGTH_S = 5;
const SEGMENT_UPDATE_INTERVAL_MS = 80;

function wordCount(text) {
    return text.split(/\s+/).filter((word) => word.length > 0).length;
}

function createStreamer(transcriber, requestId, segments) {
    const timePrecision =
        transcriber.processor.feature_extractor.config.chunk_length /
        transcriber.model.config.max_source_positions;
    let windowCount = 0;
    let lastEmit = 0;

    const publish = (force) => {
        const now = performance.now();
        if (!force && now - lastEmit < SEGMENT_UPDATE_INTERVAL_MS) {
            return;
        }
        lastEmit = now;
        const index = segments.length - 1;
        const segment = segments[index];
        emit({
            type: "Segment",
            request_id: requestId,
            chunk_index: index,
            text: segment.text,
            start_time: segment.timestamp[0] ?? 0,
            end_time: segment.timestamp[1] ?? segment.timestamp[0] ?? 0,
        });
    };

    return new WhisperTextStreamer(transcriber.tokenizer, {
        time_precision: timePrecision,
        on_chunk_start: (start) => {
            const offset = (CHUNK_LENGTH_S - STRIDE_LENGTH_S) * windowCount;
            segments.push({ text: "", timestamp: [offset + start, null], offset });
        },
        callback_function: (text) => {
            if (segments.length === 0) {
                return;
            }
            segments[segments.length - 1].text += text;
            publish(false);
        },
        on_chunk_end: (end) => {
            if (segments.length === 0) {
                return;
            }
            const current = segments[segments.length - 1];
            current.timestamp[1] = end + current.offset;
            publish(true);
        },
        on_finalize: () => {
            windowCount += 1;
        },
    });
}

export async function transcribe(transcriber, request) {
    const { request_id: requestId, audio, language, translate, english_only: englishOnly } = request;
    const startedAt = performance.now();
    const segments = [];
    const streamer = createStreamer(transcriber, requestId, segments);

    const options = {
        top_k: 0,
        do_sample: false,
        chunk_length_s: CHUNK_LENGTH_S,
        stride_length_s: STRIDE_LENGTH_S,
        return_timestamps: true,
        force_full_sequences: false,
        streamer,
    };
    if (!englishOnly) {
        options.task = translate ? "translate" : "transcribe";
        options.language = language ?? (await detectLanguage(transcriber, audio));
        if (!language) {
            emit({ type: "LanguageDetected", request_id: requestId, language: options.language });
        }
    }

    const audioDuration = audio.length / SAMPLE_RATE;
    const output = await transcriber(audio, options);
    const processingTime = (performance.now() - startedAt) / 1000;

    const chunks = Array.isArray(output) ? output[0].chunks : output.chunks;
    const finalSegments = chunks.map((chunk) => ({
        text: chunk.text,
        start_time: chunk.timestamp[0] ?? 0,
        end_time: chunk.timestamp[1] ?? audioDuration,
    }));
    emit({ type: "Segments", request_id: requestId, segments: finalSegments });
    emit({
        type: "Completed",
        request_id: requestId,
        processing_time: processingTime,
        audio_duration: audioDuration,
        word_count: wordCount(finalSegments.map((segment) => segment.text).join(" ")),
    });
}
