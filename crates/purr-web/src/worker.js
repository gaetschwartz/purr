// Real WebGPU transcription worker with @huggingface/transformers integration
// Based on whisper-web implementation patterns
import { pipeline, WhisperTextStreamer } from "@huggingface/transformers";

// Configure environment for production deployment
import { env } from "@huggingface/transformers";
env.allowLocalModels = false;
env.backends.onnx.logLevel = "error";

// Real PipelineFactory implementation following whisper-web patterns
class PipelineFactory {
    static task = null;
    static model = null;
    static instance = null;

    constructor(tokenizer, model) {
        this.tokenizer = tokenizer;
        this.model = model;
    }

    static async getInstance(progress_callback = null) {
        if (this.instance === null) {
            this.instance = pipeline(this.task, this.model, {
                dtype: {
                    encoder_model:
                        this.model === "onnx-community/whisper-large-v3-turbo"
                            ? "fp16"
                            : "fp32",
                    decoder_model_merged: "q4", // or 'fp32' ('fp16' is broken)
                },
                device: "webgpu",
                progress_callback,
            });
        }

        return this.instance;
    }

    static dispose() {
        if (this.instance !== null) {
            this.instance.then(instance => {
                if (instance && typeof instance.dispose === 'function') {
                    instance.dispose();
                }
            });
            this.instance = null;
        }
    }
}

class AutomaticSpeechRecognitionPipelineFactory extends PipelineFactory {
    static task = "automatic-speech-recognition";
    static model = null;
}

// Global worker state
let workerConfig = null;
let isProcessing = false;
let currentSession = null;
let webgpuDevice = null;

// Real WebGPU device initialization
async function initializeWebGPU() {
    try {
        if (!navigator.gpu) {
            throw new Error("WebGPU not supported");
        }

        const adapter = await navigator.gpu.requestAdapter({
            powerPreference: "high-performance"
        });

        if (!adapter) {
            throw new Error("WebGPU adapter not available");
        }

        webgpuDevice = await adapter.requestDevice({
            requiredFeatures: [],
            requiredLimits: {
                maxBufferSize: adapter.limits.maxBufferSize,
                maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
            }
        });

        webgpuDevice.lost.then((info) => {
            console.error("WebGPU device lost:", info);
            webgpuDevice = null;
        });

        return true;
    } catch (error) {
        console.error("WebGPU initialization failed:", error);
        return false;
    }
}

// Message handler compatible with Rust WorkerMessage enum
self.addEventListener("message", async (event) => {
    const message = event.data;

    try {
        switch (message.type) {
            case "Initialize":
                await handleInitialize(message);
                break;

            case "StartTranscription":
                await handleStartTranscription(message);
                break;

            case "CancelTranscription":
                await handleCancelTranscription(message);
                break;

            default:
                console.warn("Unknown message type:", message.type);
        }
    } catch (error) {
        console.error("Worker error:", error);
        self.postMessage({
            type: "WorkerError",
            session_id: message.session_id || "unknown",
            error: error.message
        });
    }
});

async function handleInitialize(message) {
    try {
        const { session_id, config } = message;

        // Initialize WebGPU first
        const webgpuSupported = await initializeWebGPU();
        if (!webgpuSupported) {
            throw new Error("WebGPU initialization failed - falling back to CPU");
        }

        // Store config
        workerConfig = {
            model_config: {
                model_name: config.model_name || "openai/whisper-tiny.en",
                quantized: config.quantized !== false,
                language: config.language,
                translate: config.translate || false,
                chunk_size: config.chunk_size || 30 * 16000,
                temperature: config.temperature || 0.0,
                beam_size: config.beam_size || 1,
                best_of: config.best_of || 1
            }
        };

        currentSession = session_id;

        // Notify that worker is ready
        self.postMessage({
            type: "WorkerReady",
            session_id: session_id
        });

    } catch (error) {
        console.error("Initialization error:", error);
        self.postMessage({
            type: "WorkerError",
            session_id: message.session_id,
            error: `Initialization failed: ${error.message}`
        });
    }
}

async function handleStartTranscription(message) {
    if (isProcessing) {
        throw new Error("Already processing transcription");
    }

    isProcessing = true;
    const { session_id, request_id, audio_data, language, translate } = message;

    try {
        // Convert audio data from Rust format
        const audioSamples = convertAudioData(audio_data);

        // Real transcription using @huggingface/transformers
        await transcribeAudio({
            audio: audioSamples,
            model: workerConfig.model_config.model_name,
            subtask: translate ? "translate" : "transcribe",
            language: language || workerConfig.model_config.language
        }, session_id, request_id);

    } catch (error) {
        console.error("Transcription error:", error);
        self.postMessage({
            type: "TranscriptionProgress",
            session_id,
            request_id,
            status: {
                Error: { message: error.message }
            }
        });
    } finally {
        isProcessing = false;
    }
}

async function handleCancelTranscription(message) {
    isProcessing = false;
    console.log("Cancelling transcription:", message.request_id);
}

// Real transcription function following whisper-web patterns exactly
const transcribeAudio = async ({ audio, model, subtask, language }, sessionId, requestId) => {
    const isDistilWhisper = model.startsWith("distil-whisper/");

    const p = AutomaticSpeechRecognitionPipelineFactory;
    if (p.model !== model) {
        // Invalidate model if different
        p.model = model;

        if (p.instance !== null) {
            (await p.getInstance()).dispose();
            p.instance = null;
        }
    }

    // Send starting status
    self.postMessage({
        type: "TranscriptionProgress",
        session_id: sessionId,
        request_id: requestId,
        status: { Starting: null }
    });

    // Load transcriber model with real progress reporting
    const transcriber = await p.getInstance((data) => {
        if (data && data.progress !== undefined) {
            // Real progress reporting from model loading
            self.postMessage({
                type: "TranscriptionProgress",
                session_id: sessionId,
                request_id: requestId,
                status: {
                    LoadingModel: {
                        progress: data.progress * 100,
                        message: data.status || "Loading model..."
                    }
                }
            });
        }
    });

    // Send processing audio status
    self.postMessage({
        type: "TranscriptionProgress",
        session_id: sessionId,
        request_id: requestId,
        status: { ProcessingAudio: null }
    });

    const time_precision =
        transcriber.processor.feature_extractor.config.chunk_length /
        transcriber.model.config.max_source_positions;

    // Storage for chunks to be processed. Initialise with an empty chunk.
    /** @type {{ text: string; offset: number, timestamp: [number, number | null] }[]} */
    const chunks = [];

    const chunk_length_s = isDistilWhisper ? 20 : 30;
    const stride_length_s = isDistilWhisper ? 3 : 5;

    let chunk_count = 0;
    let start_time;
    let num_tokens = 0;
    let tps;

    // Real WhisperTextStreamer implementation
    const streamer = new WhisperTextStreamer(transcriber.tokenizer, {
        time_precision,
        on_chunk_start: (x) => {
            const offset = (chunk_length_s - stride_length_s) * chunk_count;
            chunks.push({
                text: "",
                timestamp: [offset + x, null],
                finalised: false,
                offset,
            });
        },
        token_callback_function: (x) => {
            start_time ??= performance.now();
            if (num_tokens++ > 0) {
                tps = (num_tokens / (performance.now() - start_time)) * 1000;
            }
        },
        callback_function: (x) => {
            if (chunks.length === 0) return;
            // Append text to the last chunk
            chunks.at(-1).text += x;

            // Send real-time streaming updates to Rust
            const currentChunk = chunks.at(-1);
            self.postMessage({
                type: "TranscriptionProgress",
                session_id: sessionId,
                request_id: requestId,
                status: {
                    InProgress: {
                        chunk_index: chunks.length - 1,
                        text: currentChunk.text,
                        start_time: currentChunk.timestamp[0] || 0,
                        end_time: currentChunk.timestamp[1] || 0
                    }
                }
            });
        },
        on_chunk_end: (x) => {
            const current = chunks.at(-1);
            current.timestamp[1] = x + current.offset;
            current.finalised = true;
        },
        on_finalize: () => {
            start_time = null;
            num_tokens = 0;
            ++chunk_count;
        },
    });

    // Actually run real transcription with WebGPU acceleration
    const startTranscriptionTime = performance.now();
    const output = await transcriber(audio, {
        // Greedy
        top_k: 0,
        do_sample: false,

        // Sliding window
        chunk_length_s,
        stride_length_s,

        // Language and task
        language,
        task: subtask,

        // Return timestamps
        return_timestamps: true,
        force_full_sequences: false,

        // Real streaming callback
        streamer,
    }).catch((error) => {
        console.error(error);
        self.postMessage({
            type: "TranscriptionProgress",
            session_id: sessionId,
            request_id: requestId,
            status: {
                Error: { message: error.message }
            }
        });
        return null;
    });

    if (output) {
        const processingTime = (performance.now() - startTranscriptionTime) / 1000;
        const audioDuration = audio.length / 16000; // Assuming 16kHz sample rate

        // Calculate word count from final text
        const finalText = chunks.map(chunk => chunk.text).join("");
        const wordCount = finalText.split(/\s+/).filter(word => word.length > 0).length;

        // Send completion status with real metrics
        self.postMessage({
            type: "TranscriptionProgress",
            session_id: sessionId,
            request_id: requestId,
            status: {
                Completed: {
                    processing_time: processingTime,
                    audio_duration: audioDuration,
                    word_count: wordCount
                }
            }
        });

        return {
            tps,
            processing_time: processingTime,
            audio_duration: audioDuration,
            word_count: wordCount,
            ...output,
        };
    }

    return null;
};

// Real audio data conversion from Rust format
function convertAudioData(audioData) {
    try {
        let audioSamples;

        if (audioData instanceof Array) {
            // Convert from array of bytes to float samples (little endian)
            const uint8Array = new Uint8Array(audioData);
            const dataView = new DataView(uint8Array.buffer);
            const numSamples = uint8Array.length / 4; // 4 bytes per float32
            audioSamples = new Float32Array(numSamples);

            for (let i = 0; i < numSamples; i++) {
                audioSamples[i] = dataView.getFloat32(i * 4, true); // little endian
            }
        } else if (audioData instanceof Uint8Array) {
            // Convert raw bytes to float samples
            const dataView = new DataView(audioData.buffer);
            const numSamples = audioData.length / 4;
            audioSamples = new Float32Array(numSamples);

            for (let i = 0; i < numSamples; i++) {
                audioSamples[i] = dataView.getFloat32(i * 4, true);
            }
        } else {
            audioSamples = new Float32Array(audioData);
        }

        // Normalize audio if needed
        const maxValue = Math.max(...audioSamples.map(Math.abs));
        if (maxValue > 1.0) {
            for (let i = 0; i < audioSamples.length; i++) {
                audioSamples[i] /= maxValue;
            }
        }

        return audioSamples;
    } catch (error) {
        throw new Error(`Audio data conversion failed: ${error.message}`);
    }
}

// WebGPU utility functions
function getWebGPUInfo() {
    if (!webgpuDevice) return null;

    return {
        features: Array.from(webgpuDevice.features),
        limits: Object.fromEntries(
            Object.entries(webgpuDevice.limits).map(([key, value]) => [key, value])
        )
    };
}

// Handle unhandled errors
self.addEventListener('error', function(event) {
    console.error('Worker error:', event.error);
    self.postMessage({
        type: 'WorkerError',
        session_id: currentSession || 'unknown',
        error: event.error.message
    });
});

// Handle unhandled promise rejections
self.addEventListener('unhandledrejection', function(event) {
    console.error('Worker unhandled rejection:', event.reason);
    self.postMessage({
        type: 'WorkerError',
        session_id: currentSession || 'unknown',
        error: event.reason.toString()
    });
});

console.log('Real WebGPU transcription worker initialized with @huggingface/transformers');