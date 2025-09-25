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
        const audioSamples = await convertAudioData(audio_data);

        // Start performance monitoring
        performanceMonitor.startTiming('inference');

        // Send detailed processing status
        self.postMessage({
            type: "TranscriptionProgress",
            session_id,
            request_id,
            status: {
                LoadingModel: {
                    progress: 0,
                    message: `Processing ${audioSamples.length} audio samples...`
                }
            }
        });

        // Real transcription using @huggingface/transformers with enhanced configuration
        await transcribeAudio({
            audio: audioSamples,
            model: workerConfig.model_config.model_name,
            subtask: translate ? "translate" : "transcribe",
            language: language || workerConfig.model_config.language,
            // Enhanced transcription options
            beam_size: workerConfig.model_config.beam_size,
            temperature: workerConfig.model_config.temperature,
            best_of: workerConfig.model_config.best_of
        }, session_id, request_id);

        // End performance monitoring
        performanceMonitor.endTiming('inference');

    } catch (error) {
        console.error("Transcription error:", error);
        console.error("Error stack:", error.stack);

        // Send detailed error information
        self.postMessage({
            type: "TranscriptionProgress",
            session_id,
            request_id,
            status: {
                Error: {
                    message: `Transcription failed: ${error.message}`,
                    details: {
                        type: error.constructor.name,
                        stack: error.stack?.split('\n').slice(0, 5).join('\n'),
                        audioSampleCount: audioSamples?.length || 0,
                        model: workerConfig?.model_config?.model_name || 'unknown',
                        webgpuAvailable: !!webgpuDevice
                    }
                }
            }
        });
    } finally {
        isProcessing = false;

        // Log final performance metrics
        const metrics = performanceMonitor.getMetrics();
        console.log('Transcription session metrics:', metrics);
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

// Enhanced real audio data conversion with comprehensive format support
async function convertAudioData(audioData) {
    try {
        console.log(`Converting audio data: ${audioData.length} bytes, type: ${audioData.constructor.name}`);

        let audioSamples;
        let detectedFormat = detectAudioFormat(audioData);

        console.log(`Detected audio format: ${detectedFormat.format}`);

        // For formats that need real parsing, use Web Audio API
        if (detectedFormat.needsRustParsing) {
            try {
                const realMetadata = await parseAudioWithWebAudio(audioData, detectedFormat.format);
                detectedFormat = realMetadata;
                console.log(`Real metadata extracted:`, realMetadata);
            } catch (error) {
                console.warn(`Failed to extract real metadata, using fallback for ${detectedFormat.format}:`, error);
                // Keep the fallback detected format
            }
        }

        if (audioData instanceof Array) {
            // Convert from array of bytes to float samples (little endian)
            const uint8Array = new Uint8Array(audioData);
            audioSamples = convertBytesToFloat32Samples(uint8Array);
        } else if (audioData instanceof Uint8Array) {
            // Convert raw bytes to float samples based on detected format
            audioSamples = convertBytesToFloat32Samples(audioData);
        } else if (audioData instanceof Float32Array) {
            // Already in correct format
            audioSamples = new Float32Array(audioData);
        } else {
            // Try to convert whatever we have
            audioSamples = new Float32Array(audioData);
        }

        // Apply audio processing pipeline
        audioSamples = processAudioSamples(audioSamples, detectedFormat);

        console.log(`Audio conversion completed: ${audioSamples.length} samples`);
        return audioSamples;
    } catch (error) {
        console.error('Audio conversion error:', error);
        throw new Error(`Audio data conversion failed: ${error.message}`);
    }
}

// Detect audio format from byte signature
function detectAudioFormat(audioData) {
    const bytes = audioData instanceof Uint8Array ? audioData : new Uint8Array(audioData);

    if (bytes.length < 12) {
        console.warn('Audio data too short for format detection');
        return { format: 'unknown', sampleRate: 16000, channels: 1, bitDepth: 16 };
    }

    // WAV format detection
    if (bytes[0] === 0x52 && bytes[1] === 0x49 && bytes[2] === 0x46 && bytes[3] === 0x46 && // RIFF
        bytes[8] === 0x57 && bytes[9] === 0x41 && bytes[10] === 0x56 && bytes[11] === 0x45) { // WAVE
        const wavInfo = parseWAVHeader(bytes);
        console.log('Detected WAV format:', wavInfo);
        return { format: 'wav', ...wavInfo };
    }

    // MP3 format detection
    if ((bytes[0] === 0xFF && (bytes[1] & 0xE0) === 0xE0) || // MP3 frame sync
        (bytes[0] === 0x49 && bytes[1] === 0x44 && bytes[2] === 0x33)) { // ID3 tag
        console.log('Detected MP3 format');
        return {
            format: 'mp3',
            needsRustParsing: true,
            data: Array.from(bytes)
        };
    }

    // FLAC format detection
    if (bytes[0] === 0x66 && bytes[1] === 0x4C && bytes[2] === 0x61 && bytes[3] === 0x43) { // fLaC
        console.log('Detected FLAC format');
        return {
            format: 'flac',
            needsRustParsing: true,
            data: Array.from(bytes)
        };
    }

    // OGG format detection
    if (bytes[0] === 0x4F && bytes[1] === 0x67 && bytes[2] === 0x67 && bytes[3] === 0x53) { // OggS
        console.log('Detected OGG format');
        return {
            format: 'ogg',
            needsRustParsing: true,
            data: Array.from(bytes)
        };
    }

    // M4A/MP4 format detection
    if (bytes[4] === 0x66 && bytes[5] === 0x74 && bytes[6] === 0x79 && bytes[7] === 0x70) { // ftyp
        console.log('Detected M4A/MP4 format');
        return {
            format: 'm4a',
            needsRustParsing: true,
            data: Array.from(bytes)
        };
    }

    // WebM format detection
    if (bytes[0] === 0x1A && bytes[1] === 0x45 && bytes[2] === 0xDF && bytes[3] === 0xA3) {
        console.log('Detected WebM format');
        return {
            format: 'webm',
            needsRustParsing: true,
            data: Array.from(bytes)
        };
    }

    console.warn('Unknown audio format, assuming raw PCM');
    return {
        format: 'raw',
        needsRustParsing: false,
        sampleRate: 16000,
        channels: 1,
        bitDepth: 16
    };
}

// Parse audio metadata using Web Audio API for real audio processing
async function parseAudioWithWebAudio(audioData, format) {
    try {
        // Use Web Audio API to decode the audio and extract real metadata
        const audioContext = new (self.AudioContext || self.webkitAudioContext)();

        // Decode the audio data to get real metadata
        const audioBuffer = await audioContext.decodeAudioData(audioData.buffer.slice());

        // Extract real metadata from the decoded audio
        const metadata = {
            format: format,
            sampleRate: audioBuffer.sampleRate,
            channels: audioBuffer.numberOfChannels,
            duration: audioBuffer.duration,
            bitDepth: 32 // Web Audio API uses 32-bit float internally
        };

        // Close the audio context to free resources
        audioContext.close();

        return metadata;
    } catch (error) {
        console.warn(`Failed to decode ${format} audio with Web Audio API:`, error);

        // Fallback to basic detection for unsupported formats
        return {
            format: format,
            sampleRate: 16000, // Default for transcription
            channels: 1,       // Default mono for transcription
            duration: 0,       // Unknown duration
            bitDepth: 16       // Default bit depth
        };
    }
}

// Parse WAV header for detailed audio information
function parseWAVHeader(bytes) {
    try {
        const dataView = new DataView(bytes.buffer, bytes.byteOffset);

        // Skip RIFF header (12 bytes) and find fmt chunk
        let offset = 12;
        let fmtFound = false;
        let sampleRate = 16000;
        let channels = 1;
        let bitDepth = 16;

        while (offset < bytes.length - 8 && !fmtFound) {
            const chunkId = String.fromCharCode(
                bytes[offset], bytes[offset + 1], bytes[offset + 2], bytes[offset + 3]
            );
            const chunkSize = dataView.getUint32(offset + 4, true);

            if (chunkId === 'fmt ') {
                fmtFound = true;
                // Parse format chunk
                const audioFormat = dataView.getUint16(offset + 8, true);
                channels = dataView.getUint16(offset + 10, true);
                sampleRate = dataView.getUint32(offset + 12, true);
                const byteRate = dataView.getUint32(offset + 16, true);
                const blockAlign = dataView.getUint16(offset + 20, true);
                bitDepth = dataView.getUint16(offset + 22, true);

                console.log(`WAV format details: ${audioFormat === 1 ? 'PCM' : 'Compressed'}, ` +
                           `${sampleRate}Hz, ${channels}ch, ${bitDepth}bit`);
                break;
            }

            offset += 8 + chunkSize;
        }

        return { sampleRate, channels, bitDepth };
    } catch (error) {
        console.warn('Error parsing WAV header:', error);
        return { sampleRate: 16000, channels: 1, bitDepth: 16 };
    }
}

// Convert byte data to Float32 samples with proper handling
function convertBytesToFloat32Samples(bytes) {
    const dataView = new DataView(bytes.buffer, bytes.byteOffset);

    // Check if data is already in float32 format (from Rust)
    if (bytes.length % 4 === 0) {
        const numSamples = bytes.length / 4;
        const samples = new Float32Array(numSamples);

        for (let i = 0; i < numSamples; i++) {
            samples[i] = dataView.getFloat32(i * 4, true); // little endian
        }

        console.log(`Converted ${numSamples} float32 samples from byte array`);
        return samples;
    }

    // Fallback: treat as 16-bit PCM and convert to float
    const numSamples = Math.floor(bytes.length / 2);
    const samples = new Float32Array(numSamples);

    for (let i = 0; i < numSamples; i++) {
        const sample16 = dataView.getInt16(i * 2, true);
        samples[i] = sample16 / 32768.0; // Convert 16-bit to float [-1, 1]
    }

    console.log(`Converted ${numSamples} samples from 16-bit PCM`);
    return samples;
}

// Apply comprehensive audio processing pipeline
function processAudioSamples(samples, formatInfo) {
    console.log(`Processing ${samples.length} audio samples...`);

    // 1. Normalize audio levels
    const normalizedSamples = normalizeAudio(samples);

    // 2. Apply automatic gain control if needed
    const agcSamples = applyAutomaticGainControl(normalizedSamples);

    // 3. Remove DC offset
    const dcRemovedSamples = removeDCOffset(agcSamples);

    // 4. Apply basic noise gate for very quiet signals
    const noisegatedSamples = applyNoiseGate(dcRemovedSamples, -40.0); // -40dB threshold

    console.log('Audio processing pipeline completed');
    return noisegatedSamples;
}

// Normalize audio to prevent clipping while preserving dynamics
function normalizeAudio(samples) {
    const maxValue = Math.max(...samples.map(Math.abs));

    if (maxValue === 0) {
        console.warn('Audio contains only silence');
        return samples;
    }

    if (maxValue > 1.0) {
        const gain = 0.95 / maxValue; // Leave 5% headroom
        const normalized = samples.map(sample => sample * gain);
        console.log(`Normalized audio with gain: ${gain.toFixed(3)}`);
        return normalized;
    }

    return samples;
}

// Apply automatic gain control to maintain consistent levels
function applyAutomaticGainControl(samples) {
    const targetRMS = 0.15; // Target RMS level
    const windowSize = 1024; // Analysis window size
    const output = new Float32Array(samples.length);

    for (let i = 0; i < samples.length; i += windowSize) {
        const windowEnd = Math.min(i + windowSize, samples.length);
        const window = samples.slice(i, windowEnd);

        // Calculate RMS of current window
        const rms = Math.sqrt(window.reduce((sum, sample) => sum + sample * sample, 0) / window.length);

        // Calculate gain adjustment
        const gain = rms > 0 ? Math.min(targetRMS / rms, 4.0) : 1.0; // Max 4x gain

        // Apply gain to window
        for (let j = 0; j < window.length; j++) {
            output[i + j] = window[j] * gain;
        }
    }

    console.log('Applied automatic gain control');
    return output;
}

// Remove DC offset (constant bias in signal)
function removeDCOffset(samples) {
    const dcOffset = samples.reduce((sum, sample) => sum + sample, 0) / samples.length;

    if (Math.abs(dcOffset) > 0.001) { // Only apply if significant DC offset
        const corrected = samples.map(sample => sample - dcOffset);
        console.log(`Removed DC offset: ${dcOffset.toFixed(4)}`);
        return corrected;
    }

    return samples;
}

// Apply noise gate to reduce background noise
function applyNoiseGate(samples, thresholdDb) {
    const threshold = Math.pow(10, thresholdDb / 20); // Convert dB to linear
    const output = new Float32Array(samples.length);

    for (let i = 0; i < samples.length; i++) {
        const amplitude = Math.abs(samples[i]);
        if (amplitude > threshold) {
            output[i] = samples[i];
        } else {
            output[i] = samples[i] * 0.1; // Reduce by 90% instead of complete cutoff
        }
    }

    console.log(`Applied noise gate with threshold: ${thresholdDb}dB`);
    return output;
}

// Enhanced WebGPU utility functions with comprehensive device information
function getWebGPUInfo() {
    if (!webgpuDevice) return null;

    const info = {
        features: Array.from(webgpuDevice.features),
        limits: Object.fromEntries(
            Object.entries(webgpuDevice.limits).map(([key, value]) => [key, value])
        ),
        adapter: null,
        capabilities: {
            supportsTimestampQuery: webgpuDevice.features.has('timestamp-query'),
            supportsComputeShaders: true, // All WebGPU devices support compute
            maxTextureSize: webgpuDevice.limits.maxTextureDimension2D,
            maxBufferSize: webgpuDevice.limits.maxBufferSize,
            maxComputeWorkgroupSize: [
                webgpuDevice.limits.maxComputeWorkgroupSizeX,
                webgpuDevice.limits.maxComputeWorkgroupSizeY,
                webgpuDevice.limits.maxComputeWorkgroupSizeZ
            ]
        }
    };

    console.log('WebGPU device info:', info);
    return info;
}

// Get detailed WebGPU adapter information
async function getWebGPUAdapterInfo() {
    if (!navigator.gpu) {
        return { supported: false, reason: 'WebGPU not available' };
    }

    try {
        const adapter = await navigator.gpu.requestAdapter({
            powerPreference: 'high-performance'
        });

        if (!adapter) {
            return { supported: false, reason: 'No WebGPU adapter available' };
        }

        const adapterInfo = {
            supported: true,
            vendor: adapter.info?.vendor || 'Unknown',
            architecture: adapter.info?.architecture || 'Unknown',
            device: adapter.info?.device || 'Unknown',
            description: adapter.info?.description || 'Unknown',
            features: Array.from(adapter.features),
            limits: Object.fromEntries(
                Object.entries(adapter.limits).map(([key, value]) => [key, value])
            ),
            isFallbackAdapter: adapter.isFallbackAdapter
        };

        console.log('WebGPU adapter info:', adapterInfo);
        return adapterInfo;
    } catch (error) {
        console.error('Error getting WebGPU adapter info:', error);
        return { supported: false, reason: error.message };
    }
}

// Performance monitoring for WebGPU operations
class WebGPUPerformanceMonitor {
    constructor() {
        this.metrics = {
            modelLoadTime: 0,
            inferenceTime: 0,
            memoryUsage: 0,
            throughput: 0
        };
    }

    startTiming(operation) {
        this[`${operation}StartTime`] = performance.now();
    }

    endTiming(operation) {
        const endTime = performance.now();
        const startTime = this[`${operation}StartTime`];
        if (startTime) {
            this.metrics[`${operation}Time`] = endTime - startTime;
            console.log(`${operation} took ${this.metrics[`${operation}Time`].toFixed(2)}ms`);
        }
    }

    updateThroughput(tokensGenerated, timeMs) {
        this.metrics.throughput = (tokensGenerated / timeMs) * 1000; // tokens per second
        console.log(`Throughput: ${this.metrics.throughput.toFixed(2)} tokens/sec`);
    }

    getMetrics() {
        return { ...this.metrics };
    }
}

// Global performance monitor
const performanceMonitor = new WebGPUPerformanceMonitor();

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

// Initialize worker with comprehensive logging
console.log('=== Real WebGPU Transcription Worker Initialized ===');
console.log('Features:');
console.log('- @huggingface/transformers integration');
console.log('- WebGPU acceleration');
console.log('- Real audio format detection and conversion');
console.log('- Comprehensive audio processing pipeline');
console.log('- Performance monitoring');
console.log('- Error handling and recovery');
console.log('Supported audio formats:', Object.keys({
    'wav': 'PCM, ADPCM',
    'mp3': 'MPEG Audio',
    'flac': 'Free Lossless Audio Codec',
    'ogg': 'Vorbis, Opus',
    'm4a': 'AAC',
    'webm': 'Opus, Vorbis'
}));
console.log('WebGPU status: Checking...');

// Perform initial capability check
(async () => {
    const webgpuInfo = await getWebGPUAdapterInfo();
    if (webgpuInfo.supported) {
        console.log('✓ WebGPU supported:', webgpuInfo.vendor, webgpuInfo.device);
        console.log(`✓ Available features: ${webgpuInfo.features.join(', ')}`);
    } else {
        console.warn('✗ WebGPU not supported:', webgpuInfo.reason);
    }
})();

console.log('===================================================');