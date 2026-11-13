const MAX_TOKENS = 48;

let recordingUrl = null;
let samples = null;

export function configureWarmUp(url) {
    recordingUrl = url;
    samples = null;
}

async function recordingSamples() {
    if (samples === null) {
        const response = await fetch(recordingUrl);
        if (!response.ok) {
            throw new Error(`The warm-up recording could not be loaded (${response.status})`);
        }
        const pcm = new Int16Array(await response.arrayBuffer());
        samples = Float32Array.from(pcm, (sample) => sample / 32768);
    }
    return samples;
}

const warmed = new WeakSet();

export async function warmUp(transcriber, englishOnly) {
    if (warmed.has(transcriber)) {
        return;
    }
    const options = { top_k: 0, do_sample: false, max_new_tokens: MAX_TOKENS };
    if (!englishOnly) {
        options.language = "en";
        options.task = "transcribe";
    }
    await transcriber(await recordingSamples(), options);
    warmed.add(transcriber);
}
