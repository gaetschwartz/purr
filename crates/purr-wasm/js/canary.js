const MAX_TOKENS = 48;

let canaryUrl = null;
let samples = null;

export function configureCanary(url) {
    canaryUrl = url;
    samples = null;
}

async function canarySamples() {
    if (samples === null) {
        const response = await fetch(canaryUrl);
        if (!response.ok) {
            throw new Error(`The GPU check recording could not be loaded (${response.status})`);
        }
        const pcm = new Int16Array(await response.arrayBuffer());
        samples = Float32Array.from(pcm, (sample) => sample / 32768);
    }
    return samples;
}

export async function transcribeCanary(transcriber, englishOnly) {
    const options = { top_k: 0, do_sample: false, max_new_tokens: MAX_TOKENS };
    if (!englishOnly) {
        options.language = "en";
        options.task = "transcribe";
    }
    const output = await transcriber(await canarySamples(), options);
    return Array.isArray(output) ? output[0].text : output.text;
}
