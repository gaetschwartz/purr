import { Tensor } from "@huggingface/transformers";

const WINDOW_SAMPLES = 30 * 16000;

// transformers.js assumes English when no language is given, so the language is found the way Whisper does it: one decoder step from <|startoftranscript|>, then the most likely language token.
export async function detectLanguage(transcriber, audio) {
    const model = transcriber.model;
    const { input_features: inputFeatures } = await transcriber.processor(
        audio.subarray(0, WINDOW_SAMPLES),
    );
    const startToken = BigInt(model.generation_config.decoder_start_token_id);
    const decoderInput = new Tensor("int64", BigInt64Array.from([startToken]), [1, 1]);
    const { logits } = await model({
        input_features: inputFeatures,
        decoder_input_ids: decoderInput,
    });
    const scores = logits.location === "cpu" ? logits.data : await logits.ort_tensor.getData();

    let best = { token: null, score: -Infinity };
    for (const [token, id] of Object.entries(model.generation_config.lang_to_id)) {
        if (scores[id] > best.score) {
            best = { token, score: scores[id] };
        }
    }
    if (best.token === null) {
        throw new Error("The model has no language tokens");
    }
    return best.token.slice(2, -2);
}
