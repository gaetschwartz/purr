# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "onnxruntime==1.32.0", "scipy", "soundfile", "transformers"]
# ///
"""Checks that converted Whisper ONNX files give the same results as the originals on the CPU.

usage: uv run scripts/validate_onnx_conversion.py REPO ORIGINAL_DIR CONVERTED_DIR ENCODER DECODER CLIP [half-table]

REPO is the Hugging Face repository the tokenizer and feature extractor come from. ENCODER and
DECODER are file names found in both directories; the converted `<name>_data` sits beside them.
Exits non-zero unless the encoder output and every decoder logit are bit-identical.

With `half-table` the converted decoder stores its embedding table as half precision, so only the
decoder is compared, fed the original encoder output: it must pick the same token as the original
on at least 98% of the teacher-forced steps, and no logit may move by more than 0.25.
"""

import json
import sys

import numpy as np
import onnxruntime as ort
import soundfile as sf
from scipy.signal import resample_poly
from transformers import WhisperFeatureExtractor, WhisperTokenizer

MAX_NEW_TOKENS = 60
MIN_TOP1_AGREEMENT = 0.98
MAX_LOGIT_DIFF = 0.25


def load_audio(path):
    samples, rate = sf.read(path, dtype="float32")
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    if rate != 16000:
        samples = resample_poly(samples, 16000, rate).astype(np.float32)
    return samples[: 30 * 16000]


def session(path):
    options = ort.SessionOptions()
    options.log_severity_level = 3
    return ort.InferenceSession(path, options, providers=["CPUExecutionProvider"])


def past_inputs(sess):
    return [i for i in sess.get_inputs() if i.name.startswith("past_key_values")]


def step(sess, ids, hidden, past, use_cache):
    feed = {
        "input_ids": np.array([ids], dtype=np.int64),
        "encoder_hidden_states": hidden,
        "use_cache_branch": np.array([use_cache]),
    }
    feed.update(past)
    outputs = dict(zip([o.name for o in sess.get_outputs()], sess.run(None, feed)))
    next_past = {}
    for i in past_inputs(sess):
        present = outputs[i.name.replace("past_key_values", "present")]
        next_past[i.name] = present if (not use_cache or ".decoder." in i.name) else past[i.name]
    return outputs["logits"][0, -1], next_past


def decode(sess, hidden, prompt, eot, forced=None):
    past = {
        i.name: np.zeros([1, i.shape[1], 0, i.shape[3]], dtype=np.float32)
        for i in past_inputs(sess)
    }
    logits, past = step(sess, prompt, hidden, past, False)
    tokens, all_logits = [], [logits]
    for k in range(MAX_NEW_TOKENS):
        token = int(np.argmax(logits)) if forced is None else forced[k]
        if forced is None and token == eot:
            break
        tokens.append(token)
        if forced is not None and k + 1 >= len(forced):
            break
        logits, past = step(sess, [token], hidden, past, True)
        all_logits.append(logits)
    return tokens, all_logits


def check_half_table(repo, original_dir, converted_dir, encoder, decoder, clip):
    extractor = WhisperFeatureExtractor.from_pretrained(repo)
    tokenizer = WhisperTokenizer.from_pretrained(repo)
    prompt = tokenizer.convert_tokens_to_ids(
        ["<|startoftranscript|>", "<|en|>", "<|transcribe|>", "<|notimestamps|>"]
    )
    eot = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    features = extractor(load_audio(clip), sampling_rate=16000, return_tensors="np").input_features
    hidden = session(f"{original_dir}/{encoder}").run(None, {"input_features": features})[0]
    original = session(f"{original_dir}/{decoder}")
    converted = session(f"{converted_dir}/{decoder}")
    tokens, reference = decode(original, hidden, prompt, eot)
    _, forced = decode(converted, hidden, prompt, eot, forced=tokens + [eot])
    count = min(len(forced), len(reference))
    agreeing = sum(int(np.argmax(forced[i]) == np.argmax(reference[i])) for i in range(count))
    report = {
        "text": tokenizer.decode(tokens, skip_special_tokens=True),
        "top1_agreement": agreeing / count,
        "max_logit_diff": max(float(np.max(np.abs(forced[i] - reference[i]))) for i in range(count)),
        "steps": count,
    }
    print(json.dumps(report, indent=1))
    if report["top1_agreement"] < MIN_TOP1_AGREEMENT or report["max_logit_diff"] > MAX_LOGIT_DIFF:
        sys.exit(1)


def main():
    if len(sys.argv) == 8 and sys.argv[7] == "half-table":
        check_half_table(*sys.argv[1:7])
        return
    repo, original_dir, converted_dir, encoder, decoder, clip = sys.argv[1:7]
    extractor = WhisperFeatureExtractor.from_pretrained(repo)
    tokenizer = WhisperTokenizer.from_pretrained(repo)
    prompt = tokenizer.convert_tokens_to_ids(
        ["<|startoftranscript|>", "<|en|>", "<|transcribe|>", "<|notimestamps|>"]
    )
    eot = tokenizer.convert_tokens_to_ids("<|endoftext|>")

    features = extractor(load_audio(clip), sampling_rate=16000, return_tensors="np").input_features
    hidden = {}
    for label, directory in (("original", original_dir), ("converted", converted_dir)):
        hidden[label] = session(f"{directory}/{encoder}").run(None, {"input_features": features})[0]
    encoder_identical = np.array_equal(hidden["original"], hidden["converted"])

    original = session(f"{original_dir}/{decoder}")
    converted = session(f"{converted_dir}/{decoder}")
    tokens, reference = decode(original, hidden["original"], prompt, eot)
    converted_tokens, _ = decode(converted, hidden["original"], prompt, eot)
    _, forced = decode(converted, hidden["original"], prompt, eot, forced=tokens + [eot])
    count = min(len(forced), len(reference))
    logits_identical = all(np.array_equal(forced[i], reference[i]) for i in range(count))
    report = {
        "text": tokenizer.decode(tokens, skip_special_tokens=True),
        "same_tokens": tokens == converted_tokens,
        "encoder_identical": bool(encoder_identical),
        "logits_identical": bool(logits_identical),
        "max_logit_diff": max(float(np.max(np.abs(forced[i] - reference[i]))) for i in range(count)),
        "steps": count,
    }
    print(json.dumps(report, indent=1))
    if not (report["same_tokens"] and encoder_identical and logits_identical):
        sys.exit(1)


main()
