const HUGGING_FACE_ORIGIN = "https://huggingface.co";

let token = null;

export function useHuggingFaceToken(value) {
    token = typeof value === "string" && value.length > 0 ? value : null;
}

function targetOf(input) {
    try {
        if (input instanceof Request) {
            return new URL(input.url);
        }
        return new URL(String(input), globalThis.location?.href);
    } catch {
        return null;
    }
}

const nativeFetch = globalThis.fetch.bind(globalThis);

// transformers.js offers no way to add headers, and only huggingface.co may get the token: the CDN behind it serves pre-signed URLs
globalThis.fetch = (input, init) => {
    if (token === null || targetOf(input)?.origin !== HUGGING_FACE_ORIGIN) {
        return nativeFetch(input, init);
    }
    const headers = new Headers(init?.headers ?? (input instanceof Request ? input.headers : undefined));
    headers.set("Authorization", `Bearer ${token}`);
    return nativeFetch(input, { ...init, headers });
};
