// transformers.js warns for each file served without a Content-Length header; progress is measured against the manifest sizes instead
const NOISE = "Unable to determine content-length";
const warn = console.warn.bind(console);

console.warn = (...args) => {
    if (typeof args[0] === "string" && args[0].startsWith(NOISE)) {
        return;
    }
    warn(...args);
};
