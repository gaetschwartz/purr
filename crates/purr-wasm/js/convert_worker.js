// Runs the ONNX converter WebAssembly module over bytes that the page streams in through a
// shared ring buffer. The module reads synchronously, so this worker blocks in Atomics.wait
// between chunks; its outputs leave as transferred messages.

const HEADER_BYTES = 64;
const Slot = Object.freeze({ Head: 0, Tail: 1, Epoch: 2, State: 3, Acked: 4 });
const InputState = Object.freeze({ Open: 0, Ended: 1, Aborted: 2 });
const MAX_UNACKED_BYTES = 8 * 1024 * 1024;
const WAIT_SLICE_MS = 1000;

const encoder = new TextEncoder();
const decoder = new TextDecoder();

function startConversion({ wasm_url: wasmUrl, data_location: dataLocation, shared }) {
    const control = new Int32Array(shared, 0, HEADER_BYTES / Int32Array.BYTES_PER_ELEMENT);
    const ring = new Uint8Array(shared, HEADER_BYTES);
    const mask = ring.length - 1;
    let memory = null;
    let sent = 0;
    let failure = null;

    function readInput(pointer, length) {
        for (;;) {
            const epoch = Atomics.load(control, Slot.Epoch);
            const head = Atomics.load(control, Slot.Head);
            const tail = Atomics.load(control, Slot.Tail);
            const available = (head - tail) | 0;
            if (available > 0) {
                const index = tail & mask;
                const count = Math.min(length, available, ring.length - index);
                new Uint8Array(memory.buffer, pointer, count).set(ring.subarray(index, index + count));
                Atomics.store(control, Slot.Tail, (tail + count) | 0);
                return count;
            }
            switch (Atomics.load(control, Slot.State)) {
                case InputState.Ended:
                    return 0;
                case InputState.Aborted:
                    return -1;
                default:
                    Atomics.wait(control, Slot.Epoch, epoch, WAIT_SLICE_MS);
            }
        }
    }

    function writeOutput(sink, pointer, length) {
        for (;;) {
            const acked = Atomics.load(control, Slot.Acked);
            if (sent - acked <= MAX_UNACKED_BYTES) {
                break;
            }
            Atomics.wait(control, Slot.Acked, acked, WAIT_SLICE_MS);
        }
        const bytes = new Uint8Array(memory.buffer, pointer, length).slice();
        postMessage({ type: "Chunk", sink, bytes: bytes.buffer }, [bytes.buffer]);
        sent += length;
        return 0;
    }

    function writeLocation(pointer, capacity) {
        const { written } = encoder.encodeInto(dataLocation, new Uint8Array(memory.buffer, pointer, capacity));
        return written;
    }

    function reportError(pointer, length) {
        failure = decoder.decode(new Uint8Array(memory.buffer, pointer, length));
    }

    return fetch(wasmUrl)
        .then((response) => {
            if (!response.ok) {
                throw new Error(`Fetching the converter failed with status ${response.status}`);
            }
            return response.arrayBuffer();
        })
        .then((bytes) =>
            WebAssembly.instantiate(bytes, {
                purr: {
                    purr_read: readInput,
                    purr_write: writeOutput,
                    purr_location: writeLocation,
                    purr_report_error: reportError,
                },
            }),
        )
        .then(({ instance }) => {
            memory = instance.exports.memory;
            const status = instance.exports.purr_convert();
            return status === 0 ? null : (failure ?? `The converter failed with status ${status}`);
        });
}

self.addEventListener(
    "message",
    (event) => {
        startConversion(event.data)
            .then((error) => postMessage({ type: "Done", error }))
            .catch((error) => postMessage({ type: "Done", error: String(error?.message ?? error) }));
    },
    { once: true },
);
