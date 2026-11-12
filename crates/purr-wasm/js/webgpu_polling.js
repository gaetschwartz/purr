const DENSE_KICKS = 16;
const SPARSE_KICK_MS = 4;
const GIVE_UP_AFTER_MS = 10000;

let installed = false;

function kickUntilSettled(queue, pending) {
    let settled = false;
    const stop = () => {
        settled = true;
    };
    pending.then(stop, stop);
    const startedAt = performance.now();
    const kick = () => {
        if (settled || performance.now() - startedAt > GIVE_UP_AFTER_MS) {
            return false;
        }
        try {
            queue.submit([]);
        } catch {
            settled = true;
            return false;
        }
        return true;
    };
    for (let delay = 0; delay < DENSE_KICKS; delay += 1) {
        setTimeout(kick, delay);
    }
    const sparse = () => {
        if (kick()) {
            setTimeout(sparse, SPARSE_KICK_MS);
        }
    };
    setTimeout(sparse, DENSE_KICKS);
    return pending;
}

// Firefox settles mapAsync and onSubmittedWorkDone only on a 100 ms device poll, and a submit polls at once: https://bugzilla.mozilla.org/show_bug.cgi?id=1870699
export function keepWebGpuPolled() {
    if (installed || typeof GPUDevice === "undefined") {
        return;
    }
    installed = true;
    const queues = new WeakMap();
    const createBuffer = GPUDevice.prototype.createBuffer;
    GPUDevice.prototype.createBuffer = function (descriptor) {
        const buffer = createBuffer.call(this, descriptor);
        queues.set(buffer, this.queue);
        return buffer;
    };
    const mapAsync = GPUBuffer.prototype.mapAsync;
    GPUBuffer.prototype.mapAsync = function (...args) {
        const pending = mapAsync.apply(this, args);
        const queue = queues.get(this);
        return queue === undefined ? pending : kickUntilSettled(queue, pending);
    };
    const onSubmittedWorkDone = GPUQueue.prototype.onSubmittedWorkDone;
    GPUQueue.prototype.onSubmittedWorkDone = function () {
        return kickUntilSettled(this, onSubmittedWorkDone.call(this));
    };
}
