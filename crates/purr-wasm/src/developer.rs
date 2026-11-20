//! The page's own developer readings and the assembly of the panel snapshot.

use crate::protocol::{Provider, WorkerSnapshot};
use purr_common::developer::{
    snapshot::{
        AdapterReading, BrowserSnapshot, ExecutionProvider, JsHeapReading, Peaked, WorkerReading,
    },
    webgpu::feature_rows,
};
use std::cell::Cell;
use wasm_bindgen::JsValue;

thread_local! {
    static COLLECTING: Cell<bool> = const { Cell::new(false) };
    static CONVERTER_HEAP: Cell<Option<Peaked>> = const { Cell::new(None) };
    static PAGE_JS_PEAK: Cell<u64> = const { Cell::new(0) };
}

pub fn set_collecting(enabled: bool) {
    COLLECTING.with(|collecting| collecting.set(enabled));
}

/// Records the converter worker's WebAssembly heap while developer mode is on.
pub fn converter_heap_seen(bytes: u64) {
    if !COLLECTING.with(Cell::get) {
        return;
    }
    CONVERTER_HEAP.with(|heap| {
        let peak = heap.get().map_or(bytes, |seen| seen.peak.max(bytes));
        heap.set(Some(Peaked {
            current: bytes,
            peak,
        }));
    });
}

fn reset_peaks(page_used: Option<u64>) {
    CONVERTER_HEAP.with(|heap| {
        heap.set(heap.get().map(|seen| Peaked {
            current: seen.current,
            peak: seen.current,
        }));
    });
    PAGE_JS_PEAK.with(|peak| peak.set(page_used.unwrap_or_default()));
}

fn read_number(target: &JsValue, key: &str) -> Option<u64> {
    js_sys::Reflect::get(target, &JsValue::from_str(key))
        .ok()
        .and_then(|value| value.as_f64())
        .map(|number| number as u64)
}

/// `performance.memory` of this page, where the browser has it.
fn page_js_heap() -> Option<(u64, u64)> {
    let performance = web_sys::window()?.performance()?;
    let memory = js_sys::Reflect::get(&performance, &JsValue::from_str("memory")).ok()?;
    if memory.is_undefined() {
        return None;
    }
    Some((
        read_number(&memory, "usedJSHeapSize")?,
        read_number(&memory, "jsHeapSizeLimit")?,
    ))
}

const fn provider(provider: Provider) -> ExecutionProvider {
    match provider {
        Provider::WebGpu => ExecutionProvider::WebGpu,
        Provider::Cpu => ExecutionProvider::Cpu,
    }
}

/// The panel snapshot from the worker's reading and what the page knows itself.
pub fn browser_snapshot(worker: WorkerSnapshot, reset: bool) -> BrowserSnapshot {
    let page = page_js_heap();
    if reset {
        reset_peaks(page.map(|(used, _)| used));
    }
    let page_js_heap = page.map(|(used, limit)| {
        let peak = PAGE_JS_PEAK.with(|peak| {
            let higher = peak.get().max(used);
            peak.set(higher);
            higher
        });
        JsHeapReading {
            used: Peaked {
                current: used,
                peak,
            },
            limit,
        }
    });
    BrowserSnapshot {
        collecting: worker.hooks,
        worker: WorkerReading {
            wasm_heap: worker.wasm_heap,
            gpu_buffers: worker.gpu_buffers,
            js_heap: worker.js_heap,
            provider: worker.provider.map(provider),
            model: worker.model,
            adapter: worker.adapter.map(|adapter| AdapterReading {
                info: adapter.info,
                features: feature_rows(&adapter.features),
                limits: adapter.limits,
            }),
        },
        converter_heap: CONVERTER_HEAP.with(Cell::get),
        page_js_heap,
    }
}
