//! A short-lived Web Worker that runs the ONNX converter module.
//!
//! The module reads its input synchronously, so the page feeds it through a ring buffer in a
//! `SharedArrayBuffer`; the module's outputs come back as messages, and the page acknowledges
//! them so the worker never runs far ahead of whoever stores them. The worker is terminated when
//! the handle is dropped, which returns the converter's memory to the browser.

use crate::error::{WebError, WebResult};
use gloo_timers::future::TimeoutFuture;
use js_sys::{Atomics, Int32Array, Object, Reflect, SharedArrayBuffer, Uint8Array};
use std::cell::Cell;
use std::rc::Rc;
use tokio::sync::mpsc;
use wasm_bindgen::closure::Closure;
use wasm_bindgen::{JsCast as _, JsValue};
use web_sys::{ErrorEvent, MessageEvent, Worker, WorkerOptions, WorkerType};

const HEADER_BYTES: u32 = 64;
const RING_BYTES: u32 = 4 * 1024 * 1024;
const RING_POLL_MS: u32 = 2;

const SLOT_HEAD: u32 = 0;
const SLOT_TAIL: u32 = 1;
const SLOT_EPOCH: u32 = 2;
const SLOT_STATE: u32 = 3;
const SLOT_ACKED: u32 = 4;
const STATE_ENDED: i32 = 1;
const STATE_ABORTED: i32 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sink {
    Data,
    Graph,
}

pub enum ConverterEvent {
    Output { sink: Sink, bytes: Uint8Array },
    Finished,
    Failed(String),
}

struct Handle {
    worker: Worker,
    stopped: Cell<bool>,
    _on_message: Closure<dyn FnMut(MessageEvent)>,
    _on_error: Closure<dyn FnMut(ErrorEvent)>,
}

impl Drop for Handle {
    fn drop(&mut self) {
        self.worker.set_onmessage(None);
        self.worker.set_onerror(None);
        self.worker.terminate();
    }
}

fn js_error(api: &str, error: JsValue) -> WebError {
    WebError::web_api(api, error)
}

fn failure(message: impl Into<String>) -> WebError {
    WebError::conversion(message)
}

/// Writing end: the bytes of the model being converted.
pub struct ConverterInput {
    control: Int32Array,
    ring: Uint8Array,
    head: u32,
    handle: Rc<Handle>,
}

/// Reading end: what the converter produces.
pub struct ConverterOutput {
    control: Int32Array,
    events: mpsc::UnboundedReceiver<ConverterEvent>,
    handle: Rc<Handle>,
}

/// Starts a converter worker that names its weights file `data_location` inside the graph.
pub fn start(
    worker_url: &str,
    wasm_url: &str,
    data_location: &str,
) -> WebResult<(ConverterInput, ConverterOutput)> {
    let shared = SharedArrayBuffer::new(HEADER_BYTES + RING_BYTES);
    let control = Int32Array::new_with_byte_offset_and_length(&shared, 0, HEADER_BYTES / 4);
    let ring = Uint8Array::new_with_byte_offset_and_length(&shared, HEADER_BYTES, RING_BYTES);

    let options = WorkerOptions::new();
    options.set_type(WorkerType::Module);
    let worker =
        Worker::new_with_options(worker_url, &options).map_err(|e| js_error("Worker.new", e))?;

    let (sender, events) = mpsc::unbounded_channel();
    let on_message = {
        let sender = sender.clone();
        Closure::<dyn FnMut(MessageEvent)>::new(move |event: MessageEvent| {
            if let Some(event) = parse_message(&event.data())
                && sender.send(event).is_err()
            {
                tracing::debug!("Nobody follows the converter any more");
            }
        })
    };
    let on_error = Closure::<dyn FnMut(ErrorEvent)>::new(move |event: ErrorEvent| {
        let message =
            ConverterEvent::Failed(format!("The converter worker stopped: {}", event.message()));
        if sender.send(message).is_err() {
            tracing::debug!("Nobody follows the converter any more");
        }
    });
    worker.set_onmessage(Some(on_message.as_ref().unchecked_ref()));
    worker.set_onerror(Some(on_error.as_ref().unchecked_ref()));
    let handle = Rc::new(Handle {
        worker,
        stopped: Cell::new(false),
        _on_message: on_message,
        _on_error: on_error,
    });

    let start = Object::new();
    for (key, value) in [
        ("wasm_url", JsValue::from_str(wasm_url)),
        ("data_location", JsValue::from_str(data_location)),
        ("shared", shared.into()),
    ] {
        Reflect::set(&start, &JsValue::from_str(key), &value)
            .map_err(|e| js_error("Reflect.set", e))?;
    }
    handle
        .worker
        .post_message(&start)
        .map_err(|e| js_error("Worker.postMessage", e))?;

    Ok((
        ConverterInput {
            control: control.clone(),
            ring,
            head: 0,
            handle: Rc::clone(&handle),
        },
        ConverterOutput {
            control,
            events,
            handle,
        },
    ))
}

fn parse_message(data: &JsValue) -> Option<ConverterEvent> {
    let field = |name: &str| Reflect::get(data, &JsValue::from_str(name)).ok();
    match field("type")?.as_string()?.as_str() {
        "Chunk" => {
            let sink = match field("sink")?.as_f64()? as u32 {
                0 => Sink::Data,
                _ => Sink::Graph,
            };
            let bytes = Uint8Array::new(&field("bytes")?);
            Some(ConverterEvent::Output { sink, bytes })
        }
        "Done" => Some(
            field("error")?
                .as_string()
                .map_or(ConverterEvent::Finished, ConverterEvent::Failed),
        ),
        _ => None,
    }
}

impl ConverterInput {
    fn wake_worker(&self) -> WebResult<()> {
        Atomics::add(&self.control, SLOT_EPOCH, 1).map_err(|e| js_error("Atomics.add", e))?;
        Atomics::notify(&self.control, SLOT_EPOCH).map_err(|e| js_error("Atomics.notify", e))?;
        Ok(())
    }

    /// Feeds `bytes` to the converter, waiting while the ring is full.
    pub async fn write(&mut self, bytes: &[u8]) -> WebResult<()> {
        let mut rest = bytes;
        while !rest.is_empty() {
            if self.handle.stopped.get() {
                return Err(failure(
                    "The converter stopped before reading all of the file",
                ));
            }
            let tail = Atomics::load(&self.control, SLOT_TAIL)
                .map_err(|e| js_error("Atomics.load", e))? as u32;
            let free = RING_BYTES - self.head.wrapping_sub(tail);
            if free == 0 {
                TimeoutFuture::new(RING_POLL_MS).await;
                continue;
            }
            let index = self.head & (RING_BYTES - 1);
            let count = (rest.len() as u32).min(free).min(RING_BYTES - index);
            let Some((now, later)) = rest.split_at_checked(count as usize) else {
                return Err(failure("The converter input was split out of range"));
            };
            self.ring.subarray(index, index + count).copy_from(now);
            self.head = self.head.wrapping_add(count);
            Atomics::store(&self.control, SLOT_HEAD, self.head.cast_signed())
                .map_err(|e| js_error("Atomics.store", e))?;
            self.wake_worker()?;
            rest = later;
        }
        Ok(())
    }

    fn finish_with(&self, state: i32) -> WebResult<()> {
        Atomics::store(&self.control, SLOT_STATE, state)
            .map_err(|e| js_error("Atomics.store", e))?;
        self.wake_worker()
    }

    /// Tells the converter the file is complete.
    pub fn end(&self) -> WebResult<()> {
        self.finish_with(STATE_ENDED)
    }

    /// Makes the converter fail instead of waiting for more.
    pub fn abort(&self) -> WebResult<()> {
        self.finish_with(STATE_ABORTED)
    }
}

impl ConverterOutput {
    /// The next event, or `None` once the worker is gone. Feeding stops after the last one.
    pub async fn next(&mut self) -> Option<ConverterEvent> {
        let event = self.events.recv().await;
        if matches!(
            event,
            None | Some(ConverterEvent::Finished | ConverterEvent::Failed(_))
        ) {
            self.handle.stopped.set(true);
        }
        event
    }

    /// Makes the feeding side give up instead of waiting for a converter that is no longer read.
    pub fn stop(&self) {
        self.handle.stopped.set(true);
    }

    /// Lets the worker send `bytes` more.
    pub fn acknowledge(&self, bytes: u32) -> WebResult<()> {
        Atomics::add(&self.control, SLOT_ACKED, bytes.cast_signed())
            .map_err(|e| js_error("Atomics.add", e))?;
        Atomics::notify(&self.control, SLOT_ACKED).map_err(|e| js_error("Atomics.notify", e))?;
        Ok(())
    }
}
