//! Owns the transcription Web Worker and routes its events to the requests that caused them.
//!
//! The `Worker` and its callbacks are not `Send`, so they live in a task on the local executor.
//! [`WorkerClient`] only holds a channel into that task and is freely shareable.

use crate::error::{WebError, WebResult, WorkerError};
use crate::protocol::{
    FailureKind, RequestId, RequestKind, WorkerEvent, WorkerRequest, WorkerSetup,
};
use futures::Stream;
use js_sys::{Array, Float32Array, Object, Reflect};
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::task::{Context, Poll};
use tokio::sync::mpsc;
use wasm_bindgen::closure::Closure;
use wasm_bindgen::{JsCast as _, JsValue};
use wasm_bindgen_futures::spawn_local;
use web_sys::{ErrorEvent, MessageEvent, Worker, WorkerOptions, WorkerType};

enum WorkerSignal {
    Event(WorkerEvent),
    Malformed(String),
    Crashed(String),
}

struct Submission {
    request: WorkerRequest,
    audio: Option<Vec<f32>>,
    events: mpsc::UnboundedSender<WorkerEvent>,
}

enum Input {
    Submit(Submission),
    Cancel(RequestId),
    FromWorker(WorkerSignal),
}

/// Handle to the worker. Cheap to share between tasks.
pub struct WorkerClient {
    input: mpsc::UnboundedSender<Input>,
    next_id: AtomicU64,
}

impl WorkerClient {
    /// Starts the router task. The worker itself is created when the first request arrives.
    #[must_use]
    pub fn spawn(worker_url: String) -> Self {
        let (input, receiver) = mpsc::unbounded_channel();
        let router = Router {
            worker_url,
            input: input.clone(),
            worker: None,
            pending: HashMap::new(),
        };
        spawn_local(router.run(receiver));
        Self {
            input,
            next_id: AtomicU64::new(1),
        }
    }

    pub fn next_request_id(&self) -> RequestId {
        RequestId::new(self.next_id.fetch_add(1, Ordering::Relaxed))
    }

    /// Sends a request to the worker. Dropping the returned stream before it ends cancels the
    /// request by restarting the worker, which is the only way to interrupt a running model.
    pub fn submit(
        &self,
        request: WorkerRequest,
        audio: Option<Vec<f32>>,
    ) -> WebResult<RequestEvents> {
        let id = request.request_id();
        let kind = request.kind();
        let (events, receiver) = mpsc::unbounded_channel();
        self.input
            .send(Input::Submit(Submission {
                request,
                audio,
                events,
            }))
            .map_err(|_| WebError::Worker(WorkerError::Closed))?;
        Ok(RequestEvents {
            id,
            kind,
            receiver,
            input: self.input.clone(),
            finished: false,
        })
    }
}

/// Events of one request, ending with its terminal event.
pub struct RequestEvents {
    id: RequestId,
    kind: RequestKind,
    receiver: mpsc::UnboundedReceiver<WorkerEvent>,
    input: mpsc::UnboundedSender<Input>,
    finished: bool,
}

impl Stream for RequestEvents {
    type Item = WorkerEvent;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let polled = self.receiver.poll_recv(cx);
        if let Poll::Ready(Some(event)) = &polled
            && event.ends(self.kind)
        {
            self.finished = true;
        }
        polled
    }
}

impl Drop for RequestEvents {
    fn drop(&mut self) {
        if !self.finished && self.input.send(Input::Cancel(self.id)).is_err() {
            tracing::warn!(
                "The worker router is gone; request {:?} was not cancelled",
                self.id
            );
        }
    }
}

struct LiveWorker {
    worker: Worker,
    _on_message: Closure<dyn FnMut(MessageEvent)>,
    _on_error: Closure<dyn FnMut(ErrorEvent)>,
    _on_message_error: Closure<dyn FnMut(MessageEvent)>,
}

impl LiveWorker {
    fn start(url: &str, input: &mpsc::UnboundedSender<Input>) -> Result<Self, String> {
        let options = WorkerOptions::new();
        options.set_type(WorkerType::Module);
        let worker = Worker::new_with_options(url, &options).map_err(|e| {
            format!(
                "could not be started ({})",
                crate::error::format_js_error(e)
            )
        })?;
        let (ort_module_url, ort_wasm_url) = crate::ort_urls();
        let setup = serde_json::to_string(&WorkerSetup::Configure {
            ort_module_url,
            ort_wasm_url,
        })
        .map_err(|e| e.to_string())?;
        let message = Object::new();
        Reflect::set(
            &message,
            &JsValue::from_str("request"),
            &JsValue::from_str(&setup),
        )
        .map_err(crate::error::format_js_error)?;
        worker
            .post_message(&message)
            .map_err(crate::error::format_js_error)?;

        let sender = input.clone();
        let on_message =Closure::<dyn FnMut(MessageEvent)>::new(move |event: MessageEvent| {
            let signal = match event.data().as_string() {
                Some(text) => match serde_json::from_str::<WorkerEvent>(&text) {
                    Ok(parsed) => WorkerSignal::Event(parsed),
                    Err(e) => WorkerSignal::Malformed(format!("{e}: {text}")),
                },
                None => WorkerSignal::Malformed("the message was not text".to_owned()),
            };
            if sender.send(Input::FromWorker(signal)).is_err() {
                tracing::warn!("A worker message arrived after the router stopped");
            }
        });
        worker.set_onmessage(Some(on_message.as_ref().unchecked_ref()));

        let sender = input.clone();
        let script = url.to_owned();
        let on_error = Closure::<dyn FnMut(ErrorEvent)>::new(move |event: ErrorEvent| {
            let message = event.message();
            let detail = if message.is_empty() {
                format!("the worker script {script} failed to load or run")
            } else {
                format!("{message} ({}:{})", event.filename(), event.lineno())
            };
            if sender
                .send(Input::FromWorker(WorkerSignal::Crashed(detail)))
                .is_err()
            {
                tracing::warn!("A worker error arrived after the router stopped");
            }
        });
        worker.set_onerror(Some(on_error.as_ref().unchecked_ref()));

        let sender = input.clone();
        let on_message_error =
            Closure::<dyn FnMut(MessageEvent)>::new(move |_event: MessageEvent| {
                if sender
                    .send(Input::FromWorker(WorkerSignal::Malformed(
                        "the browser could not deserialize a worker message".to_owned(),
                    )))
                    .is_err()
                {
                    tracing::warn!("A worker message error arrived after the router stopped");
                }
            });
        worker.set_onmessageerror(Some(on_message_error.as_ref().unchecked_ref()));

        Ok(Self {
            worker,
            _on_message: on_message,
            _on_error: on_error,
            _on_message_error: on_message_error,
        })
    }

    fn post(&self, request: &WorkerRequest, audio: Option<Vec<f32>>) -> Result<(), String> {
        let json = serde_json::to_string(request).map_err(|e| e.to_string())?;
        let message = Object::new();
        let describe = |e: JsValue| crate::error::format_js_error(e);
        Reflect::set(
            &message,
            &JsValue::from_str("request"),
            &JsValue::from_str(&json),
        )
        .map_err(describe)?;
        let transfer = Array::new();
        if let Some(samples) = audio {
            let array = Float32Array::from(samples.as_slice());
            transfer.push(&array.buffer());
            Reflect::set(&message, &JsValue::from_str("audio"), &array).map_err(describe)?;
        }
        self.worker
            .post_message_with_transfer(&message, &transfer)
            .map_err(describe)
    }

    fn terminate(self) {
        self.worker.set_onmessage(None);
        self.worker.set_onerror(None);
        self.worker.set_onmessageerror(None);
        self.worker.terminate();
    }
}

struct Pending {
    events: mpsc::UnboundedSender<WorkerEvent>,
    kind: RequestKind,
}

struct Router {
    worker_url: String,
    input: mpsc::UnboundedSender<Input>,
    worker: Option<LiveWorker>,
    pending: HashMap<RequestId, Pending>,
}

impl Router {
    async fn run(mut self, mut receiver: mpsc::UnboundedReceiver<Input>) {
        while let Some(input) = receiver.recv().await {
            match input {
                Input::Submit(submission) => self.submit(submission),
                Input::Cancel(id) => self.cancel(id),
                Input::FromWorker(WorkerSignal::Event(event)) => self.deliver(event),
                Input::FromWorker(WorkerSignal::Malformed(message)) => {
                    tracing::error!("Malformed worker message: {message}");
                    self.restart(FailureKind::MalformedEvent, &format!(
                        "The transcription worker sent a message that could not be understood: {message}"
                    ));
                }
                Input::FromWorker(WorkerSignal::Crashed(message)) => {
                    tracing::error!("Worker crashed: {message}");
                    self.restart(
                        FailureKind::WorkerCrashed,
                        &format!("The transcription worker stopped unexpectedly: {message}"),
                    );
                }
            }
        }
    }

    fn submit(&mut self, submission: Submission) {
        let Submission {
            request,
            audio,
            events,
        } = submission;
        let id = request.request_id();
        let fail = |events: &mpsc::UnboundedSender<WorkerEvent>, message: String| {
            if events
                .send(WorkerEvent::Failed {
                    request_id: Some(id),
                    kind: FailureKind::WorkerCrashed,
                    message,
                })
                .is_err()
            {
                tracing::warn!("Request {id:?} was dropped before its failure could be reported");
            }
        };

        if self.worker.is_none() {
            match LiveWorker::start(&self.worker_url, &self.input) {
                Ok(worker) => self.worker = Some(worker),
                Err(reason) => {
                    fail(
                        &events,
                        format!(
                            "The transcription worker {reason}; script: {}",
                            self.worker_url
                        ),
                    );
                    return;
                }
            }
        }
        let Some(worker) = &self.worker else {
            return;
        };
        if let Err(reason) = worker.post(&request, audio) {
            fail(
                &events,
                format!("The request could not be sent to the transcription worker: {reason}"),
            );
            return;
        }
        self.pending.insert(
            id,
            Pending {
                events,
                kind: request.kind(),
            },
        );
    }

    fn cancel(&mut self, id: RequestId) {
        if self.pending.remove(&id).is_some() {
            tracing::info!("Cancelling request {id:?} by restarting the worker");
            self.restart(
                FailureKind::WorkerRestarted,
                "The transcription worker was restarted to cancel another request",
            );
        }
    }

    fn deliver(&mut self, event: WorkerEvent) {
        let Some(id) = event.request_id() else {
            if let WorkerEvent::Failed { kind, message, .. } = event {
                self.fail_all(kind, &message);
            }
            return;
        };
        let Some(pending) = self.pending.get(&id) else {
            return;
        };
        let ends = event.ends(pending.kind);
        if pending.events.send(event).is_err() {
            tracing::warn!("Request {id:?} stopped listening");
        }
        if ends {
            self.pending.remove(&id);
        }
    }

    fn fail_all(&mut self, kind: FailureKind, message: &str) {
        for (id, pending) in self.pending.drain() {
            if pending
                .events
                .send(WorkerEvent::Failed {
                    request_id: Some(id),
                    kind,
                    message: message.to_owned(),
                })
                .is_err()
            {
                tracing::warn!("Request {id:?} stopped listening before its failure arrived");
            }
        }
    }

    fn restart(&mut self, kind: FailureKind, message: &str) {
        if let Some(worker) = self.worker.take() {
            worker.terminate();
        }
        self.fail_all(kind, message);
    }
}
