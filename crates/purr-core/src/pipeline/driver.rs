//! Feeds a session from its source and sends what it decodes.

use std::{
    ops::ControlFlow,
    sync::{Arc, OnceLock},
};

use futures::{FutureExt, StreamExt};
use tokio::sync::watch;
use tracing::{debug, warn};

use crate::{AudioChunk, AudioStream, audio::DECODED_CHANNEL_CAPACITY, error::Result};

use super::{
    events::EndReason,
    session::{Events, Session},
};

/// What the owner of a transcription asks of its task.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Request {
    Run,
    Stop,
    Cancel,
}

/// Where the task records why its source ended; the first reason stays.
#[derive(Debug, Clone, Default)]
pub struct EndSlot(Arc<OnceLock<EndReason>>);

impl EndSlot {
    pub fn record(&self, reason: EndReason) {
        self.0.get_or_init(|| reason);
    }

    pub fn get(&self) -> Option<EndReason> {
        self.0.get().copied()
    }
}

/// How a run may end before its source does.
pub struct Ending {
    pub request: watch::Receiver<Request>,
    /// The most audio to take, in samples, from a source of unknown length
    pub listen: Option<usize>,
    pub reason: EndSlot,
}

enum Pulled {
    Item(Option<Result<AudioChunk>>),
    Stop,
    Cancelled,
}

async fn pull(input: &mut AudioStream, request: &mut watch::Receiver<Request>) -> Pulled {
    tokio::select! {
        item = input.next() => Pulled::Item(item),
        changed = request.wait_for(|request| *request != Request::Run) => {
            if changed.is_ok_and(|request| *request == Request::Stop) {
                Pulled::Stop
            } else {
                Pulled::Cancelled
            }
        }
    }
}

/// Runs `session` over `input` until the source ends and its last window is decoded, the
/// receiver is dropped, or the transcription is cancelled.
pub async fn drive(session: &mut Session, input: AudioStream, tx: &Events, ending: Ending) {
    let Ending {
        mut request,
        listen,
        reason,
    } = ending;
    let mut input = Some(input);
    let mut complete = false;

    let receive =
        |item: Option<Result<AudioChunk>>, session: &mut Session, complete: &mut bool| match item {
            Some(Ok(mut chunk)) => {
                let offered = chunk.samples.len();
                let kept = listen.map_or(offered, |limit| {
                    limit
                        .saturating_sub(session.received_samples())
                        .min(offered)
                });
                chunk.samples.truncate(kept);
                session.push(&chunk.samples);
                if kept < offered {
                    reason.record(EndReason::DurationReached);
                    *complete = true;
                } else if chunk.is_final {
                    reason.record(EndReason::SourceEnded);
                    *complete = true;
                } else if listen.is_some_and(|limit| session.received_samples() >= limit) {
                    reason.record(EndReason::DurationReached);
                    *complete = true;
                }
                session.ended = *complete;
                ControlFlow::Continue(())
            }
            Some(Err(e)) => {
                warn!("Audio stream error: {}", e);
                if tx.send(Err(e)).is_err() {
                    debug!("Receiver dropped, error not delivered");
                }
                ControlFlow::Break(())
            }
            None => {
                reason.record(EndReason::SourceEnded);
                *complete = true;
                session.ended = true;
                ControlFlow::Continue(())
            }
        };

    while !session.finished {
        if session.cancelled() {
            debug!("Transcription cancelled");
            return;
        }
        if !complete && *request.borrow() == Request::Stop {
            reason.record(EndReason::Stopped);
            if let Some(stream) = input.as_mut() {
                for _ in 0..DECODED_CHANNEL_CAPACITY {
                    let Some(item) = stream.next().now_or_never() else {
                        break;
                    };
                    if receive(item, session, &mut complete).is_break() {
                        return;
                    }
                    if complete {
                        break;
                    }
                }
            }
            complete = true;
            session.ended = true;
        }
        if complete {
            input = None;
        }
        if !complete
            && session.wants_audio()
            && let Some(stream) = input.as_mut()
        {
            if session.next_window(false).is_none() {
                let item = match stream.next().now_or_never() {
                    Some(item) => item,
                    None if session.tentative_due() => {
                        let event = match session.tentative() {
                            Ok(event) => event,
                            Err(e) => {
                                warn!("Failed to decode audio tentatively: {}", e);
                                if tx.send(Err(e)).is_err() {
                                    debug!("Receiver dropped, error not delivered");
                                }
                                return;
                            }
                        };
                        if tx.send(Ok(event)).is_err() {
                            debug!("Receiver dropped, stopping stream processing");
                            return;
                        }
                        continue;
                    }
                    None => match pull(stream, &mut request).await {
                        Pulled::Item(item) => item,
                        Pulled::Stop => continue,
                        Pulled::Cancelled => {
                            debug!("Transcription cancelled while waiting for audio");
                            return;
                        }
                    },
                };
                if receive(item, session, &mut complete).is_break() {
                    return;
                }
            }
            while !complete
                && session.wants_audio()
                && let Some(item) = stream.next().now_or_never()
            {
                if receive(item, session, &mut complete).is_break() {
                    return;
                }
            }
        }

        while let Some(kind) = session.next_window(complete) {
            let events = match session.process(kind) {
                Ok(events) => events,
                Err(e) => {
                    warn!("Failed to process audio window: {}", e);
                    if tx.send(Err(e)).is_err() {
                        debug!("Receiver dropped, error not delivered");
                    }
                    return;
                }
            };
            for event in events {
                if tx.send(Ok(event)).is_err() {
                    debug!("Receiver dropped, stopping stream processing");
                    return;
                }
            }
            if !complete || session.finished {
                break;
            }
        }
    }
}
