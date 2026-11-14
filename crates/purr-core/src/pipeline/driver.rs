//! Feeds a session from its source and sends what it decodes.

use std::ops::ControlFlow;

use futures::{FutureExt, StreamExt};
use tracing::{debug, warn};

use crate::{AudioChunk, AudioStream, error::Result};

use super::session::{Events, Session};

/// Runs `session` over `input` until the source ends and its last window is decoded, the
/// receiver is dropped, or the transcription is cancelled.
pub async fn drive(mut session: Session, mut input: AudioStream, tx: Events) {
    let mut complete = false;

    let receive =
        |item: Option<Result<AudioChunk>>, session: &mut Session, complete: &mut bool| match item {
            Some(Ok(chunk)) => {
                *complete |= chunk.is_final;
                session.ended = *complete;
                session.push(&chunk.samples);
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
        if !complete {
            if session.next_window(false).is_none() {
                let item = match input.next().now_or_never() {
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
                    None => input.next().await,
                };
                if receive(item, &mut session, &mut complete).is_break() {
                    return;
                }
            }
            while !complete && let Some(item) = input.next().now_or_never() {
                if receive(item, &mut session, &mut complete).is_break() {
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
