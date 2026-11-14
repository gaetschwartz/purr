//! From audio to committed text, whatever the engine.
//!
//! The source is cut into windows ([`policy`], with an optional speech detector), each window is
//! decoded by the engine, its segments are given honest spans ([`segmenter`]), and the stream
//! decides what of them is final ([`commit`], [`tentative`], [`language`]), sending
//! [`events`] as it goes.

pub mod commit;
mod driver;
pub mod events;
pub mod language;
pub mod pause;
pub mod policy;
pub mod result;
mod retry;
pub mod run;
pub mod segmenter;
mod session;
pub mod tentative;
#[cfg(test)]
mod tests;
