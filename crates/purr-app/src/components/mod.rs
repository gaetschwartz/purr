//! Reusable UI components for the Purr transcription app
//!
//! This module contains all the reusable components that make up the UI.
//! Components are organized by functionality and follow Dioxus patterns.

mod icons;
pub use icons::*;

mod button;
pub use button::*;

mod card;
pub use card::*;

mod upload_zone;
pub use upload_zone::*;

pub mod model_loading;

pub mod model_picker;

pub mod transcript_panel;

mod transcription_display;
pub use transcription_display::*;

mod progress_bar;
pub use progress_bar::*;

mod form;
pub use form::*;

mod theme_demo;
pub use theme_demo::*;
