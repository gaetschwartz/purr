//! Custom hooks for the Purr UI crate
//!
//! This module provides reusable hooks that manage application state
//! and provide convenient interfaces for common UI patterns.

pub mod use_models;
pub mod use_theme;

pub use use_theme::{UseThemeResult, use_theme};
