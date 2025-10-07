//! Simplified use_theme hook that re-exports the theme system
//!
//! This hook provides a convenient re-export of the theme system
//! for easy access throughout the application.

// Re-export the theme system's use_theme hook and types
pub use crate::theme::{use_theme, Theme, UseThemeContext};

// Re-export for backwards compatibility
pub type UseThemeResult = UseThemeContext;
