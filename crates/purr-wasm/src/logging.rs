//! Browser console output for `tracing` events.

use tracing::Subscriber;
use tracing_subscriber::Layer;
use tracing_subscriber::registry::LookupSpan;
use tracing_wasm::{ConsoleConfig, WASMLayer, WASMLayerConfigBuilder};

/// A layer that prints events to the browser console. Filtering is left to the subscriber.
#[must_use]
pub fn console_layer<S>() -> impl Layer<S>
where
    S: Subscriber + for<'span> LookupSpan<'span>,
{
    let config = WASMLayerConfigBuilder::new()
        .set_console_config(ConsoleConfig::ReportWithoutConsoleColor)
        .set_report_logs_in_timings(false)
        .build();
    WASMLayer::new(config)
}
