/// Milliseconds on a clock that only moves forward, for measuring how long something took.
#[must_use]
pub fn now_ms() -> f64 {
    #[cfg(target_arch = "wasm32")]
    {
        js_sys::Date::now()
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        use std::sync::LazyLock;
        use std::time::Instant;

        static START: LazyLock<Instant> = LazyLock::new(Instant::now);
        START.elapsed().as_secs_f64() * 1000.0
    }
}
