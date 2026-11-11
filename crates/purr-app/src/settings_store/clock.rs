//! Wall-clock time for the "Saved" status.

#[cfg(purr_web)]
pub fn now_millis() -> u64 {
    js_sys::Date::now() as u64
}

#[cfg(not(purr_web))]
pub fn now_millis() -> u64 {
    match std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
        Ok(elapsed) => elapsed.as_millis() as u64,
        Err(error) => {
            tracing::warn!("The system clock is before 1970: {error}");
            0
        }
    }
}
