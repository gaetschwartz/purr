use std::str::FromStr as _;
use std::time::Duration;

const EXAMPLES: &str = "a number of seconds like `90`, or a duration like `30m` or `1h30m`";

/// Parses `--duration`: plain digits are whole seconds, anything else is a human duration.
pub fn parse_duration_arg(text: &str) -> Result<Duration, String> {
    let duration = match u64::from_str(text) {
        Ok(seconds) => Duration::from_secs(seconds),
        Err(_) => humantime::parse_duration(text)
            .map_err(|error| format!("`{text}` is not a duration ({error}); use {EXAMPLES}"))?,
    };
    if duration.is_zero() {
        return Err("a duration of zero would end every stream before it starts".to_owned());
    }
    Ok(duration)
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn plain_digits_are_seconds_and_suffixes_are_durations() {
        assert_eq!(parse_duration_arg("90"), Ok(Duration::from_secs(90)));
        assert_eq!(parse_duration_arg("90s"), Ok(Duration::from_secs(90)));
        assert_eq!(parse_duration_arg("30m"), Ok(Duration::from_mins(30)));
        assert_eq!(parse_duration_arg("1h30m"), Ok(Duration::from_mins(90)));
    }

    #[test]
    fn zero_and_garbage_are_errors() {
        for text in ["0", "0s", "abc", "", "-5", "1.5"] {
            assert!(parse_duration_arg(text).is_err(), "{text:?}");
        }
    }
}
