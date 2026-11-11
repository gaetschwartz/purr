//! Dates as a person reads them: "Today, 14:02", "Yesterday, 09:15", "Nov 7, 17:48".

const MONTHS: [&str; 12] = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];

/// A moment on the local calendar and clock.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Moment {
    pub year: i32,
    /// 1 to 12.
    pub month: u32,
    /// 1 to 31.
    pub day: u32,
    pub hour: u32,
    pub minute: u32,
}

impl Moment {
    /// Days since 1970-01-01 of the calendar day (proleptic Gregorian).
    fn day_number(self) -> i64 {
        let year = i64::from(self.year) - i64::from(self.month <= 2);
        let era = year.div_euclid(400);
        let year_of_era = year.rem_euclid(400);
        let shifted_month = i64::from((self.month + 9) % 12);
        let day_of_year = (153 * shifted_month + 2) / 5 + i64::from(self.day) - 1;
        let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
        era * 146_097 + day_of_era - 719_468
    }

    fn month_name(self) -> &'static str {
        usize::try_from(self.month)
            .ok()
            .and_then(|month| month.checked_sub(1))
            .and_then(|index| MONTHS.get(index))
            .copied()
            .unwrap_or("")
    }
}

/// How `then` reads relative to `now`: the day word or the date, then the time.
#[must_use]
pub fn when(then: Moment, now: Moment) -> String {
    let time = format!("{:02}:{:02}", then.hour, then.minute);
    let day = match now.day_number() - then.day_number() {
        0 => "Today".to_owned(),
        1 => "Yesterday".to_owned(),
        _ if then.year == now.year => format!("{} {}", then.month_name(), then.day),
        _ => format!("{} {}, {}", then.month_name(), then.day, then.year),
    };
    format!("{day}, {time}")
}

/// The local calendar moment of `unix_ms`, when the platform can say.
#[cfg(purr_web)]
#[must_use]
pub fn local_moment(unix_ms: i64) -> Option<Moment> {
    let date = js_sys::Date::new(&wasm_bindgen::JsValue::from_f64(unix_ms as f64));
    Some(Moment {
        year: i32::try_from(date.get_full_year()).ok()?,
        month: date.get_month() + 1,
        day: date.get_date(),
        hour: date.get_hours(),
        minute: date.get_minutes(),
    })
}

/// The local calendar moment of `unix_ms`, when the platform can say.
#[cfg(all(feature = "desktop", not(purr_web)))]
#[must_use]
pub fn local_moment(unix_ms: i64) -> Option<Moment> {
    use chrono::{Datelike as _, Local, TimeZone as _, Timelike as _};

    let local = Local.timestamp_millis_opt(unix_ms).single()?;
    Some(Moment {
        year: local.year(),
        month: local.month(),
        day: local.day(),
        hour: local.hour(),
        minute: local.minute(),
    })
}

/// The local calendar moment of `unix_ms`, when the platform can say.
#[cfg(not(any(purr_web, feature = "desktop")))]
#[must_use]
pub fn local_moment(_unix_ms: i64) -> Option<Moment> {
    None
}

/// `unix_ms` read relative to now; a plain note when the clock cannot be read.
#[must_use]
pub fn when_text(unix_ms: i64, now_unix_ms: i64) -> String {
    match (local_moment(unix_ms), local_moment(now_unix_ms)) {
        (Some(then), Some(now)) => when(then, now),
        (None, _) | (_, None) => "Date unknown".to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const fn at(year: i32, month: u32, day: u32, hour: u32, minute: u32) -> Moment {
        Moment {
            year,
            month,
            day,
            hour,
            minute,
        }
    }

    #[test]
    fn the_same_day_reads_today() {
        assert_eq!(
            when(at(2026, 11, 11, 7, 5), at(2026, 11, 11, 23, 59)),
            "Today, 07:05"
        );
    }

    #[test]
    fn the_day_before_reads_yesterday_even_across_a_month_or_year() {
        assert_eq!(
            when(at(2026, 11, 10, 9, 15), at(2026, 11, 11, 0, 1)),
            "Yesterday, 09:15"
        );
        assert_eq!(
            when(at(2026, 10, 31, 22, 0), at(2026, 11, 1, 8, 0)),
            "Yesterday, 22:00"
        );
        assert_eq!(
            when(at(2025, 12, 31, 23, 30), at(2026, 1, 1, 0, 10)),
            "Yesterday, 23:30"
        );
        assert_eq!(
            when(at(2024, 2, 29, 12, 0), at(2024, 3, 1, 12, 0)),
            "Yesterday, 12:00"
        );
    }

    #[test]
    fn older_days_read_as_dates() {
        assert_eq!(
            when(at(2026, 11, 7, 17, 48), at(2026, 11, 11, 8, 0)),
            "Nov 7, 17:48"
        );
        assert_eq!(
            when(at(2025, 12, 24, 8, 0), at(2026, 11, 11, 8, 0)),
            "Dec 24, 2025, 08:00"
        );
    }

    #[test]
    fn a_time_in_the_future_still_reads_as_a_date() {
        assert_eq!(
            when(at(2026, 11, 12, 8, 0), at(2026, 11, 11, 8, 0)),
            "Nov 12, 08:00"
        );
    }

    #[test]
    fn day_numbers_count_from_the_epoch() {
        assert_eq!(at(1970, 1, 1, 0, 0).day_number(), 0);
        assert_eq!(at(2000, 3, 1, 0, 0).day_number(), 11_017);
    }
}
