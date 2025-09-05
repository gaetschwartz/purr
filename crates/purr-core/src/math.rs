use std::time::Duration;

pub trait RoundToUnit {
    /// Round a number to the nearest units of a given base.
    /// round_to_base(112, 10) = 100
    /// round_to_base(543, 10) = 500
    fn round_to_unit(self, base: Self) -> Self;
}

macro_rules! impl_round_at_base {
    ($t:ty) => {};
}
impl_round_at_base!(u8);
impl_round_at_base!(u16);
impl_round_at_base!(u32);
impl_round_at_base!(u64);
impl_round_at_base!(usize);
impl_round_at_base!(i8);
impl_round_at_base!(i16);
impl_round_at_base!(i32);
impl_round_at_base!(i64);
impl_round_at_base!(isize);

pub struct HumanSpeed(pub f64, pub ByteSpeed);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ByteSpeed {
    dividend: (usize, &'static str),
    divisor: (Duration, &'static str),
}

impl ByteSpeed {
    pub const BPS: ByteSpeed = ByteSpeed {
        dividend: (1, "B"),
        divisor: (Duration::from_secs(1), "s"),
    };
    pub const KBPS: ByteSpeed = ByteSpeed {
        dividend: (1_000, "KB"),
        divisor: (Duration::from_secs(1), "s"),
    };
    pub const MBPS: ByteSpeed = ByteSpeed {
        dividend: (1_000_000, "MB"),
        divisor: (Duration::from_secs(1), "s"),
    };
    pub const GBPS: ByteSpeed = ByteSpeed {
        dividend: (1_000_000_000, "GB"),
        divisor: (Duration::from_secs(1), "s"),
    };
    pub const TBPS: ByteSpeed = ByteSpeed {
        dividend: (1_000_000_000_000, "TB"),
        divisor: (Duration::from_secs(1), "s"),
    };
    pub const KIBPS: ByteSpeed = ByteSpeed {
        dividend: (1 << 10, "KiB"),
        divisor: (Duration::from_secs(1), "s"),
    };
    pub const MIBPS: ByteSpeed = ByteSpeed {
        dividend: (1 << 20, "MiB"),
        divisor: (Duration::from_secs(1), "s"),
    };
    pub const GIBPS: ByteSpeed = ByteSpeed {
        dividend: (1 << 30, "GiB"),
        divisor: (Duration::from_secs(1), "s"),
    };
    pub const TIBPS: ByteSpeed = ByteSpeed {
        dividend: (1 << 40, "TiB"),
        divisor: (Duration::from_secs(1), "s"),
    };

    pub const SI_SIZE_UNITS: &'static [(usize, &'static str)] = &[
        (1, "B"),
        (1_000, "KB"),
        (1_000 * 1_000, "MB"),
        (1_000 * 1_000 * 1_000, "GB"),
        (1_000 * 1_000 * 1_000 * 1_000, "TB"),
        (1_000 * 1_000 * 1_000 * 1_000 * 1_000, "PB"),
    ];
    pub const HUMAN_SIZE_UNITS: &'static [(usize, &'static str)] = &[
        (1, "B"),
        (1 << 10, "KiB"),
        (1 << 20, "MiB"),
        (1 << 30, "GiB"),
        (1 << 40, "TiB"),
        (1 << 50, "PiB"),
    ];
    pub const fn throughput(qty: usize, duration: Duration) -> Self {
        if qty == 0 || duration.is_zero() {
            return ByteSpeed::BPS; // Avoid division by zero
        }
        let speed = qty as f64 / duration.as_secs_f64();
        Self::mul_f64(speed, ByteSpeed::BPS)
    }

    pub const fn mul(value: usize, speed: ByteSpeed) -> Self {
        ByteSpeed {
            dividend: (value * speed.dividend.0, speed.dividend.1),
            divisor: speed.divisor,
        }
    }

    pub const fn mul_f64(value: f64, speed: ByteSpeed) -> Self {
        ByteSpeed {
            dividend: ((speed.dividend.0 as f64 * value) as usize, speed.dividend.1),
            divisor: speed.divisor,
        }
    }

    pub const fn reduce(self) -> Self {
        let dividend = self.dividend.0;
        let mut unit_index = 0;
        loop {
            let unit = Self::SI_SIZE_UNITS[unit_index];
            if dividend < unit.0 {
                break;
            }
            unit_index += 1;
        }
        ByteSpeed {
            dividend: (
                dividend / Self::SI_SIZE_UNITS[unit_index - 1].0,
                Self::SI_SIZE_UNITS[unit_index - 1].1,
            ),
            divisor: self.divisor,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Unit {
    pub scaling: Scaling,
    pub name: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Value<T>(pub T, pub Unit);

macro_rules! impl_for {
    ($t:ty) => {
        impl std::fmt::Display for Value<$t> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                let Value(value, unit) = self;
                let factor = unit.scaling.factor as $t;
                let mut index = 0;
                let mut scaled_value = *value;
                while scaled_value >= factor && index < unit.scaling.names.len() - 1 {
                    scaled_value /= factor;
                    index += 1;
                }
                let unit_name = unit.scaling.names[index];

                write!(f, "{} {}{}", scaled_value, unit_name, unit.name)
            }
        }
        impl std::ops::Mul<ByteSpeed> for $t {
            type Output = ByteSpeed;

            fn mul(self, rhs: ByteSpeed) -> Self::Output {
                ByteSpeed::mul(self as usize, rhs)
            }
        }
        impl std::ops::Div<ByteSpeed> for $t {
            type Output = Duration;

            fn div(self, rhs: ByteSpeed) -> Self::Output {
                if rhs.dividend.0 == 0 || rhs.divisor.0.is_zero() {
                    return Duration::MAX; // Avoid division by zero
                }
                Duration::from_secs_f64(
                    self as f64 * rhs.divisor.0.as_secs_f64() / rhs.dividend.0 as f64,
                )
            }
        }
        impl RoundToUnit for $t {
            fn round_to_unit(self, base: Self) -> Self {
                let mut unit = base;
                loop {
                    if self < unit {
                        return 0 as Self;
                    }
                    if self < unit * base {
                        return self - (self % unit);
                    }
                    unit *= base;
                }
            }
        }
    };
}
impl_for!(usize);
impl_for!(u8);
impl_for!(u16);
impl_for!(u32);
impl_for!(u64);
impl_for!(i8);
impl_for!(i16);
impl_for!(i32);
impl_for!(i64);
impl_for!(isize);
impl_for!(f32);
impl_for!(f64);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Scaling {
    pub factor: usize,
    pub names: &'static [&'static str],
    pub type_name: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DurationRange(pub Duration, pub Duration);
impl DurationRange {
    pub const fn new(min: Duration, max: Duration) -> Self {
        Self(min, max)
    }

    pub fn contains(&self, duration: Duration) -> bool {
        duration >= self.0 && duration <= self.1
    }

    pub const fn len(&self) -> Duration {
        self.1.abs_diff(self.0)
    }

    pub const fn unit_range(value: Duration, unit: Duration) -> Self {
        let (vs, vn) = (value.as_secs(), value.subsec_nanos());
        let (us, un) = (unit.as_secs(), unit.subsec_nanos());
        let min = Duration::new(vs - (vs % us), vn - (vn % un));
        let max = Duration::saturating_add(min, unit);
        Self(min, max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use pretty_assertions::assert_eq;
    use rstest::rstest;

    #[rstest]
    #[case(123, 10, 100)]
    #[case(512, 10, 500)]
    #[case(678 * 1024, 2, 512 * 1024)]
    #[case(4 * 1024 * 1024 * 1024, 1024, 4 * 1024 * 1024 * 1024)]
    fn test_round_to_unit(#[case] value: usize, #[case] base: usize, #[case] expected: usize) {
        assert_eq!(value.round_to_unit(base), expected);
    }

    #[test]
    fn test_speed_multiplication() {
        let speed = 2 * ByteSpeed::BPS;
        assert_eq!(
            speed,
            ByteSpeed {
                dividend: (2, "B"),
                divisor: (Duration::from_secs(1), "s"),
            }
        );
    }

    #[test]
    fn test_speed_division() {
        let speed = 1000 / ByteSpeed::BPS;
        assert_eq!(speed, Duration::from_secs_f64(1000.0));
        let speed = 2.0 / ByteSpeed::BPS;
        assert_eq!(speed, Duration::from_secs_f64(2.0));
    }

    #[test]
    fn test_speed_to_byte_speed() {
        let speed = ByteSpeed::throughput(512 * 1024, Duration::from_secs(16));
        let byte_speed = speed.reduce();
        assert_eq!(
            byte_speed,
            ByteSpeed {
                dividend: (32, "KB"),
                divisor: (Duration::from_secs(1), "s"),
            }
        );
    }
}
