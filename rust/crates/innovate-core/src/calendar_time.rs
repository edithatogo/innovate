//! Jiff calendar operations with the existing JSON date range and leap-second labels.

use jiff::civil::{Date as CivilDate, Time};
use jiff::tz::Offset;
use jiff::{Span, Timestamp as JiffTimestamp};
use std::{fmt, time::SystemTime};

const MIN_YEAR: i32 = -262_143;
const MAX_YEAR: i32 = 262_142;
const YEARS_PER_CYCLE: i32 = 400;
const DAYS_PER_CYCLE: i32 = 146_097;
const SECONDS_PER_DAY: i64 = 86_400;
const NANOS_PER_SECOND: u32 = 1_000_000_000;

/// A date in the established input range. The Gregorian calendar repeats
/// every 400 years, so Jiff validates and computes within one cycle.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) struct Date {
    cycle: i32,
    within_cycle: CivilDate,
}

impl Date {
    pub(crate) const MAX: Self = Self {
        cycle: MAX_YEAR.div_euclid(YEARS_PER_CYCLE),
        within_cycle: CivilDate::constant(MAX_YEAR.rem_euclid(YEARS_PER_CYCLE) as i16, 12, 31),
    };

    pub(crate) fn new(year: i32, month: i8, day: i8) -> Option<Self> {
        if !(MIN_YEAR..=MAX_YEAR).contains(&year) {
            return None;
        }
        Some(Self {
            cycle: year.div_euclid(YEARS_PER_CYCLE),
            within_cycle: CivilDate::new(year.rem_euclid(YEARS_PER_CYCLE) as i16, month, day)
                .ok()?,
        })
    }

    fn from_cycle(cycle: i32, date: CivilDate) -> Option<Self> {
        Self::new(
            cycle * YEARS_PER_CYCLE + i32::from(date.year()),
            date.month(),
            date.day(),
        )
    }

    pub(crate) fn year(self) -> i32 {
        self.cycle * YEARS_PER_CYCLE + i32::from(self.within_cycle.year())
    }

    pub(crate) fn month(self) -> i32 {
        i32::from(self.within_cycle.month())
    }

    pub(crate) fn checked_add_days(self, days: i32) -> Option<Self> {
        let cycle = self.cycle + days.div_euclid(DAYS_PER_CYCLE);
        let shifted = self
            .within_cycle
            .checked_add(Span::new().days(days.rem_euclid(DAYS_PER_CYCLE)))
            .ok()?;
        Self::from_cycle(cycle, shifted)
    }

    pub(crate) fn days_since(self, other: Self) -> i32 {
        (self.cycle - other.cycle) * DAYS_PER_CYCLE
            + (self.within_cycle - other.within_cycle).get_days()
    }

    pub(crate) fn month_start(self, offset: i32) -> Option<Self> {
        let month = self.month() - 1 + offset;
        Self::new(
            self.year() + month.div_euclid(12),
            month.rem_euclid(12) as i8 + 1,
            1,
        )
    }

    pub(crate) fn parse(mut text: &str) -> Option<Self> {
        let date = date_prefix(&mut text)?;
        text.is_empty().then_some(date)
    }
}

impl fmt::Display for Date {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        let year = self.year();
        if (0..=9999).contains(&year) {
            write!(output, "{year:04}")?;
        } else {
            write!(output, "{year:+05}")?;
        }
        write!(
            output,
            "-{:02}-{:02}",
            self.month(),
            self.within_cycle.day()
        )
    }
}

fn digits(text: &mut &str, width: usize) -> Option<u32> {
    let count = text
        .bytes()
        .take(width)
        .take_while(u8::is_ascii_digit)
        .count();
    let (value, rest) = text.split_at(count);
    *text = rest;
    value.parse().ok()
}

fn date_prefix(text: &mut &str) -> Option<Date> {
    *text = text.trim_start();
    let year = match text.as_bytes().first()? {
        b'+' | b'-' => {
            let negative = text.as_bytes()[0] == b'-';
            *text = &text[1..];
            let magnitude = i32::try_from(digits(text, usize::MAX)?).ok()?;
            if negative { -magnitude } else { magnitude }
        }
        _ => digits(text, 4)? as i32,
    };
    *text = text.strip_prefix('-')?.trim_start();
    let month = digits(text, 2)? as i8;
    *text = text.strip_prefix('-')?.trim_start();
    let day = digits(text, 2)? as i8;
    Date::new(year, month, day)
}

fn clock_prefix(text: &mut &str) -> Option<(Time, u32)> {
    let hour = digits(text, 2)? as i8;
    *text = text.strip_prefix(':')?.trim_start();
    let minute = digits(text, 2)? as i8;
    *text = text.strip_prefix(':')?.trim_start();
    let second = digits(text, 2)? as i8;
    if second > 60 {
        return None;
    }
    let mut fraction = 0;
    if let Some(decimals) = text.strip_prefix('.') {
        *text = decimals;
        fraction = digits(text, 9)? * 10u32.pow((9 - (decimals.len() - text.len())) as u32);
        *text = text.trim_start_matches(|value: char| value.is_ascii_digit());
    }
    // Jiff's civil time ends at second 59. Keep the extra second in the label.
    if second == 60 {
        fraction += NANOS_PER_SECOND;
    }
    Some((Time::new(hour, minute, second.min(59), 0).ok()?, fraction))
}

fn rfc3339_offset(text: &str) -> Option<Offset> {
    if text.eq_ignore_ascii_case("z") {
        return Some(Offset::UTC);
    }
    let (sign, text) = if let Some(rest) = text.strip_prefix('+') {
        (1, rest)
    } else {
        (
            -1,
            text.strip_prefix('-').or_else(|| text.strip_prefix('−'))?,
        )
    };
    let (hours, minutes) = text.split_once(':')?;
    if [hours, minutes]
        .iter()
        .any(|value| value.len() != 2 || !value.bytes().all(|byte| byte.is_ascii_digit()))
    {
        return None;
    }
    let (hours, minutes): (i32, i32) = (hours.parse().ok()?, minutes.parse().ok()?);
    if hours > 23 || minutes > 59 {
        return None;
    }
    Offset::from_seconds(sign * (hours * 3600 + minutes * 60)).ok()
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) struct Timestamp {
    pub(crate) date: Date,
    clock: Time,
    fraction: u32,
}

impl Timestamp {
    pub(crate) fn parse_offset(text: &str) -> Option<Self> {
        let prefix = text.get(..19)?;
        // Accept the existing RFC 3339 grammar, including the space separator.
        if !prefix.bytes().enumerate().all(|(index, byte)| match index {
            4 | 7 => byte == b'-',
            10 => matches!(byte, b'T' | b't' | b' '),
            13 | 16 => byte == b':',
            _ => byte.is_ascii_digit(),
        }) {
            return None;
        }
        let date = Date::parse(&prefix[..10])?;
        let mut rest = &text[11..];
        let (clock, fraction) = clock_prefix(&mut rest)?;
        let utc = Offset::UTC.to_datetime(
            rfc3339_offset(rest)?
                .to_timestamp(date.within_cycle.to_datetime(clock))
                .ok()?,
        );
        Some(Self {
            date: Date::from_cycle(date.cycle, utc.date())?,
            clock: utc.time(),
            fraction,
        })
    }

    pub(crate) fn parse_naive(mut text: &str) -> Option<Self> {
        let date = date_prefix(&mut text)?;
        text = text.strip_prefix('T').unwrap_or(text).trim_start();
        let (clock, fraction) = clock_prefix(&mut text)?;
        text.is_empty().then_some(Self {
            date,
            clock,
            fraction,
        })
    }

    pub(crate) fn from_system_time(time: SystemTime) -> Self {
        let time = JiffTimestamp::try_from(time).expect("system clock outside Jiff's date range");
        let utc = Offset::UTC.to_datetime(time);
        let date = Date::from_cycle(0, utc.date()).expect("Jiff date within supported range");
        Self {
            date,
            clock: utc
                .time()
                .with()
                .subsec_nanosecond(0)
                .build()
                .expect("zero nanoseconds are valid"),
            fraction: utc.subsec_nanosecond() as u32,
        }
    }

    pub(crate) fn weekly_bucket(self, anchor: Self) -> Option<Self> {
        let mut seconds = i64::from(self.date.days_since(anchor.date)) * SECONDS_PER_DAY
            + self.clock.duration_since(anchor.clock).as_secs();
        // Count an intervening leap second at a recorded endpoint.
        if self.clock > anchor.clock && anchor.fraction >= NANOS_PER_SECOND {
            seconds += 1;
        } else if self.clock < anchor.clock && self.fraction >= NANOS_PER_SECOND {
            seconds -= 1;
        }
        let nanos_per_second = i128::from(NANOS_PER_SECOND);
        let elapsed = i128::from(seconds) * nanos_per_second + i128::from(self.fraction)
            - i128::from(anchor.fraction);
        let weeks = elapsed / (7 * i128::from(SECONDS_PER_DAY) * nanos_per_second);
        if weeks == 0 {
            return Some(anchor);
        }
        Some(Self {
            date: anchor
                .date
                .checked_add_days(i32::try_from(weeks * 7).ok()?)?,
            clock: anchor.clock,
            fraction: anchor.fraction % NANOS_PER_SECOND,
        })
    }
}

impl fmt::Display for Timestamp {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        let second = self.clock.second() + i8::from(self.fraction >= NANOS_PER_SECOND);
        write!(
            output,
            "{}T{:02}:{:02}:{second:02}",
            self.date,
            self.clock.hour(),
            self.clock.minute()
        )?;
        let fraction = self.fraction % NANOS_PER_SECOND;
        if fraction != 0 {
            let (scale, width) = if fraction.is_multiple_of(1_000_000) {
                (1_000_000, 3)
            } else if fraction.is_multiple_of(1_000) {
                (1_000, 6)
            } else {
                (1, 9)
            };
            write!(output, ".{:0width$}", fraction / scale)?;
        }
        Ok(())
    }
}
