//! Gregorian date and timestamp compatibility for Innovate's serialized inputs.

use std::{borrow::Cow, fmt, time::SystemTime};
use time::format_description::well_known::Rfc3339;
use time::{Date, Duration, Month, OffsetDateTime, PrimitiveDateTime, Time, UtcDateTime};

const MIN_YEAR: i32 = -262_143;
const MAX_YEAR: i32 = 262_142;
const NANOS_PER_SECOND: u32 = 1_000_000_000;
const SECONDS_PER_WEEK: i128 = 604_800;

pub(crate) const MAX_DATE: Date = match Date::from_calendar_date(MAX_YEAR, Month::December, 31) {
    Ok(date) => date,
    Err(_) => panic!("invalid calendar boundary"),
};

pub(crate) fn supported_date(date: Date) -> Option<Date> {
    (MIN_YEAR..=MAX_YEAR).contains(&date.year()).then_some(date)
}

pub(crate) fn date(mut text: &str) -> Option<Date> {
    let parsed = date_prefix(&mut text)?;
    text.is_empty().then_some(parsed)
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
    let month = Month::try_from(digits(text, 2)? as u8).ok()?;
    *text = text.strip_prefix('-')?.trim_start();
    let day = digits(text, 2)? as u8;
    supported_date(Date::from_calendar_date(year, month, day).ok()?)
}

/// Retain a leap second as a fractional value above the ordinary second
/// immediately before it. This keeps the serialized label and ordering intact.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub(crate) struct Timestamp {
    whole: PrimitiveDateTime,
    fraction: u32,
}

impl Timestamp {
    pub(crate) fn date(self) -> Date {
        self.whole.date()
    }

    pub(crate) fn parse_offset(text: &str) -> Option<Self> {
        let leap = text.as_bytes().get(17..19)? == b"60";
        let tail = text.get(19..)?;
        let offset = tail.strip_prefix('.').map_or(tail, |digits| {
            digits.trim_start_matches(|value: char| value.is_ascii_digit())
        });
        let needs_separator = text.as_bytes()[10] == b' ';
        let needs_sign = offset.starts_with('−');
        let mut normalized = Cow::Borrowed(text);
        if leap || needs_separator || needs_sign {
            let candidate = normalized.to_mut();
            if leap {
                candidate.replace_range(17..19, "59");
            }
            if needs_separator {
                candidate.replace_range(10..11, "T");
            }
            if needs_sign {
                let position = text.len() - offset.len();
                candidate.replace_range(position..position + '−'.len_utf8(), "-");
            }
        }
        let utc = OffsetDateTime::parse(&normalized, &Rfc3339).ok()?.to_utc();
        let mut parsed = Self::from_utc(utc);
        if leap {
            parsed.fraction += NANOS_PER_SECOND;
        }
        Some(parsed)
    }

    pub(crate) fn parse_naive(mut text: &str) -> Option<Self> {
        let date = date_prefix(&mut text)?;
        text = text.strip_prefix('T').unwrap_or(text).trim_start();
        let hour = digits(&mut text, 2)? as u8;
        text = text.strip_prefix(':')?.trim_start();
        let minute = digits(&mut text, 2)? as u8;
        text = text.strip_prefix(':')?.trim_start();
        let second = digits(&mut text, 2)? as u8;
        let mut fraction = 0;
        if let Some(decimals) = text.strip_prefix('.') {
            text = decimals;
            fraction =
                digits(&mut text, 9)? * 10u32.pow((9 - (decimals.len() - text.len())) as u32);
            text = text.trim_start_matches(|value: char| value.is_ascii_digit());
        }
        if !text.is_empty() || second > 60 {
            return None;
        }
        if second == 60 {
            fraction += NANOS_PER_SECOND;
        }
        let clock = Time::from_hms(hour, minute, second.min(59)).ok()?;
        Some(Self {
            whole: date.with_time(clock),
            fraction,
        })
    }

    fn from_utc(utc: UtcDateTime) -> Self {
        let fraction = utc.nanosecond();
        let clock = utc
            .time()
            .replace_nanosecond(0)
            .expect("zero nanoseconds are valid");
        Self {
            whole: utc.date().with_time(clock),
            fraction,
        }
    }

    pub(crate) fn from_system_time(time: SystemTime) -> Self {
        Self::from_utc(time.into())
    }

    pub(crate) fn weekly_bucket(self, anchor: Self) -> Option<Self> {
        let mut seconds = (self.whole - anchor.whole).whole_seconds();
        // Include the intervening leap second when either endpoint records one.
        // The comparison is within a clock day to preserve the established weekly convention.
        if self.whole.time() > anchor.whole.time() && anchor.fraction >= NANOS_PER_SECOND {
            seconds += 1;
        } else if self.whole.time() < anchor.whole.time() && self.fraction >= NANOS_PER_SECOND {
            seconds -= 1;
        }
        let nanos_per_second = i128::from(NANOS_PER_SECOND);
        let elapsed_nanos = i128::from(seconds) * nanos_per_second + i128::from(self.fraction)
            - i128::from(anchor.fraction);
        let weeks = elapsed_nanos / (SECONDS_PER_WEEK * nanos_per_second);
        if weeks == 0 {
            return Some(anchor);
        }
        let whole = anchor
            .whole
            .checked_add(Duration::weeks(i64::try_from(weeks).ok()?))?;
        supported_date(whole.date())?;
        Some(Self {
            whole,
            fraction: anchor.fraction % NANOS_PER_SECOND,
        })
    }
}

impl fmt::Display for Timestamp {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        let clock = self.whole.time();
        let second = clock.second() + u8::from(self.fraction >= NANOS_PER_SECOND);
        write!(
            output,
            "{}T{:02}:{:02}:{second:02}",
            self.date(),
            clock.hour(),
            clock.minute()
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
