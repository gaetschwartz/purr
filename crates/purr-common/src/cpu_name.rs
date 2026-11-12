//! A short display name for a processor, from the brand string the operating system reports.

fn is_core_count(token: &str) -> bool {
    token
        .strip_suffix("-Core")
        .is_some_and(|count| !count.is_empty() && count.chars().all(|c| c.is_ascii_digit()))
}

fn is_generation(token: &str) -> bool {
    ["st", "nd", "rd", "th"].iter().any(|suffix| {
        token
            .strip_suffix(suffix)
            .is_some_and(|count| !count.is_empty() && count.chars().all(|c| c.is_ascii_digit()))
    })
}

fn without_marks(brand: &str) -> String {
    ["(R)", "(r)", "(TM)", "(tm)"]
        .iter()
        .fold(brand.to_owned(), |text, mark| text.replace(mark, ""))
}

/// The brand string without trademark marks, clock speed, core count, generation and graphics
/// suffixes; the trimmed original when nothing would be left.
#[must_use]
pub fn short_cpu_name(brand: &str) -> String {
    let marked = without_marks(brand);
    let before_clock = marked.split(" @ ").next().unwrap_or_default();
    let mut tokens = before_clock.split_whitespace().peekable();
    let mut kept = Vec::new();
    while let Some(token) = tokens.next() {
        if token == "with" || token == "w/" {
            break;
        }
        if is_generation(token) && tokens.peek() == Some(&"Gen") {
            tokens.next();
            continue;
        }
        if is_core_count(token) || token == "Processor" || token == "CPU" {
            continue;
        }
        kept.push(token);
    }
    if kept.is_empty() {
        brand.trim().to_owned()
    } else {
        kept.join(" ")
    }
}

#[cfg(test)]
mod tests {
    use super::short_cpu_name;

    #[test]
    fn amd_brand_strings_lose_the_core_count_and_processor() {
        assert_eq!(
            short_cpu_name("AMD Ryzen 9 7950X 16-Core Processor"),
            "AMD Ryzen 9 7950X"
        );
    }

    #[test]
    fn amd_laptop_strings_lose_the_graphics_suffix() {
        assert_eq!(
            short_cpu_name("AMD Ryzen 7 5800U with Radeon Graphics"),
            "AMD Ryzen 7 5800U"
        );
    }

    #[test]
    fn intel_brand_strings_lose_marks_generation_and_clock() {
        assert_eq!(
            short_cpu_name("13th Gen Intel(R) Core(TM) i7-13700K"),
            "Intel Core i7-13700K"
        );
        assert_eq!(
            short_cpu_name("Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz"),
            "Intel Core i7-8700"
        );
    }

    #[test]
    fn names_with_nothing_to_strip_are_unchanged() {
        assert_eq!(short_cpu_name("Apple M2 Pro"), "Apple M2 Pro");
    }

    #[test]
    fn a_name_made_only_of_filler_stays_as_reported() {
        assert_eq!(short_cpu_name(" Processor "), "Processor");
    }
}
