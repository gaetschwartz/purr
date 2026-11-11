/// "1 entry" or "N entries".
#[must_use]
pub fn entries(count: usize) -> String {
    match count {
        1 => "1 entry".to_owned(),
        many => format!("{many} entries"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_entry_is_singular() {
        assert_eq!(entries(0), "0 entries");
        assert_eq!(entries(1), "1 entry");
        assert_eq!(entries(2), "2 entries");
    }
}
