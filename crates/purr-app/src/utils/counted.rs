/// `count` with the noun in its singular for one and its plural otherwise, such as "1 segment".
#[must_use]
pub fn counted(count: usize, singular: &str, plural: &str) -> String {
    if count == 1 {
        format!("{count} {singular}")
    } else {
        format!("{count} {plural}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_is_singular_and_everything_else_is_plural() {
        assert_eq!(counted(0, "segment", "segments"), "0 segments");
        assert_eq!(counted(1, "segment", "segments"), "1 segment");
        assert_eq!(counted(2, "segment", "segments"), "2 segments");
    }
}
