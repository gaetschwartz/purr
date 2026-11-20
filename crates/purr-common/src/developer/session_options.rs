use serde_json::{Map, Value};

/// Options for an ONNX Runtime session, as a JSON object.
pub type SessionOptions = Map<String, Value>;

#[oopsie::oopsie]
#[derive(Clone, PartialEq, Eq)]
pub enum SessionOptionsError {
    #[oopsie("The session options are not valid JSON: {reason}")]
    NotJson { reason: String },
    #[oopsie("The session options must be a JSON object")]
    NotAnObject,
}

/// Reads the text a user typed as session options; blank text means no options.
pub fn parse(text: &str) -> Result<SessionOptions, SessionOptionsError> {
    if text.trim().is_empty() {
        return Ok(SessionOptions::new());
    }
    match serde_json::from_str::<Value>(text) {
        Ok(Value::Object(options)) => Ok(options),
        Ok(_) => Err(SessionOptionsError::NotAnObject),
        Err(error) => Err(SessionOptionsError::NotJson {
            reason: error.to_string(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blank_text_is_no_options() {
        assert_eq!(parse("  \n"), Ok(SessionOptions::new()));
    }

    #[test]
    fn an_object_is_kept_whole() {
        let options = parse(r#"{"graphOptimizationLevel":"basic","logSeverityLevel":2}"#).unwrap();
        assert_eq!(options.len(), 2);
        assert_eq!(
            options.get("graphOptimizationLevel"),
            Some(&Value::String("basic".to_owned()))
        );
    }

    #[test]
    fn anything_but_an_object_is_refused() {
        assert_eq!(parse("[1]"), Err(SessionOptionsError::NotAnObject));
        assert_eq!(parse("3"), Err(SessionOptionsError::NotAnObject));
    }

    #[test]
    fn broken_json_is_refused() {
        assert!(matches!(
            parse("{\"a\":"),
            Err(SessionOptionsError::NotJson { .. })
        ));
    }
}
