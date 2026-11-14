use serde::{Deserialize, Serialize};

/// The optional abilities a compute device reports, under the names `purr sys` shows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeviceCaps {
    #[serde(default, rename = "async_")]
    pub asynchronous: bool,
    #[serde(default)]
    pub host_buffer: bool,
    #[serde(default)]
    pub buffer_from_host_ptr: bool,
    #[serde(default)]
    pub events: bool,
}

impl DeviceCaps {
    /// Every capability with whether the device has it, in the order they are listed.
    #[must_use]
    pub const fn flags(self) -> [(&'static str, bool); 4] {
        [
            ("async", self.asynchronous),
            ("host_buffer", self.host_buffer),
            ("buffer_from_host_ptr", self.buffer_from_host_ptr),
            ("events", self.events),
        ]
    }

    /// The names of the capabilities that are set.
    #[must_use]
    pub fn names(self) -> Vec<&'static str> {
        self.flags()
            .into_iter()
            .filter_map(|(name, set)| set.then_some(name))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CAPS: DeviceCaps = DeviceCaps {
        asynchronous: true,
        host_buffer: false,
        buffer_from_host_ptr: true,
        events: false,
    };

    #[test]
    fn only_the_capabilities_that_are_set_are_named() {
        assert_eq!(CAPS.names(), ["async", "buffer_from_host_ptr"]);
    }

    #[test]
    fn a_device_with_no_capabilities_names_none() {
        let none = DeviceCaps {
            asynchronous: false,
            host_buffer: false,
            buffer_from_host_ptr: false,
            events: false,
        };
        assert_eq!(none.names(), Vec::<&str>::new());
    }

    #[test]
    fn the_json_keys_are_the_ones_purr_sys_has_always_printed() {
        assert_eq!(
            serde_json::to_string(&CAPS).unwrap(),
            r#"{"async_":true,"host_buffer":false,"buffer_from_host_ptr":true,"events":false}"#
        );
    }

    #[test]
    fn printed_capabilities_read_back() {
        let printed = serde_json::to_string(&Some(CAPS)).unwrap();
        assert_eq!(
            serde_json::from_str::<Option<DeviceCaps>>(&printed).unwrap(),
            Some(CAPS)
        );
        assert_eq!(
            serde_json::from_str::<Option<DeviceCaps>>("null").unwrap(),
            None
        );
    }
}
