//! A readable name for a WebGPU adapter, from the identification strings a browser exposes.

/// The strings of a browser's `GPUAdapterInfo`; each is empty when the browser withholds it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AdapterStrings<'a> {
    pub vendor: &'a str,
    pub architecture: &'a str,
    pub device: &'a str,
    pub description: &'a str,
}

const KNOWN_VENDORS: [(&str, &str); 8] = [
    ("nvidia", "NVIDIA"),
    ("amd", "AMD"),
    ("intel", "Intel"),
    ("apple", "Apple"),
    ("qualcomm", "Qualcomm"),
    ("arm", "Arm"),
    ("google", "Google"),
    ("microsoft", "Microsoft"),
];

fn capitalised(text: &str) -> String {
    let mut chars = text.chars();
    chars.next().map_or_else(String::new, |first| {
        first.to_uppercase().chain(chars).collect()
    })
}

fn vendor_name(vendor: &str) -> String {
    KNOWN_VENDORS
        .iter()
        .find(|(raw, _)| raw.eq_ignore_ascii_case(vendor))
        .map_or_else(|| capitalised(vendor), |(_, shown)| (*shown).to_owned())
}

/// The most descriptive name the adapter gives: its description, else vendor and device, else
/// vendor and architecture; `None` when the browser reports nothing.
#[must_use]
pub fn adapter_name(strings: AdapterStrings<'_>) -> Option<String> {
    let description = strings.description.trim();
    let vendor = strings.vendor.trim();
    let architecture = strings.architecture.trim();
    let device = strings.device.trim();
    if !description.is_empty() {
        return Some(description.to_owned());
    }
    let vendor = (!vendor.is_empty()).then(|| vendor_name(vendor));
    match (vendor, device, architecture) {
        (Some(vendor), device, _) if !device.is_empty() => Some(format!("{vendor} {device}")),
        (Some(vendor), _, architecture) if !architecture.is_empty() => {
            Some(format!("{vendor} ({})", capitalised(architecture)))
        }
        (Some(vendor), _, _) => Some(vendor),
        (None, device, _) if !device.is_empty() => Some(device.to_owned()),
        (None, _, architecture) if !architecture.is_empty() => Some(capitalised(architecture)),
        (None, _, _) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strings<'a>(
        vendor: &'a str,
        architecture: &'a str,
        device: &'a str,
        description: &'a str,
    ) -> AdapterStrings<'a> {
        AdapterStrings {
            vendor,
            architecture,
            device,
            description,
        }
    }

    #[test]
    fn a_description_wins() {
        assert_eq!(
            adapter_name(strings(
                "nvidia",
                "ampere",
                "0x2684",
                " NVIDIA GeForce RTX 4090 "
            )),
            Some("NVIDIA GeForce RTX 4090".to_owned())
        );
    }

    #[test]
    fn vendor_and_architecture_read_like_a_name() {
        assert_eq!(
            adapter_name(strings("nvidia", "ampere", "", "")),
            Some("NVIDIA (Ampere)".to_owned())
        );
    }

    #[test]
    fn a_device_string_beats_the_architecture() {
        assert_eq!(
            adapter_name(strings("intel", "gen-12lp", "uhd graphics 770", "")),
            Some("Intel uhd graphics 770".to_owned())
        );
    }

    #[test]
    fn an_unknown_vendor_is_capitalised() {
        assert_eq!(
            adapter_name(strings("acme", "", "", "")),
            Some("Acme".to_owned())
        );
    }

    #[test]
    fn what_is_left_is_used_without_a_vendor() {
        assert_eq!(
            adapter_name(strings("", "", "some device", "")),
            Some("some device".to_owned())
        );
        assert_eq!(
            adapter_name(strings("", "ampere", "", "")),
            Some("Ampere".to_owned())
        );
    }

    #[test]
    fn a_browser_that_reports_nothing_gives_no_name() {
        assert_eq!(adapter_name(strings("  ", "", " ", "")), None);
        assert_eq!(adapter_name(AdapterStrings::default()), None);
    }
}
