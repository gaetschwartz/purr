//! Picking one device from the detected ones by what `purr sys` shows about it.

use std::{fmt, str::FromStr};

use purr_common::platform::DeviceType;

use super::{backend::AccelBackend, devices::DetectedDevice};

/// A field a selector can match on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectorField {
    Index,
    Id,
    Backend,
    Type,
}

impl SelectorField {
    const ALL: [Self; 4] = [Self::Index, Self::Id, Self::Backend, Self::Type];

    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Index => "index",
            Self::Id => "id",
            Self::Backend => "backend",
            Self::Type => "type",
        }
    }

    fn parse(text: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|field| field.name() == text)
    }
}

/// Which devices a user means: any subset of the fields, all of which must match.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DeviceSelector {
    /// The device index `purr sys` prints.
    pub index: Option<i32>,
    /// The device id `purr sys` prints, such as `0000:01:00.0`; some devices report none.
    pub device_id: Option<String>,
    pub backend: Option<AccelBackend>,
    pub kind: Option<DeviceType>,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SelectorParseError {
    #[error("the device selector is empty")]
    Empty,
    #[error(
        "`{token}` is not a backend (cuda, vulkan, ...), a device type (gpu, igpu, cpu, accel), a device id (0000:01:00.0) or an index"
    )]
    UnknownToken { token: String },
    #[error("`{field}` is not a device field; use index, id, backend or type")]
    UnknownField { field: String },
    #[error("the field `{}` is given twice", field.name())]
    Duplicate { field: SelectorField },
    #[error("`{value}` is not a device index")]
    BadIndex { value: String },
    #[error("`{value}` is not an acceleration backend")]
    BadBackend { value: String },
    #[error("`{value}` is not a device type; use gpu, igpu, cpu or accel")]
    BadType { value: String },
    #[error("`{value}` is not a device id such as 0000:01:00.0")]
    BadId { value: String },
    #[error("the field `{}` has no value", field.name())]
    EmptyValue { field: SelectorField },
}

fn parse_kind(value: &str) -> Option<DeviceType> {
    match value.to_ascii_lowercase().as_str() {
        "gpu" | "discrete" => Some(DeviceType::Gpu),
        "igpu" | "integrated" => Some(DeviceType::Igpu),
        "cpu" => Some(DeviceType::Cpu),
        "accel" => Some(DeviceType::Accel),
        _ => None,
    }
}

fn parse_backend(value: &str) -> Option<AccelBackend> {
    AccelBackend::from_str(&value.to_ascii_lowercase()).ok()
}

const fn kind_name(kind: DeviceType) -> &'static str {
    match kind {
        DeviceType::Gpu => "gpu",
        DeviceType::Igpu => "igpu",
        DeviceType::Cpu => "cpu",
        DeviceType::Accel => "accel",
        DeviceType::Unknown => "unknown",
    }
}

/// `bus:device.function`, optionally led by a domain: groups of hexadecimal digits.
fn is_pci_id(text: &str) -> bool {
    let hex = |part: &str| !part.is_empty() && part.chars().all(|c| c.is_ascii_hexdigit());
    let colons: Vec<&str> = text.split(':').collect();
    let Some((last, leading)) = colons.split_last() else {
        return false;
    };
    let Some((device, function)) = last.split_once('.') else {
        return false;
    };
    (1..=2).contains(&leading.len())
        && leading.iter().all(|part| hex(part))
        && hex(device)
        && hex(function)
}

impl DeviceSelector {
    fn set(&mut self, field: SelectorField, token: &str) -> Result<(), SelectorParseError> {
        let taken = match field {
            SelectorField::Index => self.index.is_some(),
            SelectorField::Id => self.device_id.is_some(),
            SelectorField::Backend => self.backend.is_some(),
            SelectorField::Type => self.kind.is_some(),
        };
        if taken {
            return Err(SelectorParseError::Duplicate { field });
        }
        let value = token.trim();
        if value.is_empty() {
            return Err(SelectorParseError::EmptyValue { field });
        }
        match field {
            SelectorField::Index => {
                self.index =
                    Some(
                        value
                            .parse()
                            .map_err(|_not_a_number| SelectorParseError::BadIndex {
                                value: value.to_owned(),
                            })?,
                    );
            }
            SelectorField::Id => {
                if !is_pci_id(value) {
                    return Err(SelectorParseError::BadId {
                        value: value.to_owned(),
                    });
                }
                self.device_id = Some(value.to_owned());
            }
            SelectorField::Backend => {
                self.backend =
                    Some(
                        parse_backend(value).ok_or_else(|| SelectorParseError::BadBackend {
                            value: value.to_owned(),
                        })?,
                    );
            }
            SelectorField::Type => {
                self.kind = Some(
                    parse_kind(value).ok_or_else(|| SelectorParseError::BadType {
                        value: value.to_owned(),
                    })?,
                );
            }
        }
        Ok(())
    }

    fn add_token(&mut self, token: &str) -> Result<(), SelectorParseError> {
        let unknown = || SelectorParseError::UnknownToken {
            token: token.to_owned(),
        };
        if let Some((name, value)) = token.split_once('=') {
            let field =
                SelectorField::parse(&name.trim().to_ascii_lowercase()).ok_or_else(|| {
                    SelectorParseError::UnknownField {
                        field: name.trim().to_owned(),
                    }
                })?;
            return self.set(field, value);
        }
        if token.parse::<i32>().is_ok() {
            return self.set(SelectorField::Index, token);
        }
        if is_pci_id(token) {
            return self.set(SelectorField::Id, token);
        }
        if parse_backend(token).is_some() {
            return self.set(SelectorField::Backend, token);
        }
        if parse_kind(token).is_some() {
            return self.set(SelectorField::Type, token);
        }
        Err(unknown())
    }
}

impl FromStr for DeviceSelector {
    type Err = SelectorParseError;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        let text = text.trim();
        if text.is_empty() {
            return Err(SelectorParseError::Empty);
        }
        let mut selector = Self::default();
        for token in text.split('/') {
            selector.add_token(token.trim())?;
        }
        Ok(selector)
    }
}

impl fmt::Display for DeviceSelector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut parts = Vec::new();
        if let Some(index) = self.index {
            parts.push(format!("index={index}"));
        }
        if let Some(id) = &self.device_id {
            parts.push(format!("id={id}"));
        }
        if let Some(backend) = self.backend {
            parts.push(format!("backend={backend}"));
        }
        if let Some(kind) = self.kind {
            parts.push(format!("type={}", kind_name(kind)));
        }
        f.write_str(&parts.join("/"))
    }
}

impl DeviceSelector {
    /// Whether `device` has every field this selector gives.
    #[must_use]
    pub fn matches(&self, device: &DetectedDevice) -> bool {
        self.index.is_none_or(|index| device.ggml_id == Some(index))
            && self.device_id.as_ref().is_none_or(|id| {
                device
                    .pci_id
                    .as_ref()
                    .is_some_and(|pci| pci.eq_ignore_ascii_case(id))
            })
            && self.backend.is_none_or(|backend| device.backend == backend)
            && self.kind.is_none_or(|kind| device.kind == Some(kind))
    }

    /// The one device this selector names, or why it does not name exactly one.
    pub fn resolve<'a>(
        &self,
        detected: &'a [DetectedDevice],
    ) -> Result<&'a DetectedDevice, SelectorError> {
        let matching: Vec<&DetectedDevice> = detected
            .iter()
            .filter(|device| self.matches(device))
            .collect();
        match matching.as_slice() {
            [only] => Ok(only),
            [] => Err(SelectorError::NoMatch {
                selector: self.clone(),
                devices: detected.iter().map(DeviceSummary::of).collect(),
            }),
            several => Err(SelectorError::Ambiguous {
                selector: self.clone(),
                add: self.fields_telling_apart(several),
                matches: several
                    .iter()
                    .map(|device| DeviceSummary::of(device))
                    .collect(),
            }),
        }
    }

    fn fields_telling_apart(&self, devices: &[&DetectedDevice]) -> Vec<SelectorField> {
        fn differ<T: PartialEq>(values: impl Iterator<Item = T>) -> bool {
            let values: Vec<T> = values.collect();
            values.iter().any(|value| *value != values[0])
        }
        let mut fields = Vec::new();
        if self.index.is_none() && devices.iter().all(|d| d.ggml_id.is_some()) {
            fields.push(SelectorField::Index);
        }
        if self.device_id.is_none()
            && devices.iter().any(|d| d.pci_id.is_some())
            && differ(devices.iter().map(|d| &d.pci_id))
        {
            fields.push(SelectorField::Id);
        }
        if self.backend.is_none() && differ(devices.iter().map(|d| d.backend)) {
            fields.push(SelectorField::Backend);
        }
        if self.kind.is_none() && differ(devices.iter().map(|d| d.kind)) {
            fields.push(SelectorField::Type);
        }
        fields
    }
}

/// A device as a selector error lists it, the way `purr sys` shows it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceSummary {
    pub index: Option<i32>,
    pub backend: AccelBackend,
    pub kind: Option<DeviceType>,
    pub device_id: Option<String>,
    pub name: String,
}

impl DeviceSummary {
    fn of(device: &DetectedDevice) -> Self {
        Self {
            index: device.ggml_id,
            backend: device.backend,
            kind: device.kind,
            device_id: device.pci_id.clone(),
            name: device.name.clone(),
        }
    }
}

impl fmt::Display for DeviceSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let index = self
            .index
            .map_or_else(|| "-".to_owned(), |index| index.to_string());
        write!(
            f,
            "{index}: {} - {}, {}, id {}",
            self.backend.pretty_name(),
            self.name,
            self.kind.map_or("no device type", DeviceType::label),
            self.device_id.as_deref().unwrap_or("not reported"),
        )
    }
}

fn list(devices: &[DeviceSummary]) -> String {
    devices
        .iter()
        .map(|device| format!("\n  {device}"))
        .collect()
}

fn names(fields: &[SelectorField]) -> String {
    fields
        .iter()
        .map(|field| field.name())
        .collect::<Vec<_>>()
        .join(", ")
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SelectorError {
    #[error("no device matches `{selector}`; the devices are:{}", list(devices))]
    NoMatch {
        selector: DeviceSelector,
        devices: Vec<DeviceSummary>,
    },
    #[error(
        "`{selector}` matches {} devices:{}\nadd one of these fields to pick one: {}",
        matches.len(),
        list(matches),
        names(add)
    )]
    Ambiguous {
        selector: DeviceSelector,
        matches: Vec<DeviceSummary>,
        add: Vec<SelectorField>,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::accel::devices::{
        detect,
        fixtures::{device, inventory, with_pci},
    };
    use AccelBackend::{Cpu, Cuda, Vulkan};

    fn machine() -> Vec<DetectedDevice> {
        detect(&inventory(
            vec![
                with_pci(device(0, "RTX", Cuda, DeviceType::Gpu), "0000:01:00.0"),
                with_pci(device(1, "RTX", Vulkan, DeviceType::Gpu), "0000:01:00.0"),
                device(2, "Radeon", Vulkan, DeviceType::Igpu),
                device(3, "Processor", Cpu, DeviceType::Cpu),
            ],
            &[Cuda, Vulkan, Cpu],
        ))
    }

    fn selector(text: &str) -> DeviceSelector {
        text.parse().unwrap()
    }

    fn index_of(text: &str) -> Result<Option<i32>, SelectorError> {
        let devices = machine();
        selector(text)
            .resolve(&devices)
            .map(|device| device.ggml_id)
    }

    fn only(configure: impl FnOnce(&mut DeviceSelector)) -> DeviceSelector {
        let mut selector = DeviceSelector::default();
        configure(&mut selector);
        selector
    }

    #[test]
    fn a_bare_number_is_the_index() {
        assert_eq!(selector("2"), only(|s| s.index = Some(2)));
    }

    #[test]
    fn bare_tokens_name_their_own_field_in_any_order() {
        assert_eq!(
            selector("igpu/Vulkan"),
            only(|s| {
                s.backend = Some(Vulkan);
                s.kind = Some(DeviceType::Igpu);
            })
        );
        assert_eq!(
            selector("0000:01:00.0/cuda"),
            only(|s| {
                s.device_id = Some("0000:01:00.0".to_owned());
                s.backend = Some(Cuda);
            })
        );
        assert_eq!(selector("gpu"), only(|s| s.kind = Some(DeviceType::Gpu)));
        assert_eq!(
            selector("rocm"),
            only(|s| s.backend = Some(AccelBackend::Hip))
        );
        assert_eq!(
            selector("accel"),
            only(|s| s.kind = Some(DeviceType::Accel))
        );
    }

    #[test]
    fn a_short_pci_id_is_an_id() {
        assert_eq!(
            selector("01:00.0"),
            only(|s| s.device_id = Some("01:00.0".to_owned()))
        );
    }

    #[test]
    fn the_explicit_form_means_the_same() {
        assert_eq!(
            selector("backend=vulkan/type=igpu"),
            selector("vulkan/igpu")
        );
        assert_eq!(
            selector("index=3/id=0000:01:00.0"),
            only(|s| {
                s.index = Some(3);
                s.device_id = Some("0000:01:00.0".to_owned());
            })
        );
    }

    #[test]
    fn cpu_alone_selects_the_cpu() {
        assert_eq!(index_of("cpu"), Ok(Some(3)));
        assert_eq!(index_of("type=cpu"), Ok(Some(3)));
        assert_eq!(index_of("backend=cpu"), Ok(Some(3)));
    }

    #[test]
    fn malformed_selectors_are_typed_errors() {
        let parse = |text: &str| text.parse::<DeviceSelector>();
        let unknown = |token: &str| {
            Err(SelectorParseError::UnknownToken {
                token: token.to_owned(),
            })
        };
        assert_eq!(parse(" "), Err(SelectorParseError::Empty));
        assert_eq!(parse("vulcan"), unknown("vulcan"));
        assert_eq!(parse("vulkan/#2"), unknown("#2"));
        assert_eq!(parse("@0000:01:00.0"), unknown("@0000:01:00.0"));
        assert_eq!(parse("cuda@0000:01:00.0"), unknown("cuda@0000:01:00.0"));
        assert_eq!(parse("vulkan/"), unknown(""));
        assert_eq!(parse("vulkan,igpu"), unknown("vulkan,igpu"));
        assert_eq!(
            parse("colour=red"),
            Err(SelectorParseError::UnknownField {
                field: "colour".to_owned()
            })
        );
        assert_eq!(
            parse("vulkan/cuda"),
            Err(SelectorParseError::Duplicate {
                field: SelectorField::Backend
            })
        );
        assert_eq!(
            parse("backend=vulkan/vulkan"),
            Err(SelectorParseError::Duplicate {
                field: SelectorField::Backend
            })
        );
        assert_eq!(
            parse("1/2"),
            Err(SelectorParseError::Duplicate {
                field: SelectorField::Index
            })
        );
        assert_eq!(
            parse("backend=tpu"),
            Err(SelectorParseError::BadBackend {
                value: "tpu".to_owned()
            })
        );
        assert_eq!(
            parse("type=fpga"),
            Err(SelectorParseError::BadType {
                value: "fpga".to_owned()
            })
        );
        assert_eq!(
            parse("index=x"),
            Err(SelectorParseError::BadIndex {
                value: "x".to_owned()
            })
        );
        assert_eq!(
            parse("id=7"),
            Err(SelectorParseError::BadId {
                value: "7".to_owned()
            })
        );
        assert_eq!(
            parse("id="),
            Err(SelectorParseError::EmptyValue {
                field: SelectorField::Id
            })
        );
    }

    #[test]
    fn a_selector_prints_in_the_explicit_form_and_parses_back() {
        let text = "index=1/id=0000:01:00.0/backend=vulkan/type=gpu";
        assert_eq!(selector(text).to_string(), text);
    }

    #[test]
    fn the_integrated_gpu_is_picked_by_its_type() {
        assert_eq!(index_of("type=igpu"), Ok(Some(2)));
        assert_eq!(index_of("backend=vulkan/type=igpu"), Ok(Some(2)));
    }

    #[test]
    fn the_index_picks_one_device() {
        assert_eq!(index_of("1"), Ok(Some(1)));
        assert_eq!(index_of("3"), Ok(Some(3)));
    }

    #[test]
    fn an_id_and_a_backend_pick_the_card_under_one_backend() {
        assert_eq!(index_of("id=0000:01:00.0/backend=cuda"), Ok(Some(0)));
        assert_eq!(index_of("id=0000:01:00.0/backend=vulkan"), Ok(Some(1)));
    }

    #[test]
    fn a_backend_shared_by_two_devices_is_ambiguous_and_says_what_would_tell_them_apart() {
        let Err(SelectorError::Ambiguous { matches, add, .. }) = index_of("backend=vulkan") else {
            panic!("expected an ambiguous selector");
        };
        assert_eq!(
            matches.iter().map(|m| m.index).collect::<Vec<_>>(),
            [Some(1), Some(2)]
        );
        assert_eq!(
            add,
            [SelectorField::Index, SelectorField::Id, SelectorField::Type]
        );
    }

    #[test]
    fn an_id_shared_by_two_backends_is_ambiguous_by_backend() {
        let Err(SelectorError::Ambiguous { add, .. }) = index_of("id=0000:01:00.0") else {
            panic!("expected an ambiguous selector");
        };
        assert_eq!(add, [SelectorField::Index, SelectorField::Backend]);
    }

    #[test]
    fn a_selector_matching_nothing_lists_every_device() {
        let Err(SelectorError::NoMatch { devices, .. }) = index_of("backend=metal") else {
            panic!("expected no match");
        };
        assert_eq!(
            devices.iter().map(|d| d.index).collect::<Vec<_>>(),
            [Some(0), Some(1), Some(2), Some(3)]
        );
    }

    #[test]
    fn an_id_never_matches_a_device_that_reports_none() {
        assert!(matches!(
            index_of("id=0000:01:00.0/type=igpu"),
            Err(SelectorError::NoMatch { .. })
        ));
        assert!(matches!(
            index_of("type=cpu/id=0000:00:00.0"),
            Err(SelectorError::NoMatch { .. })
        ));
    }

    #[test]
    fn the_error_lists_the_devices_the_way_purr_sys_shows_them() {
        let error = index_of("backend=vulkan").unwrap_err();
        assert_eq!(
            error.to_string(),
            "`backend=vulkan` matches 2 devices:\n  \
             1: Vulkan - RTX, discrete GPU, id 0000:01:00.0\n  \
             2: Vulkan - Radeon, integrated GPU, id not reported\n\
             add one of these fields to pick one: index, id, type"
        );
    }
}
