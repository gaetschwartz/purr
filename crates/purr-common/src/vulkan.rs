//! What a Vulkan driver reports about a physical device, and how it is tied to a ggml device.

use std::fmt;

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// A Vulkan API version, decoded as the specification defines (variant bits dropped).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct VulkanApiVersion {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}

impl VulkanApiVersion {
    /// Decode a `uint32_t` Vulkan version number.
    #[must_use]
    pub const fn from_raw(raw: u32) -> Self {
        Self {
            major: (raw >> 22) & 0x7f,
            minor: (raw >> 12) & 0x3ff,
            patch: raw & 0xfff,
        }
    }
}

impl fmt::Display for VulkanApiVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

/// Where a PCI device sits on the bus.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PciBusId {
    pub domain: u32,
    pub bus: u32,
    pub device: u32,
    pub function: u32,
}

impl fmt::Display for PciBusId {
    /// `domain:bus:device.function` in lower-case hexadecimal, the way ggml writes the id.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:04x}:{:02x}:{:02x}.{:x}",
            self.domain,
            self.bus,
            self.device,
            self.function & 0xff
        )
    }
}

/// `VkPhysicalDeviceType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VulkanDeviceType {
    Other,
    IntegratedGpu,
    DiscreteGpu,
    VirtualGpu,
    Cpu,
    Unknown,
}

/// `VkDriverId`: which driver implementation answered.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VulkanDriverId {
    AmdProprietary,
    AmdOpenSource,
    MesaRadv,
    NvidiaProprietary,
    IntelProprietaryWindows,
    IntelOpenSourceMesa,
    ImaginationProprietary,
    QualcommProprietary,
    ArmProprietary,
    GoogleSwiftshader,
    GgpProprietary,
    BroadcomProprietary,
    MesaLlvmpipe,
    MoltenVk,
    CoreAviProprietary,
    JuiceProprietary,
    VerisiliconProprietary,
    MesaTurnip,
    MesaV3dv,
    MesaPanvk,
    SamsungProprietary,
    MesaVenus,
    MesaDozen,
    MesaNvk,
    ImaginationOpenSourceMesa,
    MesaAgxv,
    Unlisted(i32),
}

/// The driver's own identification, from `VkPhysicalDeviceDriverProperties`, as it reports it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VulkanDriverIdentity {
    pub id: VulkanDriverId,
    pub name: String,
    pub info: String,
}

/// One Vulkan physical device as the Vulkan driver describes it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VulkanDriverInfo {
    pub device_name: String,
    pub device_type: VulkanDeviceType,
    pub vendor_id: u32,
    pub device_id: u32,
    pub api_version: VulkanApiVersion,
    /// `None` when the driver lacks `VK_EXT_pci_bus_info`.
    pub pci_bus: Option<PciBusId>,
    /// `None` when the driver lacks `VK_KHR_driver_properties`.
    pub driver: Option<VulkanDriverIdentity>,
}

/// The Vulkan calls the probe makes that can fail.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VulkanCall {
    EnumerateInstanceVersion,
    CreateInstance,
    EnumeratePhysicalDevices,
    EnumerateDeviceExtensions,
}

/// Why the Vulkan driver could not be asked about its devices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error, Serialize, Deserialize)]
pub enum VulkanProbeError {
    #[error("the Vulkan loader could not be opened")]
    LoaderUnavailable,
    #[error("the Vulkan loader predates Vulkan 1.1")]
    LoaderBeforeVulkan11,
    #[error("{call:?} failed with Vulkan result {vk_result}")]
    Call { call: VulkanCall, vk_result: i32 },
}

/// Why a ggml Vulkan device has no driver information.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error, Serialize, Deserialize)]
pub enum VulkanDriverUnavailable {
    #[error("{0}")]
    Probe(#[from] VulkanProbeError),
    #[error("no single Vulkan device matches this one by PCI bus id or name")]
    NoUnambiguousMatch,
}

/// What `purr sys` can say about the driver of a ggml Vulkan device.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum VulkanDriverStatus {
    Reported(VulkanDriverInfo),
    Unavailable(VulkanDriverUnavailable),
}

impl VulkanDriverStatus {
    /// The `(label, value)` lines describing the driver.
    #[must_use]
    pub fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::Unavailable(reason) => {
                vec![("Driver", format!("could not be read ({reason})"))]
            }
            Self::Reported(info) => {
                let driver = info.driver.as_ref().map_or_else(
                    || "not reported by the Vulkan driver".to_owned(),
                    |driver| {
                        if driver.info.is_empty() {
                            format!("{} (version not reported by the driver)", driver.name)
                        } else {
                            format!("{} {}", driver.name, driver.info)
                        }
                    },
                );
                vec![
                    ("Driver", driver),
                    ("Vulkan API", info.api_version.to_string()),
                ]
            }
        }
    }
}

/// What ggml tells about one of its Vulkan devices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GgmlVulkanDevice<'a> {
    /// The PCI bus id string ggml reports, when it has one.
    pub pci_id: Option<&'a str>,
    /// ggml's description, which is the Vulkan device name.
    pub description: &'a str,
}

/// For each ggml device, the index of the probed device it is, if that is certain.
///
/// ggml's device order and `GGML_VK_VISIBLE_DEVICES` make positions meaningless, so devices are
/// tied by PCI bus id first, then by name when exactly one candidate remains on each side and
/// the PCI ids do not contradict it.
#[must_use]
pub fn match_devices(
    ggml: &[GgmlVulkanDevice<'_>],
    probed: &[VulkanDriverInfo],
) -> Vec<Option<usize>> {
    let probed_pci: Vec<Option<String>> = probed
        .iter()
        .map(|info| info.pci_bus.map(|pci| pci.to_string()))
        .collect();
    let mut matched: Vec<Option<usize>> = ggml
        .iter()
        .map(|device| {
            let id = device.pci_id?;
            let same_id = |other: &GgmlVulkanDevice<'_>| {
                other
                    .pci_id
                    .is_some_and(|other| other.eq_ignore_ascii_case(id))
            };
            if ggml.iter().filter(|other| same_id(other)).count() != 1 {
                return None;
            }
            let mut found = probed_pci
                .iter()
                .enumerate()
                .filter(|(_, pci)| {
                    pci.as_deref()
                        .is_some_and(|pci| pci.eq_ignore_ascii_case(id))
                })
                .map(|(index, _)| index);
            match (found.next(), found.next()) {
                (Some(index), None) => Some(index),
                _ => None,
            }
        })
        .collect();

    let claimed: Vec<usize> = matched.iter().flatten().copied().collect();
    let unresolved: Vec<usize> = matched
        .iter()
        .enumerate()
        .filter(|(_, slot)| slot.is_none())
        .map(|(index, _)| index)
        .collect();
    for &position in &unresolved {
        let Some(device) = ggml.get(position) else {
            continue;
        };
        let twins = unresolved
            .iter()
            .filter(|&&other| {
                ggml.get(other)
                    .is_some_and(|other| other.description == device.description)
            })
            .count();
        if twins != 1 {
            continue;
        }
        let mut candidates = probed.iter().enumerate().filter(|(index, info)| {
            !claimed.contains(index)
                && info.device_name == device.description
                && (device.pci_id.is_none() || info.pci_bus.is_none())
        });
        if let (Some((index, _)), None) = (candidates.next(), candidates.next())
            && let Some(slot) = matched.get_mut(position)
        {
            *slot = Some(index);
        }
    }
    matched
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn probed(name: &str, pci: Option<(u32, u32, u32, u32)>) -> VulkanDriverInfo {
        VulkanDriverInfo {
            device_name: name.to_owned(),
            device_type: VulkanDeviceType::DiscreteGpu,
            vendor_id: 0x10de,
            device_id: 0x1e84,
            api_version: VulkanApiVersion::from_raw(0x0040_4138),
            pci_bus: pci.map(|(domain, bus, device, function)| PciBusId {
                domain,
                bus,
                device,
                function,
            }),
            driver: Some(VulkanDriverIdentity {
                id: VulkanDriverId::NvidiaProprietary,
                name: "NVIDIA".to_owned(),
                info: "581.57".to_owned(),
            }),
        }
    }

    fn ggml<'a>(pci_id: Option<&'a str>, description: &'a str) -> GgmlVulkanDevice<'a> {
        GgmlVulkanDevice {
            pci_id,
            description,
        }
    }

    #[test]
    fn the_api_version_is_decoded_as_the_specification_says() {
        assert_eq!(
            VulkanApiVersion::from_raw(0x0040_4162).to_string(),
            "1.4.354"
        );
        assert_eq!(
            VulkanApiVersion::from_raw(0x2040_3007),
            VulkanApiVersion {
                major: 1,
                minor: 3,
                patch: 7
            }
        );
    }

    #[test]
    fn a_pci_bus_id_is_written_the_way_ggml_writes_it() {
        assert_eq!(
            PciBusId {
                domain: 0,
                bus: 0x0a,
                device: 0,
                function: 1
            }
            .to_string(),
            "0000:0a:00.1"
        );
    }

    #[test]
    fn devices_are_tied_by_pci_bus_id_whatever_the_order() {
        let probed = [
            probed("Same Name", Some((0, 1, 0, 0))),
            probed("Same Name", Some((0, 6, 0, 0))),
        ];
        let first = [
            ggml(Some("0000:06:00.0"), "Same Name"),
            ggml(Some("0000:01:00.0"), "Same Name"),
        ];
        assert_eq!(match_devices(&first, &probed), [Some(1), Some(0)]);
        let reversed = [first[1], first[0]];
        assert_eq!(match_devices(&reversed, &probed), [Some(0), Some(1)]);
    }

    #[test]
    fn a_pci_bus_id_is_compared_without_regard_to_case() {
        let probed = [probed("GPU", Some((0, 0xab, 0, 0)))];
        assert_eq!(
            match_devices(&[ggml(Some("0000:AB:00.0"), "GPU")], &probed),
            [Some(0)]
        );
    }

    #[test]
    fn without_a_pci_bus_id_a_unique_name_ties_the_devices() {
        let probed = [probed("Other", None), probed("Intel(R) Graphics", None)];
        assert_eq!(
            match_devices(&[ggml(None, "Intel(R) Graphics")], &probed),
            [Some(1)]
        );
    }

    #[test]
    fn a_name_shared_by_several_devices_ties_none_of_them() {
        let twins = [probed("Twin", None), probed("Twin", None)];
        assert_eq!(match_devices(&[ggml(None, "Twin")], &twins), [None]);
        assert_eq!(
            match_devices(&[ggml(None, "Twin"), ggml(None, "Twin")], &twins[..1]),
            [None, None]
        );
    }

    #[test]
    fn pci_bus_ids_that_differ_are_not_overruled_by_a_shared_name() {
        let probed = [probed("GPU", Some((0, 1, 0, 0)))];
        assert_eq!(
            match_devices(&[ggml(Some("0000:02:00.0"), "GPU")], &probed),
            [None]
        );
    }

    #[test]
    fn a_device_already_tied_by_pci_is_not_tied_again_by_name() {
        let probed = [probed("GPU", Some((0, 1, 0, 0)))];
        assert_eq!(
            match_devices(
                &[ggml(Some("0000:01:00.0"), "GPU"), ggml(None, "GPU")],
                &probed
            ),
            [Some(0), None]
        );
    }

    #[test]
    fn a_device_the_driver_did_not_list_stays_unmatched() {
        assert_eq!(match_devices(&[ggml(None, "Ghost")], &[]), [None]);
    }

    #[test]
    fn the_driver_is_shown_as_it_reports_itself_next_to_the_api_version() {
        let status = VulkanDriverStatus::Reported(probed("GPU", None));
        assert_eq!(
            status.fields(),
            [
                ("Driver", "NVIDIA 581.57".to_owned()),
                ("Vulkan API", "1.4.312".to_owned())
            ]
        );
    }

    #[test]
    fn a_driver_without_version_info_says_so_instead_of_inventing_one() {
        let mut info = probed("GPU", None);
        info.driver = Some(VulkanDriverIdentity {
            id: VulkanDriverId::Unlisted(99),
            name: "Odd driver ".to_owned(),
            info: String::new(),
        });
        assert_eq!(
            VulkanDriverStatus::Reported(info).fields().first(),
            Some(&(
                "Driver",
                "Odd driver  (version not reported by the driver)".to_owned()
            ))
        );
    }

    #[test]
    fn a_device_whose_driver_lacks_driver_properties_says_so() {
        let mut info = probed("GPU", None);
        info.driver = None;
        assert_eq!(
            VulkanDriverStatus::Reported(info).fields().first(),
            Some(&("Driver", "not reported by the Vulkan driver".to_owned()))
        );
    }

    #[test]
    fn an_unreadable_driver_is_reported_with_the_reason() {
        assert_eq!(
            VulkanDriverStatus::Unavailable(VulkanDriverUnavailable::Probe(
                VulkanProbeError::LoaderUnavailable
            ))
            .fields(),
            [(
                "Driver",
                "could not be read (the Vulkan loader could not be opened)".to_owned()
            )]
        );
        assert_eq!(
            VulkanDriverStatus::Unavailable(VulkanDriverUnavailable::NoUnambiguousMatch).fields(),
            [(
                "Driver",
                "could not be read (no single Vulkan device matches this one by PCI bus id or name)"
                    .to_owned()
            )]
        );
    }

    #[test]
    fn the_json_shape_of_a_reported_driver() {
        let status = VulkanDriverStatus::Reported(probed("RTX", Some((0, 1, 0, 0))));
        assert_eq!(
            serde_json::to_value(&status).expect("serializes"),
            json!({
                "Reported": {
                    "device_name": "RTX",
                    "device_type": "DiscreteGpu",
                    "vendor_id": 4318,
                    "device_id": 7812,
                    "api_version": { "major": 1, "minor": 4, "patch": 312 },
                    "pci_bus": { "domain": 0, "bus": 1, "device": 0, "function": 0 },
                    "driver": {
                        "id": "NvidiaProprietary",
                        "name": "NVIDIA",
                        "info": "581.57"
                    }
                }
            })
        );
    }

    #[test]
    fn the_json_shape_of_an_unavailable_driver() {
        assert_eq!(
            serde_json::to_value(VulkanDriverStatus::Unavailable(
                VulkanDriverUnavailable::Probe(VulkanProbeError::Call {
                    call: VulkanCall::CreateInstance,
                    vk_result: -9
                })
            ))
            .expect("serializes"),
            json!({
                "Unavailable": {
                    "Probe": { "Call": { "call": "CreateInstance", "vk_result": -9 } }
                }
            })
        );
    }
}
