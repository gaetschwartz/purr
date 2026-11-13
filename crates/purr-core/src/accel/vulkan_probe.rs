//! Asks the Vulkan driver what it is, since ggml's device properties do not carry the driver.

use std::{ffi::CStr, sync::OnceLock};

use ash::{Entry, Instance, vk};
use purr_common::vulkan::{
    GgmlVulkanDevice, PciBusId, VulkanApiVersion, VulkanCall, VulkanDeviceType, VulkanDriverId,
    VulkanDriverIdentity, VulkanDriverInfo, VulkanDriverStatus, VulkanDriverUnavailable,
    VulkanProbeError, match_devices,
};

use super::{
    backend::AccelBackend,
    library_loader::{SystemLoader, open_all},
    runtime_dir::{Environment, search_dirs},
    runtime_library::required,
};
use crate::dev::Device;

/// Fill in the Vulkan driver of every ggml Vulkan device.
///
/// ggml registers a Vulkan device only when the backend is built and its loader opens, so
/// nothing happens, and no Vulkan instance is created, unless there is such a device.
pub fn attach_drivers(devices: &mut [Device]) {
    let vulkan: Vec<usize> = devices
        .iter()
        .enumerate()
        .filter(|(_, device)| device.backend == Some(AccelBackend::Vulkan))
        .map(|(index, _)| index)
        .collect();
    if vulkan.is_empty() {
        return;
    }
    let statuses = match probe() {
        Ok(probed) => {
            let ggml: Vec<GgmlVulkanDevice<'_>> = vulkan
                .iter()
                .filter_map(|index| devices.get(*index))
                .map(|device| GgmlVulkanDevice {
                    pci_id: device.device_id.as_deref(),
                    description: &device.description,
                })
                .collect();
            match_devices(&ggml, &probed)
                .into_iter()
                .map(|matched| {
                    matched.and_then(|index| probed.get(index)).map_or(
                        VulkanDriverStatus::Unavailable(
                            VulkanDriverUnavailable::NoUnambiguousMatch,
                        ),
                        |info| VulkanDriverStatus::Reported(info.clone()),
                    )
                })
                .collect::<Vec<_>>()
        }
        Err(error) => vulkan
            .iter()
            .map(|_| VulkanDriverStatus::Unavailable(VulkanDriverUnavailable::Probe(error)))
            .collect(),
    };
    for (index, status) in vulkan.into_iter().zip(statuses) {
        if let Some(device) = devices.get_mut(index) {
            device.vulkan_driver = Some(status);
        }
    }
}

/// Every Vulkan physical device the loader exposes, as its driver describes it.
pub fn probe() -> Result<Vec<VulkanDriverInfo>, VulkanProbeError> {
    let entry = entry()?;
    let loader_version = unsafe { entry.try_enumerate_instance_version() }
        .map_err(|result| call_failed(VulkanCall::EnumerateInstanceVersion, result))?
        .ok_or(VulkanProbeError::LoaderBeforeVulkan11)?;
    let api_version = loader_version.min(vk::API_VERSION_1_2);

    let application = vk::ApplicationInfo::default().api_version(api_version);
    let create_info = vk::InstanceCreateInfo::default().application_info(&application);
    let instance = unsafe { entry.create_instance(&create_info, None) }
        .map_err(|result| call_failed(VulkanCall::CreateInstance, result))?;
    let instance = InstanceGuard(instance);

    let devices = unsafe { instance.0.enumerate_physical_devices() }
        .map_err(|result| call_failed(VulkanCall::EnumeratePhysicalDevices, result))?;
    let mut described = Vec::with_capacity(devices.len());
    for device in devices {
        if let Some(info) = describe(&instance.0, device, api_version)? {
            described.push(info);
        }
    }
    Ok(described)
}

/// The loader stays mapped for the process: the drivers it pulled in must not outlive it.
fn entry() -> Result<&'static Entry, VulkanProbeError> {
    static ENTRY: OnceLock<Option<Entry>> = OnceLock::new();
    ENTRY
        .get_or_init(|| {
            let dirs: Vec<_> = search_dirs(&Environment::current())
                .into_iter()
                .map(|dir| dir.path)
                .collect();
            open_all(&SystemLoader, &required(AccelBackend::Vulkan), &dirs).ok()?;
            unsafe { Entry::load() }.ok()
        })
        .as_ref()
        .ok_or(VulkanProbeError::LoaderUnavailable)
}

struct InstanceGuard(Instance);

impl Drop for InstanceGuard {
    fn drop(&mut self) {
        unsafe { self.0.destroy_instance(None) };
    }
}

fn call_failed(call: VulkanCall, result: vk::Result) -> VulkanProbeError {
    VulkanProbeError::Call {
        call,
        vk_result: result.as_raw(),
    }
}

fn has_extension(extensions: &[vk::ExtensionProperties], name: &CStr) -> bool {
    extensions
        .iter()
        .any(|extension| extension.extension_name_as_c_str() == Ok(name))
}

fn text(chars: Result<&CStr, core::ffi::FromBytesUntilNulError>) -> Option<String> {
    chars.ok().map(|chars| chars.to_string_lossy().into_owned())
}

/// `None` when the driver gives the device no readable name.
fn describe(
    instance: &Instance,
    device: vk::PhysicalDevice,
    instance_version: u32,
) -> Result<Option<VulkanDriverInfo>, VulkanProbeError> {
    let properties = unsafe { instance.get_physical_device_properties(device) };
    let extensions = unsafe { instance.enumerate_device_extension_properties(device) }
        .map_err(|result| call_failed(VulkanCall::EnumerateDeviceExtensions, result))?;
    let driver_in_core =
        instance_version >= vk::API_VERSION_1_2 && properties.api_version >= vk::API_VERSION_1_2;
    let has_driver =
        driver_in_core || has_extension(&extensions, ash::khr::driver_properties::NAME);
    let has_pci = has_extension(&extensions, ash::ext::pci_bus_info::NAME);

    let mut driver = vk::PhysicalDeviceDriverProperties::default();
    let mut pci = vk::PhysicalDevicePCIBusInfoPropertiesEXT::default();
    let mut chain = vk::PhysicalDeviceProperties2::default();
    if has_driver {
        chain = chain.push_next(&mut driver);
    }
    if has_pci {
        chain = chain.push_next(&mut pci);
    }
    unsafe { instance.get_physical_device_properties2(device, &mut chain) };

    let Some(device_name) = text(properties.device_name_as_c_str()) else {
        return Ok(None);
    };
    Ok(Some(VulkanDriverInfo {
        device_name,
        device_type: device_type(properties.device_type),
        vendor_id: properties.vendor_id,
        device_id: properties.device_id,
        api_version: VulkanApiVersion::from_raw(properties.api_version),
        pci_bus: has_pci.then_some(PciBusId {
            domain: pci.pci_domain,
            bus: pci.pci_bus,
            device: pci.pci_device,
            function: pci.pci_function,
        }),
        driver: if has_driver {
            text(driver.driver_name_as_c_str())
                .zip(text(driver.driver_info_as_c_str()))
                .map(|(name, info)| VulkanDriverIdentity {
                    id: driver_id(driver.driver_id),
                    name,
                    info,
                })
        } else {
            None
        },
    }))
}

fn device_type(raw: vk::PhysicalDeviceType) -> VulkanDeviceType {
    match raw {
        vk::PhysicalDeviceType::OTHER => VulkanDeviceType::Other,
        vk::PhysicalDeviceType::INTEGRATED_GPU => VulkanDeviceType::IntegratedGpu,
        vk::PhysicalDeviceType::DISCRETE_GPU => VulkanDeviceType::DiscreteGpu,
        vk::PhysicalDeviceType::VIRTUAL_GPU => VulkanDeviceType::VirtualGpu,
        vk::PhysicalDeviceType::CPU => VulkanDeviceType::Cpu,
        _ => VulkanDeviceType::Unknown,
    }
}

fn driver_id(raw: vk::DriverId) -> VulkanDriverId {
    match raw {
        vk::DriverId::AMD_PROPRIETARY => VulkanDriverId::AmdProprietary,
        vk::DriverId::AMD_OPEN_SOURCE => VulkanDriverId::AmdOpenSource,
        vk::DriverId::MESA_RADV => VulkanDriverId::MesaRadv,
        vk::DriverId::NVIDIA_PROPRIETARY => VulkanDriverId::NvidiaProprietary,
        vk::DriverId::INTEL_PROPRIETARY_WINDOWS => VulkanDriverId::IntelProprietaryWindows,
        vk::DriverId::INTEL_OPEN_SOURCE_MESA => VulkanDriverId::IntelOpenSourceMesa,
        vk::DriverId::IMAGINATION_PROPRIETARY => VulkanDriverId::ImaginationProprietary,
        vk::DriverId::QUALCOMM_PROPRIETARY => VulkanDriverId::QualcommProprietary,
        vk::DriverId::ARM_PROPRIETARY => VulkanDriverId::ArmProprietary,
        vk::DriverId::GOOGLE_SWIFTSHADER => VulkanDriverId::GoogleSwiftshader,
        vk::DriverId::GGP_PROPRIETARY => VulkanDriverId::GgpProprietary,
        vk::DriverId::BROADCOM_PROPRIETARY => VulkanDriverId::BroadcomProprietary,
        vk::DriverId::MESA_LLVMPIPE => VulkanDriverId::MesaLlvmpipe,
        vk::DriverId::MOLTENVK => VulkanDriverId::MoltenVk,
        vk::DriverId::COREAVI_PROPRIETARY => VulkanDriverId::CoreAviProprietary,
        vk::DriverId::JUICE_PROPRIETARY => VulkanDriverId::JuiceProprietary,
        vk::DriverId::VERISILICON_PROPRIETARY => VulkanDriverId::VerisiliconProprietary,
        vk::DriverId::MESA_TURNIP => VulkanDriverId::MesaTurnip,
        vk::DriverId::MESA_V3DV => VulkanDriverId::MesaV3dv,
        vk::DriverId::MESA_PANVK => VulkanDriverId::MesaPanvk,
        vk::DriverId::SAMSUNG_PROPRIETARY => VulkanDriverId::SamsungProprietary,
        vk::DriverId::MESA_VENUS => VulkanDriverId::MesaVenus,
        vk::DriverId::MESA_DOZEN => VulkanDriverId::MesaDozen,
        vk::DriverId::MESA_NVK => VulkanDriverId::MesaNvk,
        vk::DriverId::IMAGINATION_OPEN_SOURCE_MESA => VulkanDriverId::ImaginationOpenSourceMesa,
        vk::DriverId::MESA_AGXV => VulkanDriverId::MesaAgxv,
        other => VulkanDriverId::Unlisted(other.as_raw()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "needs a Vulkan loader and driver on this machine"]
    fn print_this_machines_vulkan_devices() {
        let probed = probe().expect("the Vulkan driver answers");
        let pci_ids: Vec<Option<String>> = probed
            .iter()
            .map(|info| info.pci_bus.map(|pci| pci.to_string()))
            .collect();
        let ggml: Vec<GgmlVulkanDevice<'_>> = probed
            .iter()
            .zip(&pci_ids)
            .map(|(info, pci)| GgmlVulkanDevice {
                pci_id: pci.as_deref(),
                description: &info.device_name,
            })
            .collect();
        let matched = match_devices(&ggml, &probed);
        for (info, index) in probed.iter().zip(&matched) {
            println!("{}", info.device_name);
            for (label, value) in VulkanDriverStatus::Reported(info.clone()).fields() {
                println!("  {label}: {value}");
            }
            println!("  matched to itself: {}", index.is_some());
        }
        assert_eq!(matched, (0..probed.len()).map(Some).collect::<Vec<_>>());
    }
}
