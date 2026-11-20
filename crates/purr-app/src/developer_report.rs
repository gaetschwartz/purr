//! The developer panel as sections of labelled rows, shown on screen and copied as text.

use crate::utils::byte_size::ByteSize;
use purr_common::{
    developer::snapshot::{
        AdapterReading, BrowserSnapshot, DeveloperSnapshot, ExecutionProvider, JsHeapReading,
        NativeSnapshot, Peaked,
    },
    platform::DeviceType,
};

const NOT_AVAILABLE: &str = "not available";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Row {
    pub label: String,
    pub value: String,
}

impl Row {
    fn new(label: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            value: value.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Section {
    pub title: String,
    pub rows: Vec<Row>,
}

impl Section {
    fn new(title: impl Into<String>, rows: Vec<Row>) -> Self {
        Self {
            title: title.into(),
            rows,
        }
    }
}

fn peaked(label: &str, reading: Option<Peaked>) -> Row {
    Row::new(
        label,
        reading.map_or_else(
            || NOT_AVAILABLE.to_owned(),
            |reading| {
                format!(
                    "{} (peak {})",
                    ByteSize(reading.current),
                    ByteSize(reading.peak)
                )
            },
        ),
    )
}

fn count(label: &str, reading: Peaked) -> Row {
    Row::new(
        label,
        format!("{} (peak {})", reading.current, reading.peak),
    )
}

fn js_heap(label: &str, reading: Option<JsHeapReading>) -> Vec<Row> {
    let Some(reading) = reading else {
        return vec![Row::new(label, NOT_AVAILABLE)];
    };
    vec![
        peaked(label, Some(reading.used)),
        Row::new(
            format!("{label} limit"),
            ByteSize(reading.limit).to_string(),
        ),
    ]
}

const fn provider_label(provider: ExecutionProvider) -> &'static str {
    match provider {
        ExecutionProvider::WebGpu => "WebGPU",
        ExecutionProvider::Cpu => "CPU (WebAssembly)",
    }
}

fn adapter_sections(adapter: &AdapterReading) -> Vec<Section> {
    let info = adapter
        .info
        .iter()
        .map(|field| Row::new(field.name.clone(), field.value.clone()))
        .collect();
    let features = adapter
        .features
        .iter()
        .map(|feature| {
            let value = match (feature.supported, feature.known) {
                (true, true) => "yes",
                (false, _) => "no",
                (true, false) => "yes (not in the known list)",
            };
            Row::new(feature.name.clone(), value)
        })
        .collect();
    let limits = adapter
        .limits
        .iter()
        .map(|field| Row::new(field.name.clone(), field.value.clone()))
        .collect();
    vec![
        Section::new("WebGPU adapter", info),
        Section::new("WebGPU features", features),
        Section::new("WebGPU limits", limits),
    ]
}

fn browser_sections(snapshot: &BrowserSnapshot) -> Vec<Section> {
    let worker = &snapshot.worker;
    let mut run = vec![
        Row::new(
            "Execution provider",
            worker.provider.map_or(NOT_AVAILABLE, provider_label),
        ),
        Row::new(
            "Model",
            worker.model.as_deref().unwrap_or("no model loaded yet"),
        ),
    ];
    if !snapshot.collecting {
        run.push(Row::new(
            "Collection",
            "the worker started before developer mode was on; it restarts when idle",
        ));
    }
    let mut memory = vec![
        peaked("Transcription worker WebAssembly heap", worker.wasm_heap),
        peaked("Converter worker WebAssembly heap", snapshot.converter_heap),
        peaked("GPU buffers allocated", Some(worker.gpu_buffers.bytes)),
        count("GPU buffers live", worker.gpu_buffers.live),
    ];
    memory.extend(js_heap("Worker JS heap", worker.js_heap));
    memory.extend(js_heap("Page JS heap", snapshot.page_js_heap));
    let mut sections = vec![Section::new("Run", run), Section::new("Memory", memory)];
    match &worker.adapter {
        Some(adapter) => sections.extend(adapter_sections(adapter)),
        None => sections.push(Section::new(
            "WebGPU adapter",
            vec![Row::new("Adapter", NOT_AVAILABLE)],
        )),
    }
    sections
}

const fn device_type_label(device_type: DeviceType) -> &'static str {
    match device_type {
        DeviceType::Cpu => "CPU",
        DeviceType::Gpu => "GPU",
        DeviceType::Igpu => "Integrated GPU",
        DeviceType::Accel => "Accelerator",
        DeviceType::Unknown => "Unknown type",
    }
}

fn native_sections(snapshot: &NativeSnapshot) -> Vec<Section> {
    let run = snapshot.last_run.as_ref().map_or_else(
        || vec![Row::new("Last run", "no transcription yet")],
        |last| {
            vec![
                Row::new("Backend", last.backend.label()),
                Row::new("Device", last.device.as_deref().unwrap_or("the processor")),
            ]
        },
    );
    let memory = vec![
        peaked("Process working set", snapshot.process_memory),
        peaked("Process virtual memory", snapshot.process_virtual),
    ];
    let mut sections = vec![
        Section::new("Run", run),
        Section::new("Process memory", memory),
    ];
    for device in &snapshot.devices {
        let mut rows = vec![
            Row::new("Engine", device.engine.label()),
            Row::new("Description", device.description.clone()),
            Row::new("Type", device_type_label(device.device_type)),
            Row::new(
                "Backend",
                device
                    .backend
                    .map_or("not classified", |backend| backend.label()),
            ),
            Row::new(
                "Device id",
                device.device_id.as_deref().unwrap_or(NOT_AVAILABLE),
            ),
        ];
        rows.push(Row::new(
            "Device memory free",
            device.memory_free.map_or_else(
                || NOT_AVAILABLE.to_owned(),
                |bytes| ByteSize(bytes).to_string(),
            ),
        ));
        rows.push(Row::new(
            "Device memory total",
            device.memory_total.map_or_else(
                || NOT_AVAILABLE.to_owned(),
                |bytes| ByteSize(bytes).to_string(),
            ),
        ));
        sections.push(Section::new(device.name.clone(), rows));
    }
    if snapshot.devices.is_empty() {
        sections.push(Section::new(
            "Devices",
            vec![Row::new("Engines", "no engine reports a device")],
        ));
    }
    sections
}

/// The panel's content for a snapshot.
#[must_use]
pub fn sections(snapshot: &DeveloperSnapshot) -> Vec<Section> {
    match snapshot {
        DeveloperSnapshot::Browser(browser) => browser_sections(browser),
        DeveloperSnapshot::Native(native) => native_sections(native),
    }
}

/// The same content as plain text, for the clipboard.
#[must_use]
pub fn as_text(sections: &[Section]) -> String {
    sections
        .iter()
        .map(|section| {
            let rows: String = section
                .rows
                .iter()
                .map(|row| format!("  {}: {}\n", row.label, row.value))
                .collect();
            format!("{}\n{rows}\n", section.title)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use purr_common::developer::{
        snapshot::{Field, GpuBuffers, WorkerReading},
        webgpu::feature_rows,
    };

    fn browser(adapter: Option<AdapterReading>, wasm_heap: Option<Peaked>) -> DeveloperSnapshot {
        DeveloperSnapshot::Browser(BrowserSnapshot {
            collecting: true,
            worker: WorkerReading {
                wasm_heap,
                gpu_buffers: GpuBuffers {
                    bytes: Peaked {
                        current: 1024,
                        peak: 4096,
                    },
                    live: Peaked {
                        current: 3,
                        peak: 5,
                    },
                },
                js_heap: None,
                provider: Some(ExecutionProvider::WebGpu),
                model: Some("tiny".to_owned()),
                adapter,
            },
            converter_heap: None,
            page_js_heap: None,
        })
    }

    fn row<'a>(sections: &'a [Section], title: &str, label: &str) -> Option<&'a Row> {
        sections
            .iter()
            .find(|section| section.title == title)?
            .rows
            .iter()
            .find(|row| row.label == label)
    }

    #[test]
    fn gpu_memory_is_labelled_as_buffers_allocated() {
        let sections = sections(&browser(None, None));
        let gpu = row(&sections, "Memory", "GPU buffers allocated").unwrap();
        assert_eq!(gpu.value, "1.0 KiB (peak 4.0 KiB)");
    }

    #[test]
    fn missing_figures_say_so() {
        let sections = sections(&browser(None, None));
        assert_eq!(
            row(&sections, "Memory", "Page JS heap").unwrap().value,
            NOT_AVAILABLE
        );
        assert_eq!(
            row(&sections, "Memory", "Transcription worker WebAssembly heap")
                .unwrap()
                .value,
            NOT_AVAILABLE
        );
    }

    #[test]
    fn every_known_feature_gets_a_row_with_its_support() {
        let adapter = AdapterReading {
            info: vec![Field {
                name: "vendor".to_owned(),
                value: "nvidia".to_owned(),
            }],
            features: feature_rows(&["shader-f16".to_owned()]),
            limits: Vec::new(),
        };
        let sections = sections(&browser(Some(adapter), None));
        assert_eq!(
            row(&sections, "WebGPU features", "shader-f16")
                .unwrap()
                .value,
            "yes"
        );
        assert_eq!(
            row(&sections, "WebGPU features", "subgroups")
                .unwrap()
                .value,
            "no"
        );
        assert_eq!(
            row(&sections, "WebGPU adapter", "vendor").unwrap().value,
            "nvidia"
        );
    }

    #[test]
    fn the_text_lists_each_section_with_its_rows() {
        let text = as_text(&[Section::new("Run", vec![Row::new("Model", "tiny")])]);
        assert_eq!(text, "Run\n  Model: tiny\n\n");
    }
}
