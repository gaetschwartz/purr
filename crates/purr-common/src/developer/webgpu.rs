use serde_with::{DeserializeFromStr, SerializeDisplay};
use strum::{Display, EnumString, VariantArray};

/// The features the WebGPU specification names for an adapter (`GPUFeatureName` of
/// `@webgpu/types` 0.1.69), in the order the specification lists them.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    Display,
    EnumString,
    VariantArray,
    SerializeDisplay,
    DeserializeFromStr,
)]
pub enum WebGpuFeature {
    #[strum(serialize = "core-features-and-limits")]
    CoreFeaturesAndLimits,
    #[strum(serialize = "depth-clip-control")]
    DepthClipControl,
    #[strum(serialize = "depth32float-stencil8")]
    Depth32FloatStencil8,
    #[strum(serialize = "texture-compression-bc")]
    TextureCompressionBc,
    #[strum(serialize = "texture-compression-bc-sliced-3d")]
    TextureCompressionBcSliced3d,
    #[strum(serialize = "texture-compression-etc2")]
    TextureCompressionEtc2,
    #[strum(serialize = "texture-compression-astc")]
    TextureCompressionAstc,
    #[strum(serialize = "texture-compression-astc-sliced-3d")]
    TextureCompressionAstcSliced3d,
    #[strum(serialize = "timestamp-query")]
    TimestampQuery,
    #[strum(serialize = "indirect-first-instance")]
    IndirectFirstInstance,
    #[strum(serialize = "shader-f16")]
    ShaderF16,
    #[strum(serialize = "rg11b10ufloat-renderable")]
    Rg11b10UfloatRenderable,
    #[strum(serialize = "bgra8unorm-storage")]
    Bgra8UnormStorage,
    #[strum(serialize = "float32-filterable")]
    Float32Filterable,
    #[strum(serialize = "float32-blendable")]
    Float32Blendable,
    #[strum(serialize = "clip-distances")]
    ClipDistances,
    #[strum(serialize = "dual-source-blending")]
    DualSourceBlending,
    #[strum(serialize = "subgroups")]
    Subgroups,
    #[strum(serialize = "texture-formats-tier1")]
    TextureFormatsTier1,
    #[strum(serialize = "texture-formats-tier2")]
    TextureFormatsTier2,
    #[strum(serialize = "primitive-index")]
    PrimitiveIndex,
    #[strum(serialize = "texture-component-swizzle")]
    TextureComponentSwizzle,
}

/// One feature row of the adapter table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeatureRow {
    pub name: String,
    pub supported: bool,
    /// Whether the specification names the feature; browsers add their own.
    pub known: bool,
}

/// Every known feature with whether the adapter has it, then the adapter's features the list
/// does not know.
#[must_use]
pub fn feature_rows(supported: &[String]) -> Vec<FeatureRow> {
    let mut rows: Vec<FeatureRow> = WebGpuFeature::VARIANTS
        .iter()
        .map(|feature| {
            let name = feature.to_string();
            FeatureRow {
                supported: supported.contains(&name),
                name,
                known: true,
            }
        })
        .collect();
    let mut extras: Vec<&String> = supported
        .iter()
        .filter(|name| name.parse::<WebGpuFeature>().is_err())
        .collect();
    extras.sort();
    rows.extend(extras.into_iter().map(|name| FeatureRow {
        name: name.clone(),
        supported: true,
        known: false,
    }));
    rows
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_features_come_first_and_extras_follow() {
        let rows = feature_rows(&[
            "shader-f16".to_owned(),
            "chromium-experimental-x".to_owned(),
        ]);
        assert_eq!(rows.len(), WebGpuFeature::VARIANTS.len() + 1);
        let f16 = rows.iter().find(|row| row.name == "shader-f16").unwrap();
        assert_eq!((f16.supported, f16.known), (true, true));
        let subgroups = rows.iter().find(|row| row.name == "subgroups").unwrap();
        assert!(!subgroups.supported);
        let last = rows.last().unwrap();
        assert_eq!(
            (last.name.as_str(), last.supported, last.known),
            ("chromium-experimental-x", true, false)
        );
    }
}
