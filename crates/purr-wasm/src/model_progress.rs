//! Aggregates per-file download progress into one figure for a model.

use purr_common::web_model::registry::WebModelSpec;

#[derive(Debug)]
struct FileProgress {
    path: String,
    size_bytes: u64,
    loaded_bytes: u64,
}

/// Download progress of every file of a model, measured against the manifest sizes.
#[derive(Debug)]
pub struct ModelProgress {
    files: Vec<FileProgress>,
}

impl ModelProgress {
    #[must_use]
    pub fn for_model(spec: &WebModelSpec) -> Self {
        Self {
            files: spec
                .files
                .iter()
                .map(|file| FileProgress {
                    path: file.path.clone(),
                    size_bytes: file.size_bytes,
                    loaded_bytes: 0,
                })
                .collect(),
        }
    }

    /// Records the bytes received so far for `path`. Files outside the manifest are ignored.
    pub fn record(&mut self, path: &str, loaded_bytes: u64) {
        if let Some(file) = self.files.iter_mut().find(|file| file.path == path) {
            file.loaded_bytes = loaded_bytes.min(file.size_bytes);
        }
    }

    pub fn finish(&mut self, path: &str) {
        if let Some(file) = self.files.iter_mut().find(|file| file.path == path) {
            file.loaded_bytes = file.size_bytes;
        }
    }

    #[must_use]
    pub fn loaded_bytes(&self) -> u64 {
        self.files.iter().map(|file| file.loaded_bytes).sum()
    }

    #[must_use]
    pub fn total_bytes(&self) -> u64 {
        self.files.iter().map(|file| file.size_bytes).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use purr_common::web_model::registry::WebModelRegistry;

    fn tiny_progress() -> Result<ModelProgress, Box<dyn std::error::Error>> {
        let registry = WebModelRegistry::embedded()?;
        Ok(ModelProgress::for_model(registry.get("tiny")?))
    }

    #[test]
    fn total_matches_the_manifest() -> Result<(), Box<dyn std::error::Error>> {
        let registry = WebModelRegistry::embedded()?;
        let progress = tiny_progress()?;
        assert_eq!(
            progress.total_bytes(),
            registry.get("tiny")?.total_size_bytes()
        );
        assert_eq!(progress.loaded_bytes(), 0);
        Ok(())
    }

    #[test]
    fn files_sum_and_never_exceed_their_size() -> Result<(), Box<dyn std::error::Error>> {
        let mut progress = tiny_progress()?;
        progress.record("config.json", 100);
        progress.record("onnx/encoder_model.onnx", 1_000);
        assert_eq!(progress.loaded_bytes(), 1_100);
        progress.record("config.json", u64::MAX);
        assert_eq!(progress.loaded_bytes(), 2_243 + 1_000);
        Ok(())
    }

    #[test]
    fn finishing_a_file_counts_all_of_it() -> Result<(), Box<dyn std::error::Error>> {
        let mut progress = tiny_progress()?;
        progress.finish("onnx/encoder_model.onnx");
        assert_eq!(progress.loaded_bytes(), 32_904_992);
        Ok(())
    }

    #[test]
    fn unknown_files_are_ignored() -> Result<(), Box<dyn std::error::Error>> {
        let mut progress = tiny_progress()?;
        progress.record("not-in-the-manifest.bin", 500);
        progress.finish("not-in-the-manifest.bin");
        assert_eq!(progress.loaded_bytes(), 0);
        Ok(())
    }
}
