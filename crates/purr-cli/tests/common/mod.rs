use purr_core::ModelManager;
use std::{
    fmt,
    path::{Path, PathBuf},
};

pub type TestResult<T = ()> = Result<T, Box<dyn std::error::Error>>;

struct MissingBaseModel {
    dir: PathBuf,
}

impl fmt::Debug for MissingBaseModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl fmt::Display for MissingBaseModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "the `base` model is missing or incomplete in {}: run `purr models download base` or `just download-test-models`",
            self.dir.display()
        )
    }
}

impl std::error::Error for MissingBaseModel {}

pub async fn base_model() -> TestResult<PathBuf> {
    let models = ModelManager::new()?;
    let spec = models.model("base")?;
    if !models.is_model_downloaded(spec).await? {
        return Err(MissingBaseModel {
            dir: models.models_dir().to_path_buf(),
        }
        .into());
    }
    Ok(models.get_model_path(spec))
}

pub fn golden_path(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/golden")
        .join(name)
}
