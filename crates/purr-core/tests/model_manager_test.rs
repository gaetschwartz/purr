use purr_core::ModelManager;
use purr_core::model::status::{ModelFileStatus, ModelIntegrity};
use std::error::Error;
use tempfile::TempDir;

mod common;
use common::{digest_of, fixture_bytes, fixture_registry};

struct Fixture {
    _dir: TempDir,
    manager: ModelManager,
    content: Vec<u8>,
}

fn fixture() -> Result<Fixture, Box<dyn Error>> {
    let content = fixture_bytes();
    let dir = tempfile::tempdir()?;
    let manager =
        ModelManager::with_registry(dir.path().to_path_buf(), fixture_registry(&content)?);
    Ok(Fixture {
        _dir: dir,
        manager,
        content,
    })
}

async fn downloaded_ids(manager: &ModelManager) -> Result<Vec<String>, Box<dyn Error>> {
    Ok(manager
        .list_downloaded_models()
        .await?
        .into_iter()
        .map(|model| model.id.clone())
        .collect())
}

#[tokio::test]
async fn absent_file_is_missing() -> Result<(), Box<dyn Error>> {
    let fx = fixture()?;
    let model = fx.manager.model("fixture")?;

    assert_eq!(
        fx.manager.file_status(model).await?,
        ModelFileStatus::Missing
    );
    assert!(!fx.manager.is_model_downloaded(model).await?);
    assert_eq!(
        fx.manager.verify_model(model).await?,
        ModelIntegrity::Missing
    );
    assert_eq!(fx.manager.find_first_available_model().await?, None);
    assert_eq!(downloaded_ids(&fx.manager).await?, Vec::<String>::new());
    Ok(())
}

#[tokio::test]
async fn file_with_the_manifest_size_counts_as_downloaded_and_verifies()
-> Result<(), Box<dyn Error>> {
    let fx = fixture()?;
    let model = fx.manager.model("fixture")?;
    let path = fx.manager.get_model_path(model);
    std::fs::write(&path, &fx.content)?;

    assert_eq!(
        fx.manager.file_status(model).await?,
        ModelFileStatus::Present
    );
    assert!(fx.manager.is_model_downloaded(model).await?);
    assert_eq!(
        fx.manager.verify_model(model).await?,
        ModelIntegrity::Verified
    );
    assert_eq!(fx.manager.find_first_available_model().await?, Some(path));
    assert_eq!(downloaded_ids(&fx.manager).await?, ["fixture"]);
    Ok(())
}

#[tokio::test]
async fn file_with_the_wrong_size_is_not_downloaded() -> Result<(), Box<dyn Error>> {
    let fx = fixture()?;
    let model = fx.manager.model("fixture")?;
    let truncated = &fx.content[..fx.content.len() - 10];
    std::fs::write(fx.manager.get_model_path(model), truncated)?;

    let mismatch = ModelFileStatus::SizeMismatch {
        expected: fx.content.len() as u64,
        actual: truncated.len() as u64,
    };
    assert_eq!(fx.manager.file_status(model).await?, mismatch);
    assert!(!fx.manager.is_model_downloaded(model).await?);
    assert_eq!(fx.manager.find_first_available_model().await?, None);
    assert_eq!(downloaded_ids(&fx.manager).await?, Vec::<String>::new());
    let local = fx.manager.local_models().await?;
    assert_eq!(
        local
            .iter()
            .map(|(m, status)| (m.id.as_str(), *status))
            .collect::<Vec<_>>(),
        [("fixture", mismatch)]
    );
    assert_eq!(
        fx.manager.verify_model(model).await?,
        ModelIntegrity::SizeMismatch {
            expected: fx.content.len() as u64,
            actual: truncated.len() as u64,
        }
    );
    Ok(())
}

#[tokio::test]
async fn same_size_corruption_is_found_only_by_full_verification() -> Result<(), Box<dyn Error>> {
    let fx = fixture()?;
    let model = fx.manager.model("fixture")?;
    let mut corrupt = fx.content.clone();
    if let Some(byte) = corrupt.last_mut() {
        *byte ^= 0xFF;
    }
    std::fs::write(fx.manager.get_model_path(model), &corrupt)?;

    assert!(fx.manager.is_model_downloaded(model).await?);
    assert_eq!(
        fx.manager.verify_model(model).await?,
        ModelIntegrity::ChecksumMismatch {
            expected: digest_of(&fx.content),
            actual: digest_of(&corrupt),
        }
    );
    Ok(())
}

#[tokio::test]
async fn delete_removes_the_file_and_tolerates_absence() -> Result<(), Box<dyn Error>> {
    let fx = fixture()?;
    let model = fx.manager.model("fixture")?;
    let path = fx.manager.get_model_path(model);
    std::fs::write(&path, &fx.content)?;

    fx.manager.delete_model(model).await?;
    assert!(!path.exists());
    fx.manager.delete_model(model).await?;
    Ok(())
}

#[tokio::test]
async fn unknown_model_id_is_a_typed_error() -> Result<(), Box<dyn Error>> {
    let fx = fixture()?;
    let result = fx.manager.model("nonexistent");

    assert!(matches!(
        result,
        Err(purr_core::WhisperError::Configuration {
            source: purr_core::error::ConfigurationError::UnknownModel(ref unknown)
        }) if unknown.model_name == "nonexistent"
    ));
    Ok(())
}
