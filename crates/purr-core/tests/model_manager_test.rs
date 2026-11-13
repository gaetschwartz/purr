use parking_lot::Mutex;
use purr_core::error::ConfigurationError;
use purr_core::model::status::{ModelFileStatus, ModelIntegrity};
use purr_core::model::vad_or_none;
use purr_core::{ModelManager, WhisperError};
use std::error::Error;
use std::path::PathBuf;
use std::sync::Arc;
use tempfile::TempDir;

mod common;
use common::{VAD_FILENAME, digest_of, fixture_bytes, fixture_registry, vad_fixture_bytes};

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
async fn verification_reports_non_decreasing_progress_ending_at_the_file_size()
-> Result<(), Box<dyn Error>> {
    let fx = fixture()?;
    let model = fx.manager.model("fixture")?;
    std::fs::write(fx.manager.get_model_path(model), &fx.content)?;
    let total = fx.content.len() as u64;
    let seen = Arc::new(Mutex::new(Vec::new()));

    let sink = Arc::clone(&seen);
    let integrity = fx
        .manager
        .verify_model_with_progress(model, move |hashed, of| {
            sink.lock().push((hashed, of));
        })
        .await?;

    assert_eq!(integrity, ModelIntegrity::Verified);
    let seen = seen.lock().clone();
    assert!(seen.windows(2).all(|pair| pair[0].0 <= pair[1].0));
    assert_eq!(seen.first(), Some(&(0, total)));
    assert_eq!(seen.last(), Some(&(total, total)));
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

#[tokio::test]
async fn absent_voice_activity_model_has_no_path() -> Result<(), Box<dyn Error>> {
    let fx = fixture()?;

    assert_eq!(fx.manager.vad_model_path().await?, None);
    assert_eq!(
        fx.manager.get_vad_path(),
        fx.manager.models_dir().join(VAD_FILENAME)
    );
    Ok(())
}

#[tokio::test]
async fn voice_activity_model_has_a_path_only_when_it_verifies() -> Result<(), Box<dyn Error>> {
    let fx = fixture()?;
    let path = fx.manager.get_vad_path();
    let content = vad_fixture_bytes();

    std::fs::write(&path, &content[..content.len() - 1])?;
    assert_eq!(fx.manager.vad_model_path().await?, None);

    let mut corrupt = content.clone();
    if let Some(byte) = corrupt.last_mut() {
        *byte ^= 0xFF;
    }
    std::fs::write(&path, &corrupt)?;
    assert_eq!(fx.manager.vad_model_path().await?, None);

    std::fs::write(&path, &content)?;
    assert_eq!(fx.manager.vad_model_path().await?, Some(path.clone()));
    assert_eq!(fx.manager.ensure_vad_model(None).await?, path);
    Ok(())
}

#[tokio::test]
async fn the_voice_activity_model_is_not_one_of_the_models() -> Result<(), Box<dyn Error>> {
    let fx = fixture()?;
    std::fs::write(fx.manager.get_vad_path(), vad_fixture_bytes())?;

    assert_eq!(
        fx.manager.vad_model_path().await?,
        Some(fx.manager.get_vad_path())
    );
    assert_eq!(downloaded_ids(&fx.manager).await?, Vec::<String>::new());
    let local = fx.manager.local_models().await?;
    assert_eq!(
        local
            .iter()
            .map(|(model, _)| model.id.as_str())
            .collect::<Vec<_>>(),
        Vec::<&str>::new()
    );
    assert_eq!(fx.manager.find_first_available_model().await?, None);
    let listed: Vec<&str> = fx
        .manager
        .registry()
        .models()
        .iter()
        .map(|model| model.id.as_str())
        .collect();
    assert_eq!(listed, ["fixture"]);
    let grouped: Vec<&str> = fx
        .manager
        .registry()
        .groups()
        .iter()
        .flat_map(|group| group.models.iter().map(|model| model.id.as_str()))
        .collect();
    assert_eq!(grouped, ["fixture"]);
    Ok(())
}

#[tokio::test]
async fn deleting_a_model_leaves_the_voice_activity_model() -> Result<(), Box<dyn Error>> {
    let fx = fixture()?;
    let model = fx.manager.model("fixture")?;
    std::fs::write(fx.manager.get_model_path(model), &fx.content)?;
    std::fs::write(fx.manager.get_vad_path(), vad_fixture_bytes())?;

    fx.manager.delete_model(model).await?;

    assert_eq!(
        fx.manager.vad_model_path().await?,
        Some(fx.manager.get_vad_path())
    );
    Ok(())
}

#[test]
fn a_voice_activity_failure_is_not_a_transcription_failure() {
    let path = PathBuf::from("ggml-silero.bin");
    let failure = WhisperError::from(ConfigurationError::NoModelFound);

    assert_eq!(vad_or_none(Ok(Some(path.clone()))), Some(path));
    assert_eq!(vad_or_none(Ok(None)), None);
    assert_eq!(vad_or_none(Err(failure)), None);
}
