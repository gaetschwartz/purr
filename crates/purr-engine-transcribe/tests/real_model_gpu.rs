//! The text a GPU produces compared with the CPU goldens of the same whole-source runs.
//!
//! Only a build with a GPU backend has these tests. A run that did not end up on a GPU fails them:
//! the point is the GPU's text, not whichever device the plan fell back to.

#![cfg(any(feature = "cuda", feature = "metal", feature = "vulkan"))]

mod common;

use common::{Sample, TestResult, TranscribeModel};
use purr_common::devices::DeviceOrder;
use rstest::rstest;

#[rstest]
#[case::parakeet(TranscribeModel::Parakeet)]
#[case::qwen3(TranscribeModel::Qwen3)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_gpu_gives_the_text_of_the_cpu_golden(#[case] model: TranscribeModel) -> TestResult {
    let path = common::model_path(model).await?;
    let samples = common::load(Sample::Jfk).await?;
    let config = common::run_config().with_devices(DeviceOrder::automatic());
    let result = common::whole_result(&path, config, &samples).await?;
    assert!(
        result.accel.selected.is_gpu(),
        "the run was not on a GPU: {:?}",
        result.accel
    );
    pretty_assertions::assert_eq!(common::golden_whole_text(model, Sample::Jfk)?, result.text);
    Ok(())
}
