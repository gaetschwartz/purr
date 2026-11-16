use purr_core::config::RunConfig;

pub fn run_config() -> RunConfig {
    RunConfig::new().with_threads(1)
}
