use purr_core::{transcribe_file_sync, TranscriptionConfig};
use std::path::PathBuf;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    println!("Testing Whisper transcription...");

    let audio_path = "/Users/gaetan/dev/purr/samples/jfk.wav";
    let model_path = dirs::home_dir()
        .unwrap()
        .join(".local/share/purr/models/ggml-small.en.bin");

    println!("Audio file: {}", audio_path);
    println!("Model file: {}", model_path.display());

    if !model_path.exists() {
        eprintln!("Model file not found at: {}", model_path.display());
        return Ok(());
    }

    let config = TranscriptionConfig::new()
        .with_model_path(model_path)
        .with_language("en")
        .with_verbose(true);

    match transcribe_file_sync(audio_path, Some(config)).await {
        Ok(result) => {
            println!("\n=== TRANSCRIPTION RESULT ===");
            println!("Text: {}", result.text);
            println!("Processing time: {:.2}s", result.processing_time);
            println!("Audio duration: {:.2}s", result.audio_duration);
            println!("Segments: {}", result.segments.len());
        }
        Err(e) => {
            eprintln!("Transcription failed: {}", e);
        }
    }

    Ok(())
}
