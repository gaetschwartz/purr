use crate::error::{Result, WhisperError};
use rustls::crypto::{CryptoProvider, ring};

fn ensure_crypto_provider() -> Result<()> {
    if CryptoProvider::get_default().is_some() {
        return Ok(());
    }
    match ring::default_provider().install_default() {
        Ok(()) => Ok(()),
        Err(_) if CryptoProvider::get_default().is_some() => Ok(()),
        Err(_) => Err(WhisperError::NetworkError {
            message: "Failed to install a TLS crypto provider".to_owned(),
        }),
    }
}

pub fn client_builder() -> Result<reqwest::ClientBuilder> {
    ensure_crypto_provider()?;
    Ok(reqwest::Client::builder())
}
