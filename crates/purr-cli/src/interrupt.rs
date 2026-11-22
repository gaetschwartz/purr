use crate::{output, statusln};
use std::process;
use std::time::Duration;
use tokio::sync::watch;
use tracing::{debug, warn};

const EXIT_AT_ONCE: i32 = 130;

#[derive(Clone)]
pub struct Interrupt {
    requested: watch::Receiver<bool>,
}

impl Interrupt {
    /// The first Ctrl-C asks the run to stop; the second ends the process at once.
    pub fn install() -> Self {
        let (tx, requested) = watch::channel(false);
        tokio::spawn(async move {
            if let Err(error) = tokio::signal::ctrl_c().await {
                warn!("Ctrl-C cannot be caught: {error}");
                return;
            }
            statusln!("Stopping; press Ctrl-C again to quit at once");
            tx.send_replace(true);
            if let Err(error) = tokio::signal::ctrl_c().await {
                warn!("Ctrl-C cannot be caught: {error}");
                return;
            }
            output::clear_bars();
            process::exit(EXIT_AT_ONCE);
        });
        Self { requested }
    }

    /// Asks for a stop once `delay` has passed, as the first Ctrl-C would.
    pub fn after(delay: Duration) -> Self {
        let (tx, requested) = watch::channel(false);
        tokio::spawn(async move {
            tokio::time::sleep(delay).await;
            tx.send_replace(true);
        });
        Self { requested }
    }

    pub async fn requested(&self) {
        let mut requested = self.requested.clone();
        if requested.wait_for(|asked| *asked).await.is_err() {
            debug!("The interrupt source is gone; no stop will be asked for");
            std::future::pending::<()>().await;
        }
    }

    pub fn is_requested(&self) -> bool {
        *self.requested.borrow()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn a_stop_is_asked_for_once_the_delay_has_passed() {
        let interrupt = Interrupt::after(Duration::from_millis(20));
        assert!(!interrupt.is_requested());
        interrupt.requested().await;
        assert!(interrupt.is_requested());
    }

    #[tokio::test]
    async fn every_clone_sees_the_stop() {
        let interrupt = Interrupt::after(Duration::from_millis(20));
        let other = interrupt.clone();
        other.requested().await;
        assert!(interrupt.is_requested());
    }
}
