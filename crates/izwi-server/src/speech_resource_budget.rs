//! Process-wide byte reservations for streaming persistence. These limits are
//! separate from device inference admission and apply across all live requests.
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, OnceLock,
};

#[derive(Debug)]
pub(crate) struct ByteBudget {
    limit: usize,
    used: AtomicUsize,
}

impl ByteBudget {
    pub(crate) fn new(limit: usize) -> Arc<Self> {
        Arc::new(Self {
            limit,
            used: AtomicUsize::new(0),
        })
    }

    pub(crate) fn reserve(self: &Arc<Self>, bytes: usize) -> anyhow::Result<ByteReservation> {
        self.used
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |used| {
                used.checked_add(bytes).filter(|next| *next <= self.limit)
            })
            .map_err(|_| anyhow::anyhow!("Aggregate speech persistence byte capacity exhausted"))?;
        Ok(ByteReservation {
            budget: self.clone(),
            bytes,
        })
    }
}

pub(crate) struct ByteReservation {
    budget: Arc<ByteBudget>,
    bytes: usize,
}

impl ByteReservation {
    pub(crate) fn grow(&mut self, bytes: usize) -> anyhow::Result<()> {
        let mut added = self.budget.reserve(bytes)?;
        self.bytes += added.bytes;
        added.bytes = 0;
        Ok(())
    }
}

impl Drop for ByteReservation {
    fn drop(&mut self) {
        self.budget.used.fetch_sub(self.bytes, Ordering::AcqRel);
    }
}

fn configured_budget(name: &str, default: usize) -> Arc<ByteBudget> {
    let limit = std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|value| *value >= 44)
        .unwrap_or(default);
    ByteBudget::new(limit)
}

pub(crate) fn spool_budget() -> &'static Arc<ByteBudget> {
    static BUDGET: OnceLock<Arc<ByteBudget>> = OnceLock::new();
    BUDGET.get_or_init(|| configured_budget("IZWI_TTS_TOTAL_SPOOL_BYTES", 1024 * 1024 * 1024))
}

pub(crate) fn upload_budget() -> &'static Arc<ByteBudget> {
    static BUDGET: OnceLock<Arc<ByteBudget>> = OnceLock::new();
    BUDGET.get_or_init(|| configured_budget("IZWI_TTS_TOTAL_UPLOAD_BYTES", 256 * 1024 * 1024))
}

pub(crate) fn event_budget() -> &'static Arc<ByteBudget> {
    static BUDGET: OnceLock<Arc<ByteBudget>> = OnceLock::new();
    BUDGET
        .get_or_init(|| configured_budget("IZWI_AUDIO_STREAM_TOTAL_EVENT_BYTES", 64 * 1024 * 1024))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aggregate_reservations_reject_without_changing_ownership_and_return_on_drop() {
        let budget = ByteBudget::new(10);
        let mut first = budget.reserve(4).unwrap();
        let second = budget.reserve(5).unwrap();
        assert!(first.grow(2).is_err());
        assert_eq!(budget.used.load(Ordering::Acquire), 9);
        drop(second);
        first.grow(6).unwrap();
        assert!(budget.reserve(1).is_err());
        drop(first);
        assert_eq!(budget.used.load(Ordering::Acquire), 0);
        assert!(budget.reserve(usize::MAX).is_err());
    }

    #[tokio::test]
    async fn abort_releases_live_reservation() {
        let budget = ByteBudget::new(16);
        let task_budget = budget.clone();
        let (tx, rx) = tokio::sync::oneshot::channel();
        let task = tokio::spawn(async move {
            let _reservation = task_budget.reserve(16).unwrap();
            tx.send(()).unwrap();
            std::future::pending::<()>().await;
        });
        rx.await.unwrap();
        assert!(budget.reserve(1).is_err());
        task.abort();
        let _ = task.await;
        assert_eq!(budget.used.load(Ordering::Acquire), 0);
    }
}
