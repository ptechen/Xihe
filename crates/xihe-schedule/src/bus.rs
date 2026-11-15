//! Typed publish/subscribe between tasks, agents and the outside world.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use tokio::sync::{Mutex, broadcast};
use xihe_core::Message;

/// Default backlog kept per subscriber.
pub const DEFAULT_CAPACITY: usize = 256;

/// Counters describing bus traffic.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize)]
pub struct BusStats {
    /// Messages published since creation.
    pub published: u64,
    /// Receives that skipped messages because a subscriber fell behind.
    pub lagged: u64,
    /// Subscribers currently attached.
    pub subscribers: usize,
}

/// A typed broadcast bus, stored in the world as a resource.
///
/// Every agent entity gets its own [`Inbox`] from the same bus, so a message
/// published once reaches each subscriber exactly once and each subscriber keeps
/// its own read position. External clients use [`MessageBus::handle`] to inject
/// user requests without ever touching the world lock.
#[derive(Debug)]
pub struct MessageBus<M: Message> {
    sender: broadcast::Sender<M>,
    capacity: usize,
    published: Arc<AtomicU64>,
    lagged: Arc<AtomicU64>,
}

impl<M: Message> MessageBus<M> {
    /// Creates a bus; `capacity` is the per-subscriber backlog.
    pub fn with_capacity(capacity: usize) -> Self {
        let (sender, _) = broadcast::channel(capacity.max(1));
        Self {
            sender,
            capacity: capacity.max(1),
            published: Arc::new(AtomicU64::new(0)),
            lagged: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Creates a bus with [`DEFAULT_CAPACITY`] backlog.
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_CAPACITY)
    }

    /// Publishes a message, returning how many subscribers received it.
    ///
    /// Publishing never blocks: a subscriber that cannot keep up loses messages
    /// and reports the gap through [`Inbox::lagged`].
    pub fn publish(&self, message: M) -> usize {
        self.published.fetch_add(1, Ordering::Relaxed);
        self.sender.send(message).unwrap_or(0)
    }

    /// Attaches a new inbox starting at the current position.
    pub fn subscribe(&self) -> Inbox<M> {
        Inbox {
            receiver: Arc::new(Mutex::new(self.sender.subscribe())),
            lagged: self.lagged.clone(),
        }
    }

    /// Cloned handle for producers outside the world (HTTP servers, CLIs, tests).
    pub fn handle(&self) -> BusHandle<M> {
        BusHandle {
            sender: self.sender.clone(),
            published: self.published.clone(),
            capacity: self.capacity,
        }
    }

    /// Current subscriber count.
    pub fn subscribers(&self) -> usize {
        self.sender.receiver_count()
    }

    /// Per-subscriber backlog size.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Traffic counters.
    pub fn stats(&self) -> BusStats {
        BusStats {
            published: self.published.load(Ordering::Relaxed),
            lagged: self.lagged.load(Ordering::Relaxed),
            subscribers: self.subscribers(),
        }
    }
}

impl<M: Message> Default for MessageBus<M> {
    fn default() -> Self {
        Self::new()
    }
}

/// Cloneable producer handle that does not need world access.
#[derive(Clone)]
pub struct BusHandle<M: Message> {
    sender: broadcast::Sender<M>,
    published: Arc<AtomicU64>,
    capacity: usize,
}

impl<M: Message> BusHandle<M> {
    /// Publishes a message, returning the number of subscribers notified.
    pub fn publish(&self, message: M) -> usize {
        self.published.fetch_add(1, Ordering::Relaxed);
        self.sender.send(message).unwrap_or(0)
    }

    /// Attaches another inbox.
    pub fn subscribe(&self) -> Inbox<M> {
        Inbox {
            receiver: Arc::new(Mutex::new(self.sender.subscribe())),
            lagged: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Current subscriber count.
    pub fn subscribers(&self) -> usize {
        self.sender.receiver_count()
    }

    /// Backlog size of new subscribers.
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

impl<M: Message> std::fmt::Debug for BusHandle<M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BusHandle")
            .field("subscribers", &self.subscribers())
            .finish()
    }
}

/// Per-subscriber read end, stored as an entity component.
///
/// The inner receiver sits behind a `Mutex` so a system holding only a *read*
/// guard of the world can still pull messages out of its inbox.
#[derive(Clone)]
pub struct Inbox<M: Message> {
    receiver: Arc<Mutex<broadcast::Receiver<M>>>,
    lagged: Arc<AtomicU64>,
}

impl<M: Message> Inbox<M> {
    /// An inbox that never yields messages — handy as a test double.
    pub fn closed() -> Self {
        let (sender, receiver) = broadcast::channel(1);
        drop(sender);
        Self {
            receiver: Arc::new(Mutex::new(receiver)),
            lagged: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Waits for the next message; `None` only when the bus is gone.
    pub async fn recv(&self) -> Option<M> {
        let mut receiver = self.receiver.lock().await;
        loop {
            match receiver.recv().await {
                Ok(message) => return Some(message),
                Err(broadcast::error::RecvError::Lagged(skipped)) => {
                    self.lagged.fetch_add(skipped, Ordering::Relaxed);
                    continue;
                }
                Err(broadcast::error::RecvError::Closed) => return None,
            }
        }
    }

    /// Takes one buffered message without waiting.
    pub fn try_recv(&self) -> Option<M> {
        let mut receiver = self.receiver.try_lock().ok()?;
        loop {
            match receiver.try_recv() {
                Ok(message) => return Some(message),
                Err(broadcast::error::TryRecvError::Lagged(skipped)) => {
                    self.lagged.fetch_add(skipped, Ordering::Relaxed);
                    continue;
                }
                Err(broadcast::error::TryRecvError::Empty)
                | Err(broadcast::error::TryRecvError::Closed) => return None,
            }
        }
    }

    /// Takes up to `max` buffered messages, oldest first.
    pub fn drain(&self, max: usize) -> Vec<M> {
        let mut out = Vec::new();
        while out.len() < max {
            match self.try_recv() {
                Some(message) => out.push(message),
                None => break,
            }
        }
        out
    }

    /// `true` when nothing is buffered.
    pub fn is_empty(&self) -> bool {
        self.receiver
            .try_lock()
            .map(|receiver| receiver.len() == 0)
            .unwrap_or(true)
    }

    /// Messages this subscriber lost because it fell behind.
    pub fn lagged(&self) -> u64 {
        self.lagged.load(Ordering::Relaxed)
    }
}

impl<M: Message> std::fmt::Debug for Inbox<M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Inbox")
            .field("lagged", &self.lagged())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Debug, PartialEq)]
    struct Ping(&'static str);

    #[test]
    fn every_subscriber_gets_its_own_copy() {
        let bus = MessageBus::new();
        let first = bus.subscribe();
        let second = bus.subscribe();
        assert_eq!(bus.publish(Ping("hi")), 2);
        assert_eq!(first.drain(4), vec![Ping("hi")]);
        assert_eq!(second.drain(4), vec![Ping("hi")]);
        assert!(first.is_empty());
        assert_eq!(bus.stats().published, 1);
    }

    #[test]
    fn handle_publishes_without_world_access() {
        let bus = MessageBus::with_capacity(2);
        let handle = bus.handle();
        let inbox = handle.subscribe();
        assert_eq!(handle.publish(Ping("a")), 1);
        assert_eq!(inbox.try_recv(), Some(Ping("a")));
        assert_eq!(handle.subscribers(), 1);
        assert_eq!(bus.capacity(), 2);
    }

    #[test]
    fn slow_subscribers_report_lag_instead_of_blocking() {
        let bus = MessageBus::with_capacity(2);
        let inbox = bus.subscribe();
        for i in 0..5 {
            bus.publish(Ping(if i == 0 { "a" } else { "b" }));
        }
        let got = inbox.drain(10);
        assert_eq!(got.len(), 2, "backlog capped at capacity");
        assert!(inbox.lagged() > 0);
    }

    #[tokio::test]
    async fn recv_wakes_on_publish() {
        let bus = MessageBus::new();
        let inbox = bus.subscribe();
        let handle = bus.handle();
        let waiter = async move { inbox.recv().await };
        tokio::pin!(waiter);
        assert!(tokio::time::timeout(std::time::Duration::from_millis(20), &mut waiter)
            .await
            .is_err());
        handle.publish(Ping("late"));
        assert_eq!(waiter.await, Some(Ping("late")));
    }
}
