//! 任务、智能体与外部世界之间的类型化发布/订阅。

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use tokio::sync::{Mutex, broadcast};
use xihe_core::Message;

/// 每个订阅者保留的默认积压量。
pub const DEFAULT_CAPACITY: usize = 256;

/// 描述总线流量的计数器。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize)]
pub struct BusStats {
    /// 自创建以来发布的消息数。
    pub published: u64,
    /// 因订阅者落后而跳过消息的接收次数。
    pub lagged: u64,
    /// 当前接入的订阅者数。
    pub subscribers: usize,
}

/// 一条类型化广播总线，作为资源存放在世界中。
///
/// 每个智能体实体都从同一条总线获得自己的 [`Inbox`]，因此一次发布的
/// 消息恰好送达每个订阅者一次，且每个订阅者保持自己的读取位置。
/// 外部客户端用 [`MessageBus::handle`] 注入用户请求，全程不触碰世界锁。
#[derive(Debug)]
pub struct MessageBus<M: Message> {
    sender: broadcast::Sender<M>,
    capacity: usize,
    published: Arc<AtomicU64>,
    lagged: Arc<AtomicU64>,
}

impl<M: Message> MessageBus<M> {
    /// 创建一条总线；`capacity` 是每个订阅者的积压量。
    pub fn with_capacity(capacity: usize) -> Self {
        let (sender, _) = broadcast::channel(capacity.max(1));
        Self {
            sender,
            capacity: capacity.max(1),
            published: Arc::new(AtomicU64::new(0)),
            lagged: Arc::new(AtomicU64::new(0)),
        }
    }

    /// 创建一条带 [`DEFAULT_CAPACITY`] 积压量的总线。
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_CAPACITY)
    }

    /// 发布一条消息，返回有多少订阅者收到了它。
    ///
    /// 发布永不阻塞：跟不上的订阅者会丢失消息，
    /// 并通过 [`Inbox::lagged`] 报告缺口。
    pub fn publish(&self, message: M) -> usize {
        self.published.fetch_add(1, Ordering::Relaxed);
        self.sender.send(message).unwrap_or(0)
    }

    /// 接入一个从当前位置开始的新收件箱。
    pub fn subscribe(&self) -> Inbox<M> {
        Inbox {
            receiver: Arc::new(Mutex::new(self.sender.subscribe())),
            lagged: self.lagged.clone(),
        }
    }

    /// 供世界之外的生产者（HTTP 服务器、CLI、测试）使用的克隆句柄。
    pub fn handle(&self) -> BusHandle<M> {
        BusHandle {
            sender: self.sender.clone(),
            published: self.published.clone(),
            capacity: self.capacity,
        }
    }

    /// 当前订阅者数量。
    pub fn subscribers(&self) -> usize {
        self.sender.receiver_count()
    }

    /// 每个订阅者的积压大小。
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// 流量计数器。
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

/// 可克隆的生产者句柄，无需访问世界。
#[derive(Clone)]
pub struct BusHandle<M: Message> {
    sender: broadcast::Sender<M>,
    published: Arc<AtomicU64>,
    capacity: usize,
}

impl<M: Message> BusHandle<M> {
    /// 发布一条消息，返回被通知的订阅者数量。
    pub fn publish(&self, message: M) -> usize {
        self.published.fetch_add(1, Ordering::Relaxed);
        self.sender.send(message).unwrap_or(0)
    }

    /// 接入另一个收件箱。
    pub fn subscribe(&self) -> Inbox<M> {
        Inbox {
            receiver: Arc::new(Mutex::new(self.sender.subscribe())),
            lagged: Arc::new(AtomicU64::new(0)),
        }
    }

    /// 当前订阅者数量。
    pub fn subscribers(&self) -> usize {
        self.sender.receiver_count()
    }

    /// 新订阅者的积压大小。
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

/// 每个订阅者的读取端，作为实体组件存放。
///
/// 内部接收器置于一把 `Mutex` 之后，因此只持有世界*读*守卫的系统
/// 仍能从它的收件箱中拉取消息。
#[derive(Clone)]
pub struct Inbox<M: Message> {
    receiver: Arc<Mutex<broadcast::Receiver<M>>>,
    lagged: Arc<AtomicU64>,
}

impl<M: Message> Inbox<M> {
    /// 一个永不产出消息的收件箱——便于用作测试替身。
    pub fn closed() -> Self {
        let (sender, receiver) = broadcast::channel(1);
        drop(sender);
        Self {
            receiver: Arc::new(Mutex::new(receiver)),
            lagged: Arc::new(AtomicU64::new(0)),
        }
    }

    /// 等待下一条消息；仅当总线已消失时为 `None`。
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

    /// 不等待地取出一条缓冲消息。
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

    /// 取出至多 `max` 条缓冲消息，最旧的优先。
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

    /// 没有缓冲时为 `true`。
    pub fn is_empty(&self) -> bool {
        self.receiver
            .try_lock()
            .map(|receiver| receiver.len() == 0)
            .unwrap_or(true)
    }

    /// 此订阅者因落后而丢失的消息数。
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
