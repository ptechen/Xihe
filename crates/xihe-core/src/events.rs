//! 帧作用域的事件缓冲区，是异步消息总线的同步版姊妹。

use crate::component::Message;

/// 某个组件类型的双缓冲事件。
///
/// 写入者追加到前缓冲区；读取者在整帧内看到稳定的后缓冲区，
/// 因此除非驱动器在阶段之间调用 [`Events::update`]，系统无法对
/// 同一帧内产生的事件作出反应。
///
/// 异步系统通常更偏好 `xihe_schedule` 中的总线，它能跨帧边界存活；
/// `Events` 则是用于 `&mut World` 阶段（如启动与拆卸）的廉价选择。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Events<E: Message> {
    fresh: Vec<E>,
    stable: Vec<E>,
}

impl<E: Message> Default for Events<E> {
    fn default() -> Self {
        Self::new()
    }
}

impl<E: Message> Events<E> {
    /// 创建两个空缓冲区。
    pub fn new() -> Self {
        Self {
            fresh: Vec::new(),
            stable: Vec::new(),
        }
    }

    /// 追加一个事件，在下一次 [`Events::update`] 后可见。
    pub fn send(&mut self, event: E) {
        self.fresh.push(event);
    }

    /// 追加多个事件。
    pub fn send_all(&mut self, events: impl IntoIterator<Item = E>) {
        self.fresh.extend(events);
    }

    /// 按发送顺序读取稳定缓冲区。
    pub fn iter(&self) -> std::iter::Chain<std::slice::Iter<'_, E>, std::slice::Iter<'_, E>> {
        self.stable.iter().chain(self.fresh.iter())
    }

    /// 两个缓冲区中排队的事件总数。
    #[inline]
    pub fn len(&self) -> usize {
        self.fresh.len() + self.stable.len()
    }

    /// 没有排队事件时为 `true`。
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// 轮转缓冲区：新事件变为稳定，旧的被丢弃。
    pub fn update(&mut self) {
        std::mem::swap(&mut self.stable, &mut self.fresh);
        self.fresh.clear();
    }

    /// 移除并返回所有排队的事件。
    pub fn drain(&mut self) -> Vec<E> {
        let mut out = std::mem::take(&mut self.stable);
        out.append(&mut self.fresh);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Debug, PartialEq)]
    struct Tick(u32);

    #[test]
    fn buffers_rotate_on_update() {
        let mut events = Events::new();
        events.send(Tick(1));
        assert_eq!(events.iter().cloned().collect::<Vec<_>>(), vec![Tick(1)], "readable in-frame");
        events.update();
        events.send(Tick(2));
        assert_eq!(
            events.iter().cloned().collect::<Vec<_>>(),
            vec![Tick(1), Tick(2)]
        );
        events.update();
        assert_eq!(events.iter().cloned().collect::<Vec<_>>(), vec![Tick(2)]);
        assert_eq!(events.drain(), vec![Tick(2)]);
        assert!(events.is_empty());
    }
}
