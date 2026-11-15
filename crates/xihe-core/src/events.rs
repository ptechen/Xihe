//! Frame-scoped event buffers, the synchronous sibling of the async message bus.

use crate::component::Message;

/// Double-buffered events for a component type.
///
/// Writers append to the front buffer; readers see the stable back buffer for the
/// whole frame so a system cannot react to an event raised by the same frame
/// unless [`Events::update`] is called by the driver between stages.
///
/// Async systems usually prefer the bus in `xihe_schedule`, which survives frame
/// boundaries; `Events` is the cheap option for `&mut World` phases such as
/// startup and teardown.
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
    /// Creates two empty buffers.
    pub fn new() -> Self {
        Self {
            fresh: Vec::new(),
            stable: Vec::new(),
        }
    }

    /// Appends an event, visible after the next [`Events::update`].
    pub fn send(&mut self, event: E) {
        self.fresh.push(event);
    }

    /// Appends many events.
    pub fn send_all(&mut self, events: impl IntoIterator<Item = E>) {
        self.fresh.extend(events);
    }

    /// Reads the stable buffer in send order.
    pub fn iter(&self) -> std::iter::Chain<std::slice::Iter<'_, E>, std::slice::Iter<'_, E>> {
        self.stable.iter().chain(self.fresh.iter())
    }

    /// Total queued events, both buffers.
    #[inline]
    pub fn len(&self) -> usize {
        self.fresh.len() + self.stable.len()
    }

    /// `true` when no event is queued.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Rotates buffers: the fresh events become stable, older ones are dropped.
    pub fn update(&mut self) {
        std::mem::swap(&mut self.stable, &mut self.fresh);
        self.fresh.clear();
    }

    /// Removes and returns every queued event.
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
