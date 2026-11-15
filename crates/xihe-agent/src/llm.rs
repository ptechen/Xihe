//! The provider client as a world resource.
//!
//! One [`LlmService`] per process is usually right: it owns the connection pool
//! and the retry policy, and every agent reads it from the world instead of
//! carrying its own client. Because the ReAct system must call the provider
//! *without* holding the world lock, the service hands out cheap `Arc` clones
//! rather than borrowed clients.

use std::fmt::{self, Debug, Formatter};
use std::sync::Arc;

use xihe_core::World;
use xihe_llm::{ChatClient, DynProvider, Provider};
use xihe_schedule::SystemContext;

/// A shared chat client, stored as a resource so systems can find it by type.
#[derive(Clone)]
pub struct LlmService(Arc<ChatClient>);

impl LlmService {
    /// Wrap an existing client, keeping the retry policy it was built with.
    pub fn new(client: ChatClient) -> Self {
        Self(Arc::new(client))
    }

    /// Wrap a provider, adopting its default model.
    pub fn from_provider<P: Provider>(provider: P) -> Self {
        Self::new(ChatClient::from_provider(provider))
    }

    /// Wrap a provider behind an `Arc`, naming the model to use by default.
    pub fn from_dyn(provider: DynProvider, model: impl Into<String>) -> Self {
        Self::new(ChatClient::new(provider, model))
    }

    /// The client behind the shared handle.
    pub fn client(&self) -> &ChatClient {
        &self.0
    }

    /// A second handle to the same client.
    pub fn shared(&self) -> Arc<ChatClient> {
        Arc::clone(&self.0)
    }

    /// The model this service answers with when a request does not name one.
    pub fn model(&self) -> String {
        self.0.model().to_string()
    }

    /// The provider's own identifier, e.g. `mock`.
    pub fn provider_id(&self) -> String {
        self.0.provider_id().to_string()
    }

    /// Put the service in the world, returning any previous one.
    pub fn install(&self, world: &mut World) -> Option<Self> {
        world.insert_resource(self.clone())
    }

    /// Read the service out of a locked world.
    pub fn from_world(world: &World) -> Option<Self> {
        world.resource::<Self>().cloned()
    }

    /// Read the service through a system context; the lock is gone when this
    /// returns, which is what makes it safe to await a generation afterwards.
    pub async fn read(ctx: &SystemContext) -> Option<Self> {
        ctx.read_resource(|world| world.and_then(Self::from_world))
            .await
    }
}

impl Debug for LlmService {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("LlmService")
            .field("provider", &self.0.provider_id())
            .field("model", &self.0.model())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xihe_llm::MockProvider;

    #[test]
    fn a_service_clones_cheaply_and_shares_one_client() {
        let service = LlmService::from_provider(MockProvider::text("hello"));
        let other = service.clone();
        assert!(
            Arc::ptr_eq(&service.0, &other.0),
            "cloning a service must not clone the client"
        );
        assert_eq!(service.provider_id(), "mock");
    }

    #[test]
    fn a_service_is_found_by_type() {
        let mut world = World::new();
        let service = LlmService::from_provider(MockProvider::text("hello"));
        assert!(service.install(&mut world).is_none());
        assert_eq!(
            LlmService::from_world(&world).expect("installed").model(),
            service.model()
        );
        assert!(LlmService::from_world(&World::new()).is_none());
    }

    #[test]
    fn debug_names_the_provider_without_leaking_the_client() {
        let service = LlmService::from_provider(MockProvider::text("hi").with_model("m-1"));
        let text = format!("{service:?}");
        assert!(text.contains("provider: \"mock\""), "{text}");
        assert!(text.contains("model: \"m-1\""), "{text}");
    }
}
