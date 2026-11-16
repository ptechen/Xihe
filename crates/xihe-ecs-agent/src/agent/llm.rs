//! 作为世界资源的提供方客户端。
//!
//! 每进程一个 [`LlmService`] 通常是对的：它拥有连接池
//! 与重试策略，每个智能体从世界读取它，而非自带客户端。
//! 因为 ReAct 系统必须在不持有世界锁的情况下调用提供方，
//! 服务发放廉价的 `Arc` 克隆，而非借用的客户端。

use std::fmt::{self, Debug, Formatter};
use std::sync::Arc;

use crate::ecs_core::World;
use crate::agent_llm::{ChatClient, DynProvider, Provider};
use crate::ecs_schedule::SystemContext;

/// 共享的聊天客户端，存为资源，使系统能按类型找到它。
#[derive(Clone)]
pub struct LlmService(Arc<ChatClient>);

impl LlmService {
    /// 包装一个已有客户端，保留它构建时所带的重试策略。
    pub fn new(client: ChatClient) -> Self {
        Self(Arc::new(client))
    }

    /// 包装一个提供方，采用其默认模型。
    pub fn from_provider<P: Provider>(provider: P) -> Self {
        Self::new(ChatClient::from_provider(provider))
    }

    /// 把提供方包装在 `Arc` 之后，命名默认使用的模型。
    pub fn from_dyn(provider: DynProvider, model: impl Into<String>) -> Self {
        Self::new(ChatClient::new(provider, model))
    }

    /// 共享句柄背后的客户端。
    pub fn client(&self) -> &ChatClient {
        &self.0
    }

    /// 指向同一客户端的第二个句柄。
    pub fn shared(&self) -> Arc<ChatClient> {
        Arc::clone(&self.0)
    }

    /// 当请求未指定模型时，本服务用以作答的模型。
    pub fn model(&self) -> String {
        self.0.model().to_string()
    }

    /// 提供方自己的标识符，例如 `mock`。
    pub fn provider_id(&self) -> String {
        self.0.provider_id().to_string()
    }

    /// 把服务放入世界，返回先前的那个（若有）。
    pub fn install(&self, world: &mut World) -> Option<Self> {
        world.insert_resource(self.clone())
    }

    /// 从已加锁的世界中读出服务。
    pub fn from_world(world: &World) -> Option<Self> {
        world.resource::<Self>().cloned()
    }

    /// 通过系统上下文读取服务；返回时锁已释放，
    /// 这正是之后能安全 await 一次生成的原因。
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
    use crate::agent_llm::MockProvider;

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
