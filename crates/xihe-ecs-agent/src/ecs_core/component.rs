//! 组件、资源与消息的标记 trait。

/// 附着在 [`Entity`](crate::ecs_core::Entity) 上的数据。
///
/// 该 trait 自动实现：任何 `Send + Sync + 'static` 类型都是组件，
/// 因此普通结构体和 `#[derive]` 的数据模型无需样板代码，也无需导入宏。
/// 保留 `Sync` 约束，正是为了让只读系统能在并发运行的 tokio 任务间
/// 共享组件存储。
///
/// ```
/// use xihe::ecs_core::World;
///
/// #[derive(Debug)]
/// struct Counter { value: u32 }
///
/// let mut world = World::new();
/// let e = world.spawn();
/// world.insert(e, Counter { value: 1 });
/// assert_eq!(world.get::<Counter>(e).unwrap().value, 1);
/// ```
pub trait Component: Send + Sync + 'static {}

impl<T: Send + Sync + 'static> Component for T {}

/// 存放在 [`World`](crate::ecs_core::World) 中而非实体上的共享单例状态：
/// 提供方客户端、工具注册表、token 账本、配置。
///
/// 与 [`Component`] 一样，这是一个全覆盖标记。资源必须能廉价构造，
/// 或在首次使用前插入。
pub trait Resource: Send + Sync + 'static {}

impl<T: Send + Sync + 'static> Resource for T {}

/// [`Events`](crate::ecs_core::Events) 缓冲区与异步消息总线的载荷。
///
/// 消息是 `Clone` 的，因为一次广播可能在每帧投递给多个智能体实体；
/// 请保持载荷小巧，重数据改用 [`Entity`](crate::ecs_core::Entity) 句柄传递。
pub trait Message: Send + Sync + Clone + 'static {}

impl<T: Send + Sync + Clone + 'static> Message for T {}
