//! 记忆存储的快照与恢复：JSONL 用于持久，内存用于测试。
//!
//! 这里存在两种持久化形态，因为它们回答不同需求：
//!
//!   * [`save_snapshot`] / [`restore_snapshot`] 把整个存储写为一个
//!     JSON-lines 文件——某部署在某一时刻所知内容的记录，
//!     也是重启后重新打开它的方式。
//!   * [`append_journal`] 不读取文件就把记录追加到末尾，
//!     供一个运行期间持续写入的进程使用。
//!
//! 记录在写出与读回时保持其 id，因此恢复不会重新编号
//! 其他组件已引用的记忆。文件是纯文本格式（每行一个
//! [`MemoryRecord`]，无信封）：被截断的最后一行会被报告为
//! 一次中断写入的尾部，而非被静默跳过。

use std::path::{Path, PathBuf};

use serde_json::to_string;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

use crate::agent_memory::error::{MemoryError, MemoryResult};
use crate::agent_memory::record::MemoryRecord;
use crate::agent_memory::store::MemoryStore;

/// 记忆文件所在之处。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryFile {
    path: PathBuf,
}

impl MemoryFile {
    /// 指向一个可能尚不存在的文件。
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    /// 此句柄写入与读取的文件。
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// 写出存储中的每条记录，替换任何先前的快照。
    pub async fn save(&self, store: &dyn MemoryStore) -> MemoryResult<usize> {
        save_snapshot(store, &self.path).await
    }

    /// 把文件中的每条记录插入存储。
    pub async fn restore(&self, store: &dyn MemoryStore) -> MemoryResult<usize> {
        restore_snapshot(store, &self.path).await
    }

    /// 追加记录，不先读取文件。
    pub async fn append(&self, records: &[MemoryRecord]) -> MemoryResult<usize> {
        append_journal(records, &self.path).await
    }

    /// 读取文件，不触碰存储。
    pub async fn read(&self) -> MemoryResult<Vec<MemoryRecord>> {
        read_records(&self.path).await
    }

    /// 文件是否存在。
    pub async fn exists(&self) -> bool {
        tokio::fs::try_exists(&self.path).await.unwrap_or(false)
    }

    /// 删除文件；文件缺失不算错误。
    pub async fn delete(&self) -> MemoryResult<()> {
        match tokio::fs::remove_file(&self.path).await {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(MemoryError::persistence(error)),
        }
    }
}

/// 把存储中的每条记录以 JSON lines 写到 `path`。
pub async fn save_snapshot(store: &dyn MemoryStore, path: &Path) -> MemoryResult<usize> {
    let records = store.snapshot().await;
    let mut body = String::new();
    for record in &records {
        body.push_str(&to_string(record).map_err(MemoryError::persistence)?);
        body.push('\n');
    }
    write_all(path, body.as_bytes()).await?;
    Ok(records.len())
}

/// 已在存储中的记录被跳过；其余被插入。
pub async fn restore_snapshot(store: &dyn MemoryStore, path: &Path) -> MemoryResult<usize> {
    let records = read_records(path).await?;
    let mut restored = 0;
    for record in records {
        if store.get(record.id).await?.is_some() {
            continue;
        }
        store.insert(record).await?;
        restored += 1;
    }
    Ok(restored)
}

/// 把记录追加到 `path`，缺失时创建它。
pub async fn append_journal(records: &[MemoryRecord], path: &Path) -> MemoryResult<usize> {
    if records.is_empty() {
        return Ok(0);
    }
    let mut body = String::new();
    for record in records {
        body.push_str(&to_string(record).map_err(MemoryError::persistence)?);
        body.push('\n');
    }
    append_all(path, body.as_bytes()).await?;
    Ok(records.len())
}

/// 解析一个 JSON-lines 记忆文件。
pub async fn read_records(path: &Path) -> MemoryResult<Vec<MemoryRecord>> {
    let file = tokio::fs::File::open(path)
        .await
        .map_err(MemoryError::persistence)?;
    let mut lines = BufReader::new(file).lines();
    let mut records = Vec::new();
    let mut number = 0usize;
    while let Some(line) = lines.next_line().await.map_err(MemoryError::persistence)? {
        number += 1;
        if line.trim().is_empty() {
            continue;
        }
        match serde_json::from_str::<MemoryRecord>(&line) {
            Ok(record) => records.push(record),
            Err(error) => {
                return Err(MemoryError::persistence(format!(
                    "{} line {number}: {error}",
                    path.display()
                )));
            }
        }
    }
    Ok(records)
}

async fn write_all(path: &Path, body: &[u8]) -> MemoryResult<()> {
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .map_err(MemoryError::persistence)?;
    }
    let mut file = tokio::fs::File::create(path)
        .await
        .map_err(MemoryError::persistence)?;
    file.write_all(body)
        .await
        .map_err(MemoryError::persistence)?;
    file.flush().await.map_err(MemoryError::persistence)
}

async fn append_all(path: &Path, body: &[u8]) -> MemoryResult<()> {
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .map_err(MemoryError::persistence)?;
    }
    let mut file = tokio::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .await
        .map_err(MemoryError::persistence)?;
    file.write_all(body)
        .await
        .map_err(MemoryError::persistence)?;
    file.flush().await.map_err(MemoryError::persistence)
}

/// 进程临时目录下的一个唯一路径，用于测试与一次性转储。
pub fn temp_path(label: &str) -> PathBuf {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|elapsed| elapsed.subsec_nanos())
        .unwrap_or(0);
    std::env::temp_dir().join(format!("xihe-memory-{}-{nonce}.jsonl", sanitize(label)))
}

fn sanitize(label: &str) -> String {
    label
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent_memory::embed::HashingEmbedder;
    use crate::agent_memory::record::{MemoryKind, MemoryRecord};
    use crate::agent_memory::store::{InMemoryStore, MemoryStore};

    fn store() -> InMemoryStore {
        InMemoryStore::with_embedder(HashingEmbedder::default())
    }

    async fn seed(store: &InMemoryStore, texts: &[&str]) {
        for text in texts {
            store.remember(MemoryKind::Fact, text).await.unwrap();
        }
    }

    async fn cleanup(path: &Path) {
        let _ = tokio::fs::remove_file(path).await;
    }

    #[tokio::test]
    async fn a_snapshot_round_trips_through_disk() {
        let path = temp_path("round-trip");
        let source = store();
        seed(
            &source,
            &["the report needs per-region totals", "use ISO dates"],
        )
        .await;
        assert_eq!(save_snapshot(&source, &path).await.unwrap(), 2);

        let target = store();
        assert_eq!(restore_snapshot(&target, &path).await.unwrap(), 2);
        let records = target.snapshot().await;
        assert_eq!(records.len(), 2);
        assert!(
            records
                .iter()
                .any(|record| record.text.contains("per-region")),
            "{records:?}"
        );
        assert_eq!(
            records[0].embedding.dimensions(),
            source.dimensions(),
            "vectors survive the write"
        );
        cleanup(&path).await;
    }

    #[tokio::test]
    async fn restoring_twice_does_not_duplicate() {
        let path = temp_path("idempotent");
        let source = store();
        seed(&source, &["one fact"]).await;
        save_snapshot(&source, &path).await.unwrap();
        let target = store();
        assert_eq!(restore_snapshot(&target, &path).await.unwrap(), 1);
        assert_eq!(
            restore_snapshot(&target, &path).await.unwrap(),
            0,
            "records already present are skipped"
        );
        assert_eq!(target.snapshot().await.len(), 1);
        cleanup(&path).await;
    }

    #[tokio::test]
    async fn ids_are_preserved_across_a_snapshot() {
        let path = temp_path("ids");
        let source = store();
        seed(&source, &["first", "second"]).await;
        let mut before: Vec<u64> = source
            .snapshot()
            .await
            .into_iter()
            .map(|record| record.id)
            .collect();
        save_snapshot(&source, &path).await.unwrap();
        let target = store();
        restore_snapshot(&target, &path).await.unwrap();
        let mut after: Vec<u64> = target
            .snapshot()
            .await
            .into_iter()
            .map(|record| record.id)
            .collect();
        before.sort_unstable();
        after.sort_unstable();
        assert_eq!(after, before, "ids must not be renumbered");
        cleanup(&path).await;
    }

    #[tokio::test]
    async fn a_journal_appends_without_rewriting() {
        let path = temp_path("journal");
        let source = store();
        seed(&source, &["early"]).await;
        let first = source.snapshot().await;
        assert_eq!(append_journal(&first, &path).await.unwrap(), 1);
        seed(&source, &["later"]).await;
        let matches = |record: &MemoryRecord| record.text == "later";
        let second = source.find(&matches).await;
        assert_eq!(append_journal(&second, &path).await.unwrap(), 1);
        assert_eq!(read_records(&path).await.unwrap().len(), 2);
        cleanup(&path).await;
    }

    #[tokio::test]
    async fn appending_nothing_leaves_the_file_alone() {
        let path = temp_path("empty-append");
        assert_eq!(append_journal(&[], &path).await.unwrap(), 0);
        assert!(!path.exists());
    }

    #[tokio::test]
    async fn a_malformed_line_is_reported_with_its_position() {
        let path = temp_path("malformed");
        tokio::fs::write(&path, "not json\n").await.unwrap();
        let error = read_records(&path).await.unwrap_err();
        assert!(error.to_string().contains("line 1"), "{error}");
        assert!(matches!(error, MemoryError::Persistence { .. }), "{error}");
        cleanup(&path).await;
    }

    #[tokio::test]
    async fn a_missing_file_is_a_persistence_error_not_a_panic() {
        let error = read_records(Path::new("/nonexistent/xihe-memory-file.jsonl"))
            .await
            .unwrap_err();
        assert!(matches!(error, MemoryError::Persistence { .. }));
    }

    #[tokio::test]
    async fn blank_lines_are_skipped() {
        let path = temp_path("blanks");
        let source = store();
        seed(&source, &["only one"]).await;
        let mut body = "\n\n".to_string();
        for record in source.snapshot().await {
            body.push_str(&to_string(&record).unwrap());
            body.push_str("\n\n");
        }
        tokio::fs::write(&path, body).await.unwrap();
        assert_eq!(read_records(&path).await.unwrap().len(), 1);
        cleanup(&path).await;
    }

    #[tokio::test]
    async fn the_file_handle_tracks_its_own_path() {
        let path = temp_path("handle");
        let file = MemoryFile::new(&path);
        assert_eq!(file.path(), path.as_path());
        assert!(!file.exists().await);
        let source = store();
        seed(&source, &["something"]).await;
        assert_eq!(file.save(&source).await.unwrap(), 1);
        assert!(file.exists().await);
        assert_eq!(file.read().await.unwrap().len(), 1);
        let fresh = store();
        assert_eq!(file.restore(&fresh).await.unwrap(), 1);
        let shared: &dyn MemoryStore = &fresh;
        assert_eq!(save_snapshot(shared, &path).await.unwrap(), 1);
        file.delete().await.unwrap();
        file.delete().await.unwrap();
        assert!(!file.exists().await);
    }

    #[test]
    fn temp_paths_are_unique_and_safe() {
        let a = temp_path("a/b c");
        let b = temp_path("a/b c");
        assert_ne!(a, b, "each call gets its own file");
        assert!(
            a.file_name()
                .unwrap()
                .to_string_lossy()
                .starts_with("xihe-memory-a_b_c-"),
            "{a:?}"
        );
    }
}
