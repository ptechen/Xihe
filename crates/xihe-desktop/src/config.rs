//! 模型配置：OpenAI 兼容端点，缺省回退离线 mock。

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// 连接一个 OpenAI 兼容端点所需的全部字段。
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelConfig {
    /// 端点基址，例如 `https://api.openai.com/v1`。
    pub base_url: String,
    /// 密钥；为空则使用离线 mock 提供方。
    pub api_key: String,
    /// 模型名。
    pub model: String,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            base_url: "https://api.openai.com/v1".to_string(),
            api_key: String::new(),
            model: "gpt-4o-mini".to_string(),
        }
    }
}

impl ModelConfig {
    /// 是否配置了真实端点。
    pub fn is_live(&self) -> bool {
        !self.api_key.trim().is_empty()
    }

    /// 配置文件路径：`$HOME/.xihe-desktop.json`。
    pub fn path() -> Option<PathBuf> {
        std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".xihe-desktop.json"))
    }

    /// 读配置；文件缺失或损坏时回退环境变量，再回退默认值。
    pub fn load() -> Self {
        if let Some(path) = Self::path()
            && let Ok(text) = std::fs::read_to_string(&path)
            && let Ok(config) = serde_json::from_str::<Self>(&text)
        {
            return config;
        }
        let mut config = Self::default();
        if let Ok(key) = std::env::var("OPENAI_API_KEY") {
            config.api_key = key;
        }
        if let Ok(base) = std::env::var("OPENAI_BASE_URL") {
            config.base_url = base;
        }
        if let Ok(model) = std::env::var("OPENAI_MODEL") {
            config.model = model;
        }
        config
    }

    /// 写回配置文件。
    pub fn save(&self) -> std::io::Result<()> {
        let Some(path) = Self::path() else {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "HOME is not set",
            ));
        };
        std::fs::write(path, serde_json::to_string_pretty(self)?)
    }
}