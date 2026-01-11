use anyhow::Result;
use candle::quantized::gguf_file::Content;
use derive_new::new;
use hf_hub::api::tokio::ApiBuilder;
use serde::Deserialize;
use std::{default, path::PathBuf};
use strum::{Display, EnumString};
use tokenizers::Tokenizer;

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ModelType {
    Gguf,
    Safetensors,
}

/// models.toml单个仓库配置（原始配置）
#[serde_inline_default]
#[derive(Debug, Clone, Deserialize)]
pub struct HubInfoRaw {
    pub model_repo: String,
    #[serde_inline_default("model.safetensors".to_string())]
    pub model_file: String,
    pub tokenizer_repo: Option<String>,
    #[serde(default)]
    pub default: bool,
}

/// 处理后的模型配置（tokenizer_repo 已确定）
#[derive(Debug, Clone)]
pub struct HubInfo {
    pub model_repo: String,
    pub model_file: String,
    pub tokenizer_repo: String,
    pub default: bool,
}

impl From<HubInfoRaw> for HubInfo {
    fn from(raw: HubInfoRaw) -> Self {
        Self {
            model_repo: raw.model_repo.clone(),
            model_file: raw.model_file,
            tokenizer_repo: raw.tokenizer_repo.unwrap_or(raw.model_repo),
            default: raw.default,
        }
    }
}

#[derive(Debug, Clone, EnumString, Display)]
#[strum(serialize_all = "lowercase")]
pub enum ModelArch {
    Qwen3,
    Llama,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        model::{hub, registry::ModelRegistry},
        utils::load::{download_gguf, load_tokenizer},
    };
    use std::{collections::HashMap, fs::File};

    #[tokio::test]
    async fn test_hub_info_load() -> Result<()> {
        let toml_str = r#"
            # [qwen3]
            [4b_base]
            model_repo = "Qwen/Qwen3-4B"

            [4b_q4]
            model_repo = "Qwen/Qwen3-4B-GGUF"
            model_file = "Qwen3-4B-Q4_K_M.gguf"
            default = true
        "#;

        let qwen3: HashMap<String, HubInfoRaw> = toml::from_str(toml_str)?;

        dbg!(&qwen3);

        Ok(())
    }

    #[test]
    fn test_hub_info_conversion() -> Result<()> {
        // 测试 HubInfoRaw 到 HubInfo 的转换
        let raw = HubInfoRaw {
            model_repo: "Qwen/Qwen3-8B".to_string(),
            model_file: "model.safetensors".to_string(),
            tokenizer_repo: None, // 测试自动填充
            default: true,
        };

        let hub_info = HubInfo::from(raw);

        assert_eq!(hub_info.model_repo, "Qwen/Qwen3-8B");
        assert_eq!(hub_info.model_file, "model.safetensors");
        assert_eq!(hub_info.tokenizer_repo, "Qwen/Qwen3-8B"); // 自动填充
        assert!(hub_info.default);

        Ok(())
    }
}
