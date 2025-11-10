use crate::utils::load::download_gguf;
use anyhow::Result;
use candle::quantized::gguf_file::Content;
use derive_new::new;
use hf_hub::api::tokio::ApiBuilder;
use serde::Deserialize;
use std::path::PathBuf;
use strum::{Display, EnumString};
use tokenizers::Tokenizer;

#[derive(Debug, Deserialize, Clone, new)]
pub struct HubInfo {
    pub model_repo: String,
    pub model_file: String,
    pub tokenizer_repo: String,
    #[new(default)]
    #[serde(default)]
    pub default: bool,
}

impl HubInfo {
    pub async fn load(&self) -> Result<(PathBuf, PathBuf)> {
        let config = config::Config::builder()
            .add_source(config::File::with_name("config.toml"))
            .build()?;
        let token = config.get_string("huggingface.token")?;

        let pth = ApiBuilder::new()
            .with_token(Some(token))
            .build()?
            .model(self.tokenizer_repo.clone())
            .get("tokenizer.json")
            .await?;

        Ok((
            download_gguf(&self.model_repo, &self.model_file).await?,
            pth,
        ))
    }
}

#[derive(Debug, Clone, EnumString, Display)]
#[strum(serialize_all = "lowercase")]
pub enum ModelArch {
    Qwen2,
    Qwen3,
    Llama,
}

#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub arch: ModelArch,
    pub eos_token_id: u32,
    pub chat_template: String,
}

impl TryFrom<&Content> for ModelInfo {
    type Error = anyhow::Error;

    fn try_from(ct: &Content) -> Result<Self> {
        let eos_token_id = ct
            .metadata
            .get("tokenizer.ggml.eos_token_id")
            .unwrap()
            .to_u32()?;
        let chat_template = ct
            .metadata
            .get("tokenizer.chat_template")
            .unwrap()
            .to_string()?
            .clone();
        let arch_str = ct
            .metadata
            .get("general.architecture")
            .unwrap()
            .to_string()?;
        let arch: ModelArch = arch_str.parse()?;
        Ok(Self {
            eos_token_id,
            chat_template,
            arch,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::fs::File;

    use super::*;

    #[tokio::test]
    async fn test_hub_info_load() -> Result<()> {
        let hub_info = HubInfo::new(
            "Qwen/Qwen3-8B-GGUF".to_string(),
            "Qwen3-8B-Q4_K_M".to_string(),
            "Qwen/Qwen3-8B".to_string(),
        );
        let (model_pth, tokenizer_pth) = hub_info.load().await?;
        assert!(!hub_info.default);
        assert!(model_pth.exists());
        assert!(tokenizer_pth.exists());

        Ok(())
    }

    #[tokio::test]
    async fn test_content2model_info() -> Result<()> {
        let hub_info = HubInfo::new(
            "Qwen/Qwen3-8B-GGUF".to_string(),
            "Qwen3-8B-Q4_K_M".to_string(),
            "Qwen/Qwen3-8B".to_string(),
        );
        let (model_pth, tokenizer_pth) = hub_info.load().await?;

        let mut file = File::open(model_pth)?;
        let ct = Content::read(&mut file)?;
        let model_info = ModelInfo::try_from(&ct)?;

        dbg!(model_info);

        Ok(())
    }
}
