use crate::model::{ModelInference, Qwen3MoEModel};
use crate::model::hub::{HubInfo, ModelArch, ModelType};
use crate::model::registry::ModelRegistry;
use crate::utils::load::ApiRepoExt;
use crate::utils::load::{download_gguf, load_tokenizer};
use anyhow::{Result, anyhow};
use candle::quantized::gguf_file::Content;
use candle::quantized::tokenizer::TokenizerFromGguf;
use candle::{DType, Device};
use candle_nn::VarBuilder;
use candle_transformers::models::{
    quantized_qwen3,
    qwen3::{Config as Qwen3Config, ModelForCausalLM as Qwen3Model},
};
use hf_hub::api::tokio::{Api, ApiBuilder};
use serde_json::Value;
use std::fs::File;
use tokenizers::Tokenizer;

/// 推理参数配置
#[derive(Debug, Clone)]
pub struct InferenceConfig {
    /// The length of the sample to generate (in tokens).
    pub sample_len: usize,

    /// The temperature used to generate samples, use 0 for greedy sampling.
    pub temperature: f64,

    /// Nucleus sampling probability cutoff.
    pub top_p: Option<f64>,

    /// The seed to use when generating random samples.
    pub seed: u64,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    pub repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    pub repeat_last_n: usize,

    /// The device to use for inference.
    pub device: Device,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            sample_len: 1000,
            temperature: 0.8,
            top_p: None,
            seed: 299792458,
            repeat_penalty: 1.1,
            repeat_last_n: 64,
            device: candle::Device::cuda_if_available(0).unwrap(),
        }
    }
}

/// 模型加载器 - 专门负责模型相关操作
pub struct ModelLoader;

impl ModelLoader {
    /// 一次性加载所有需要的组件
    pub async fn load(
        hub_info: &HubInfo,
        device: &Device,
    ) -> Result<(Box<dyn ModelInference>, Tokenizer)> {
        if hub_info.model_repo.to_lowercase().contains("gguf") {
            Self::load_gguf(hub_info, device).await
        } else {
            Self::load_safetensors(hub_info, device).await
        }
    }

    /// 加载 GGUF 量化模型
    async fn load_gguf(
        hub_info: &HubInfo,
        device: &Device,
    ) -> Result<(Box<dyn ModelInference>, Tokenizer)> {
        let model_pth = download_gguf(&hub_info.model_repo, &hub_info.model_file).await?;

        let mut file = File::open(model_pth)?;
        let ct = Content::read(&mut file)?;

        let tokenizer = Tokenizer::from_gguf(&ct)?;

        // 从 GGUF metadata 读取架构名，比 repo 名匹配更准确
        let arch = ct
            .metadata
            .get("general.architecture")
            .and_then(|v| v.to_string().ok())
            .cloned()
            .unwrap_or_default();

        let model: Box<dyn ModelInference> = match arch.as_str() {
            "qwen3" | "qwen2" => {
                let model = quantized_qwen3::ModelWeights::from_gguf(ct, &mut file, device)?;
                Box::new(model)
            }
            "qwen3_moe" | "qwen2_moe" => {
                let model =
                    candle_transformers::models::quantized_qwen3_moe::GGUFQWenMoE::from_gguf(ct, &mut file, device, DType::F32)?;
                Box::new(Qwen3MoEModel { inner: model })
            }
            "qwen3.5" | "qwen35" => {
                let model = crate::model::qwen3_5::QuantizedModelWeights::from_gguf(
                    ct, &mut file, device,
                )?;
                Box::new(model)
            }
            _ => bail!("不支持的 GGUF 架构: {arch}"),
        };

        Ok((model, tokenizer))
    }

    /// 加载 Safetensors 完整模型 暂时支持qwen
    async fn load_safetensors(
        hub_info: &HubInfo,
        device: &Device,
    ) -> Result<(Box<dyn ModelInference>, Tokenizer)> {
        let api = ApiBuilder::from_env().build()?;
        let repo = api.model(hub_info.model_repo.clone());

        // 加载模型权重文件
        let model_files = match repo.get(&hub_info.model_file).await {
            Ok(single_file) => vec![single_file],
            Err(_) => {
                // 单文件不存在，尝试获取分片文件
                repo.get_safetensors().await?
            }
        };

        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&model_files, DType::BF16, device)? };

        let arch = ModelArch::Qwen3;

        // 加载配置文件
        let config_path = repo.get("config.json").await?;
        let config_content = std::fs::read(&config_path)?;

        let model: Box<dyn ModelInference> = match arch {
            ModelArch::Qwen3 => {
                let config: Qwen3Config = serde_json::from_slice(&config_content)?;
                let model = Qwen3Model::new(&config, vb)?;
                Box::new(model)
            }
            ModelArch::Qwen3VL => {
                bail!("Qwen3-VL safetensors 使用专用管线加载，不通过 ModelLoader")
            }
            ModelArch::Qwen3MoE => {
                bail!("Qwen3-MoE safetensors 加载暂未实现")
            }
            ModelArch::Qwen3_5 => {
                let config: crate::model::qwen3_5::Config =
                    serde_json::from_slice(&config_content)?;
                let model = crate::model::qwen3_5::ModelForCausalLM::new(&config, vb)?;
                Box::new(model)
            }
        };

        let tokenizer = load_tokenizer(&hub_info.tokenizer_repo)?;

        Ok((model, tokenizer))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore] // 需要下载模型文件
    async fn test_model_loader_load() -> Result<()> {
        let device = Device::cuda_if_available(0)?;
        let registry = ModelRegistry::new()?;

        // 测试加载 GGUF 量化模型
        assert!(ModelLoader::load(registry.get("qwen3.4b_q4")?, &device).await.is_ok());

        // 测试加载 Safetensors 完整模型
        assert!(ModelLoader::load(registry.get("qwen3.4b_base")?, &device).await.is_ok());

        // 测试加载不存在的模型
        assert!(
            ModelLoader::load(registry.get("nonexistent_model")?, &device)
                .await
                .is_err()
        );

        Ok(())
    }
}
