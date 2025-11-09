use crate::model::hub::{HubInfo, ModelArch, ModelInfo};
use crate::model::Forward;
use crate::model::registry::ModelRegistry;
use crate::utils::load::{download_gguf, load_tokenizer};
use anyhow::{Result, anyhow};
use candle::Device;
use candle::quantized::gguf_file::Content;
use candle_transformers::models::{quantized_llama, quantized_qwen2, quantized_qwen3};
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
        model_id: &str,
        device: &Device,
    ) -> Result<(Box<dyn Forward>, Tokenizer, ModelInfo)> {
        // 从注册表获取模型配置
        let registry = ModelRegistry::load()?;
        let hub_info = registry.get(model_id).unwrap();

        let (model_pth, tokenizer_pth) = hub_info.load().await?;

        let mut file = File::open(model_pth)?;
        let ct = Content::read(&mut file)?;

        let model_info = ModelInfo::try_from(&ct)?;

        let model = match model_info.arch {
            ModelArch::Qwen2 => {
                let model = quantized_qwen2::ModelWeights::from_gguf(ct, &mut file, device)?;
                Box::new(model) as Box<dyn Forward>
            }
            ModelArch::Qwen3 => {
                let model = quantized_qwen3::ModelWeights::from_gguf(ct, &mut file, device)?;
                Box::new(model) as Box<dyn Forward>
            }
            ModelArch::Llama => {
                let model = quantized_llama::ModelWeights::from_gguf(ct, &mut file, device)?;
                Box::new(model) as Box<dyn Forward>
            }
        };

        // 加载分词器
        let tokenizer = load_tokenizer(&hub_info.tokenizer_repo).await?;

        Ok((model, tokenizer, model_info))
    }
}
