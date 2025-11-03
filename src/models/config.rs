use crate::models::registry::{HubModelInfo, ModelFamily};
use crate::utils::load::{load_gguf, load_tokenizer};
use anyhow::Result;
use candle::Device;
use candle_transformers::models::{quantized_llama, quantized_qwen2, quantized_qwen3};
use tokenizers::Tokenizer;

#[derive(Debug, Clone)]
pub struct HParams {
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
}

impl Default for HParams {
    fn default() -> Self {
        Self {
            sample_len: 1000,
            temperature: 0.8,
            top_p: None,
            seed: 299792458,
            repeat_penalty: 1.1,
            repeat_last_n: 64,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Config {
    pub hparams: HParams,

    /// The device to use for inference.
    pub device: Device,

    /// The model to use.
    pub model: ModelFamily,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            hparams: HParams::default(),
            device: candle::Device::cuda_if_available(0).unwrap(),
            model: ModelFamily::default_qwen3(),
        }
    }
}

impl Config {
    /// 创建使用特定模型的配置
    pub fn with_model(model: ModelFamily) -> Self {
        Self {
            model,
            ..Default::default()
        }
    }

    /// 创建 Qwen2 配置
    pub fn qwen2(variant: &str) -> Self {
        Self::with_model(ModelFamily::qwen2(variant))
    }

    /// 创建 Qwen3 配置
    pub fn qwen3(variant: &str) -> Self {
        Self::with_model(ModelFamily::qwen3(variant))
    }

    pub fn default_qwen2() -> Self {
        Self {
            model: ModelFamily::default_qwen2(),
            ..Default::default()
        }
    }

    pub fn default_qwen3() -> Self {
        Self {
            model: ModelFamily::default_qwen3(),
            ..Default::default()
        }
    }

    /// 获取模型配置信息
    pub fn get_hub_model_info(&self) -> Result<HubModelInfo> {
        self.model.get_config()
    }

    /// 设置模型并加载相关组件
    pub async fn setup_model(&self) -> Result<(Box<dyn crate::models::Forward>, u32)> {
        let info = self.get_hub_model_info()?;
        let (mut file, ct) = load_gguf(&info.model_repo, &info.model_file).await?;
        let eos_token_id = ct
            .metadata
            .get("tokenizer.ggml.eos_token_id")
            .unwrap()
            .to_u32()?;

        // 根据模型类型创建相应的模型权重
        match &self.model {
            ModelFamily::Qwen2(_) => {
                let model = quantized_qwen2::ModelWeights::from_gguf(ct, &mut file, &self.device)?;
                Ok((Box::new(model), eos_token_id))
            }
            ModelFamily::Qwen3(_) => {
                let model = quantized_qwen3::ModelWeights::from_gguf(ct, &mut file, &self.device)?;
                Ok((Box::new(model), eos_token_id))
            }
            ModelFamily::Llama(_) => {
                let model = quantized_llama::ModelWeights::from_gguf(ct, &mut file, &self.device)?;
                Ok((Box::new(model), eos_token_id))
            }
        }
    }

    /// 设置分词器
    pub async fn setup_tokenizer(&self) -> Result<Tokenizer> {
        let model_config = self.get_hub_model_info()?;
        load_tokenizer(&model_config.tokenizer_repo).await
    }
}
