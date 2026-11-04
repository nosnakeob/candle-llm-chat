use crate::model::hub::{HubInfo, ModelArch};
use crate::model::{ModelInference, Qwen3MoEModel};
use crate::utils::hf::{fetch, model_repo};
use crate::utils::load::{download_gguf, download_safetensors, load_tokenizer};
use anyhow::Result;
use candle::quantized::gguf_file::Content;
use candle::quantized::tokenizer::TokenizerFromGguf;
use candle::{DType, Device};
use candle_nn::VarBuilder;
use candle_transformers::models::{
    quantized_qwen3,
    qwen3::{Config as Qwen3Config, ModelForCausalLM as Qwen3Model},
};
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
            // temp 0.7 + top_p 0.8：Qwen3.5 词表 248k，无 top_p 截断时长尾噪声
            // 会采进重复吸引子（实测 0.8B 无截断时输出死循环，贪心/top_p 均正常）
            temperature: 0.7,
            top_p: Some(0.8),
            seed: 299792458,
            repeat_penalty: 1.1,
            repeat_last_n: 64,
            device: candle::Device::cuda_if_available(0).unwrap(),
        }
    }
}

/// 由 config.json 判定模型架构。
///
/// 原先这里硬编码 `ModelArch::Qwen3`，导致下面 match 里 Qwen3VL / Qwen3MoE /
/// Qwen3_5 三个分支是死代码，非 Qwen3 的 safetensors 仓库会被静默按 Qwen3 解析
/// （config 字段对不上时只会报一个与真实原因无关的反序列化错误）。
///
/// 判据：优先 `model_type`（HF 标准字段，实测取值 qwen3 / qwen3_vl / qwen3_5 /
/// qwen3_moe），缺失时退回 `architectures[0]`（Qwen3ForCausalLM /
/// Qwen3VLForConditionalGeneration / Qwen3_5ForConditionalGeneration /
/// Qwen3MoeForCausalLM）。两者都认不出时明确报错，不猜。
fn arch_from_config(config: &serde_json::Value) -> Result<ModelArch> {
    let model_type = config.get("model_type").and_then(|v| v.as_str());
    let arch0 = config
        .get("architectures")
        .and_then(|v| v.as_array())
        .and_then(|a| a.first())
        .and_then(|v| v.as_str());

    if let Some(mt) = model_type {
        match mt {
            "qwen3" => return Ok(ModelArch::Qwen3),
            "qwen3_vl" | "qwen2_vl" => return Ok(ModelArch::Qwen3VL),
            "qwen3_moe" | "qwen2_moe" => return Ok(ModelArch::Qwen3MoE),
            "qwen3_5" | "qwen35" => return Ok(ModelArch::Qwen3_5),
            _ => {}
        }
    }

    if let Some(name) = arch0 {
        // 顺序敏感：Qwen3Moe* 与 Qwen3_5* 都以 Qwen3 开头，先判更具体的。
        let n = name.to_ascii_lowercase();
        if n.contains("moe") {
            return Ok(ModelArch::Qwen3MoE);
        }
        if n.starts_with("qwen3_5") || n.starts_with("qwen35") {
            return Ok(ModelArch::Qwen3_5);
        }
        if n.contains("vl") {
            return Ok(ModelArch::Qwen3VL);
        }
        if n.starts_with("qwen3") || n.starts_with("qwen2") {
            return Ok(ModelArch::Qwen3);
        }
    }

    bail!(
        "无法识别的模型架构：model_type={model_type:?} architectures={arch0:?}。\
         请确认仓库的 config.json 完整，或改用受支持的架构。"
    )
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

        // candle 自带的 GGUF → tokenizers 转换
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
                    candle_transformers::models::quantized_qwen3_moe::GGUFQWenMoE::from_gguf(
                        ct,
                        &mut file,
                        device,
                        DType::F32,
                    )?;
                Box::new(Qwen3MoEModel::new(model))
            }
            "qwen3.5" | "qwen35" => {
                let model =
                    crate::model::qwen3_5::QuantizedModelWeights::from_gguf(ct, &mut file, device)?;
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
        let repo = model_repo(&hub_info.model_repo)?;

        // 加载模型权重文件（单文件优先，缺失时回退分片）
        let model_files = download_safetensors(&repo, &hub_info.model_file).await?;

        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&model_files, DType::BF16, device)? };

        // 加载配置文件
        let config_path = fetch(&repo, "config.json").await?;
        let config_content = std::fs::read(&config_path)?;

        // 架构必须由 config.json 判定，不能写死：写死会让下面 match 的
        // Qwen3VL / Qwen3MoE / Qwen3_5 三个分支永远走不到。
        let config_value: serde_json::Value = serde_json::from_slice(&config_content)?;
        let arch = arch_from_config(&config_value)?;

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

        let tokenizer = load_tokenizer(&hub_info.tokenizer_repo).await?;

        Ok((model, tokenizer))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::registry::ModelRegistry;

    /// C2 回归：架构由 config.json 判定，四个 ModelArch 分支都必须可达。
    ///
    /// 这里覆盖两条判据（model_type 与 architectures[0]），取值来自本机 HF 缓存
    /// 里四份真实 config.json（Qwen3-4B-Instruct-2507 / Qwen3-VL-2B-Instruct /
    /// Qwen3.8-27B / Qwen3.5-0.8B），MoE 的架构名按 HF 命名惯例（Qwen3MoeForCausalLM）。
    #[test]
    fn test_arch_from_config_dispatch() -> Result<()> {
        let cases = [
            (serde_json::json!({"model_type": "qwen3"}), ModelArch::Qwen3),
            (serde_json::json!({"model_type": "qwen3_vl"}), ModelArch::Qwen3VL),
            (serde_json::json!({"model_type": "qwen3_moe"}), ModelArch::Qwen3MoE),
            (serde_json::json!({"model_type": "qwen3_5"}), ModelArch::Qwen3_5),
            (
                serde_json::json!({"architectures": ["Qwen3ForCausalLM"]}),
                ModelArch::Qwen3,
            ),
            (
                serde_json::json!({"architectures": ["Qwen3VLForConditionalGeneration"]}),
                ModelArch::Qwen3VL,
            ),
            (
                serde_json::json!({"architectures": ["Qwen3MoeForCausalLM"]}),
                ModelArch::Qwen3MoE,
            ),
            (
                serde_json::json!({"architectures": ["Qwen3_5ForConditionalGeneration"]}),
                ModelArch::Qwen3_5,
            ),
        ];
        for (cfg, want) in cases {
            assert_eq!(arch_from_config(&cfg)?, want, "config={cfg}");
        }

        // 认不出时必须明确报错，绝不能再静默按 Qwen3 解析
        let unknown = serde_json::json!({
            "model_type": "llama",
            "architectures": ["LlamaForCausalLM"]
        });
        let err = arch_from_config(&unknown).unwrap_err().to_string();
        assert!(err.contains("无法识别的模型架构"), "错误信息不够明确: {err}");
        Ok(())
    }

    #[tokio::test]
    #[ignore] // 需要下载模型文件
    async fn test_model_loader_load() -> Result<()> {
        let device = Device::cuda_if_available(0)?;
        let registry = ModelRegistry::new()?;

        // 测试加载 GGUF 量化模型
        assert!(
            ModelLoader::load(registry.get("qwen3.4b_q4")?, &device)
                .await
                .is_ok()
        );

        // 测试加载 Safetensors 完整模型
        assert!(
            ModelLoader::load(registry.get("qwen3.4b_base")?, &device)
                .await
                .is_ok()
        );

        // 测试加载不存在的模型
        assert!(
            ModelLoader::load(registry.get("nonexistent_model")?, &device)
                .await
                .is_err()
        );

        Ok(())
    }
}
