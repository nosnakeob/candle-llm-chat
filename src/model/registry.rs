use crate::model::hub::{HubInfo, HubInfoRaw, ModelArch};
use anyhow::{Error, Result};
use config::Config;
use serde::Deserialize;
use std::{collections::HashMap, str::FromStr};

#[derive(Debug, Deserialize)]
pub struct ModelRegistryRaw {
    pub qwen3: HashMap<String, HubInfoRaw>,
    pub llama: Option<HashMap<String, HubInfoRaw>>,
}

#[derive(Debug)]
pub struct ModelRegistry {
    pub qwen3: HashMap<String, HubInfo>,
    pub llama: Option<HashMap<String, HubInfo>>,
}

impl ModelRegistry {
    pub fn new() -> Result<Self> {
        let raw_registry: ModelRegistryRaw = Config::builder()
            .add_source(config::File::with_name("models.toml"))
            .build()?
            .try_deserialize()
            .map_err(Error::from)?;

        // 处理 tokenizer_repo 的自动填充并转换为最终结构
        Ok(Self::from_raw(raw_registry))
    }

    /// 从原始配置转换为最终配置
    fn from_raw(mut raw: ModelRegistryRaw) -> Self {
        // 处理 qwen3 系列
        Self::fill_arch_tokenizer_repos(&mut raw.qwen3);
        let qwen3 = raw.qwen3.into_iter()
            .map(|(k, v)| (k, HubInfo::from(v)))
            .collect();

        // 处理 llama 系列
        let llama = raw.llama.map(|mut llama_models| {
            Self::fill_arch_tokenizer_repos(&mut llama_models);
            llama_models.into_iter()
                .map(|(k, v)| (k, HubInfo::from(v)))
                .collect()
        });

        Self { qwen3, llama }
    }

    /// 为特定架构的模型填充 tokenizer_repo
    fn fill_arch_tokenizer_repos(models: &mut HashMap<String, HubInfoRaw>) {
        // 第一步：为 base 模型设置 tokenizer_repo
        let mut base_tokenizers = HashMap::new();
        
        for (variant_name, hub_info) in models.iter_mut() {
            if variant_name.ends_with("_base") {
                if hub_info.tokenizer_repo.is_none() {
                    hub_info.tokenizer_repo = Some(hub_info.model_repo.clone());
                }
                // 记录 base 模型的 tokenizer_repo 供其他变体使用
                if let Some(ref tokenizer_repo) = hub_info.tokenizer_repo {
                    let base_key = variant_name.strip_suffix("_base").unwrap();
                    base_tokenizers.insert(base_key.to_string(), tokenizer_repo.clone());
                }
            }
        }

        // 第二步：为非 base 模型设置 tokenizer_repo
        for (variant_name, hub_info) in models.iter_mut() {
            if !variant_name.ends_with("_base") && hub_info.tokenizer_repo.is_none() {
                // 提取基础名称（如 "8b_q4" -> "8b"）
                let base_name = if let Some(pos) = variant_name.rfind('_') {
                    &variant_name[..pos]
                } else {
                    variant_name
                };
                
                // 查找对应的 base 模型的 tokenizer_repo
                if let Some(tokenizer_repo) = base_tokenizers.get(base_name) {
                    hub_info.tokenizer_repo = Some(tokenizer_repo.clone());
                }
            }
        }
    }

    /// 获取模型配置
    ///
    /// # 参数
    /// - `model_id`: 模型标识符
    ///   - 格式1: "qwen3.8b_q4" - 获取量化模型
    ///   - 格式2: "qwen3.8b_base" - 获取官方完整模型
    ///   - 格式3: "qwen3" - 获取默认模型
    ///
    /// # 示例
    /// ```
    /// let registry = ModelRegistry::load()?;
    /// let quantized = registry.get("qwen3.8b_q4")?;   // 量化模型
    /// let official = registry.get("qwen3.8b_full")?;  // 官方模型
    /// let default = registry.get("qwen3")?;           // 默认模型
    /// ```
    pub fn get(&self, model_id: &str) -> Result<&HubInfo> {
        let (arch_str, variant) = match model_id.split_once('.') {
            Some((arch, variant)) => (arch, Some(variant)),
            None => (model_id, None),
        };

        let models = match ModelArch::from_str(arch_str)? {
            ModelArch::Qwen3 => &self.qwen3,
            ModelArch::Llama => self
                .llama
                .as_ref()
                .ok_or_else(|| anyhow!("Llama 模型未配置"))?,
            _ => bail!("不支持的模型架构: {}", arch_str),
        };

        match variant {
            Some(variant) => models
                .get(variant)
                .ok_or_else(|| anyhow!("模型变体 '{}' 不存在", variant)),
            None => models
                .values()
                .find(|config| config.default)
                .ok_or_else(|| anyhow!("架构 '{}' 没有默认模型", arch_str)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_registry_parse() -> Result<()> {
        let registry = ModelRegistry::new()?;
        dbg!(&registry);

        assert!(!registry.qwen3.is_empty());

        Ok(())
    }

    #[test]
    fn test_get_model() -> Result<()> {
        let registry = ModelRegistry::new()?;

        // 测试获取量化模型
        let model = registry.get("qwen3.4b_q4")?;
        assert_eq!(model.model_repo, "Qwen/Qwen3-4B-GGUF");
        dbg!(model);

        // 测试获取基础模型（Safetensors）
        let model = registry.get("qwen3.8b_base")?;
        assert_eq!(model.model_repo, "Qwen/Qwen3-8B");

        // 测试获取默认模型 - 仅使用架构名
        let default_qwen3 = registry.get("qwen3")?;
        assert_eq!(default_qwen3.model_repo, "Qwen/Qwen3-8B");
        assert!(default_qwen3.default);

        // 测试不存在的架构
        assert!(registry.get("unknown").is_err());

        // 测试不存在的变体
        assert!(registry.get("qwen3.NonExistent").is_err());

        Ok(())
    }

    #[test]
    fn test_tokenizer_repo_auto_fill() -> Result<()> {
        let registry = ModelRegistry::new()?;

        // 测试 base 模型的 tokenizer_repo 自动填充
        let base_model = registry.get("qwen3.8b_base")?;
        assert_eq!(base_model.tokenizer_repo, "Qwen/Qwen3-8B");

        // 测试量化模型的 tokenizer_repo 自动从对应 base 模型获取
        let q4_model = registry.get("qwen3.8b_q4")?;
        assert_eq!(q4_model.tokenizer_repo, "Qwen/Qwen3-8B");

        // 测试另一个 base 模型
        let base_4b = registry.get("qwen3.4b_base")?;
        assert_eq!(base_4b.tokenizer_repo, "Qwen/Qwen3-4B");

        // 测试对应的量化模型
        let q4_4b = registry.get("qwen3.4b_q4")?;
        assert_eq!(q4_4b.tokenizer_repo, "Qwen/Qwen3-4B");

        // 测试已经配置了 tokenizer_repo 的模型（不应该被覆盖）
        if let Some(llama_models) = &registry.llama {
            if let Some(deepseek_model) = llama_models.get("8b_deepseek_r1_q4") {
                assert_eq!(
                    deepseek_model.tokenizer_repo,
                    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
                );
            }
        }

        Ok(())
    }
}
