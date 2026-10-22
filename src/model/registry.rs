use crate::model::hub::{HubInfo, HubInfoRaw, ModelArch};
use anyhow::{Error, Result};
use config::Config;
use serde::Deserialize;
use std::collections::HashMap;
use std::str::FromStr;

/// models.toml 的原始结构：按架构名分组的模型表
///
/// 用 flatten 把所有 `[<架构>.<变体>]` 收进一张表，加新架构时
/// 只需在 ModelArch 里加一个枚举值，这里不用改。
#[derive(Debug, Deserialize)]
pub struct ModelRegistryRaw {
    #[serde(flatten)]
    pub models: HashMap<String, HashMap<String, HubInfoRaw>>,
}

/// 模型注册表：架构 -> (变体名 -> 配置)
#[derive(Debug)]
pub struct ModelRegistry {
    models: HashMap<ModelArch, HashMap<String, HubInfo>>,
}

impl ModelRegistry {
    pub fn new() -> Result<Self> {
        let raw: ModelRegistryRaw = Config::builder()
            .add_source(config::File::with_name("models.toml"))
            .build()?
            .try_deserialize()
            .map_err(Error::from)?;

        let mut models = HashMap::with_capacity(raw.models.len());
        for (arch_str, entries) in raw.models {
            // 未知架构直接报错，避免配置写错时被静默忽略
            let arch = ModelArch::from_str(&arch_str)
                .map_err(|_| anyhow!("models.toml 里有不认识的架构名: {arch_str}"))?;
            let entries = entries
                .into_iter()
                .map(|(name, raw)| (name, HubInfo::from(raw)))
                .collect();
            models.insert(arch, entries);
        }

        Ok(Self { models })
    }

    /// 获取模型配置
    ///
    /// - `"qwen3.8b_q4"`：取指定变体
    /// - `"qwen3"`：取该架构下 `default = true` 的变体
    pub fn get(&self, model_id: &str) -> Result<&HubInfo> {
        let (arch_str, variant) = match model_id.split_once('.') {
            Some((arch, variant)) => (arch, Some(variant)),
            None => (model_id, None),
        };

        let arch =
            ModelArch::from_str(arch_str).map_err(|_| anyhow!("不认识的架构名: {arch_str}"))?;
        let models = self
            .models
            .get(&arch)
            .ok_or_else(|| anyhow!("架构 '{arch_str}' 未在 models.toml 中配置"))?;

        match variant {
            Some(variant) => models
                .get(variant)
                .ok_or_else(|| anyhow!("模型变体 '{variant}' 不存在")),
            None => models
                .values()
                .find(|config| config.default)
                .ok_or_else(|| anyhow!("架构 '{arch_str}' 没有 default = true 的模型")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_model() -> Result<()> {
        let registry = ModelRegistry::new()?;

        // 指定变体
        let model = registry.get("qwen3.4b_q4")?;
        assert_eq!(model.model_repo, "byteshape/Qwen3-4B-Instruct-2507-GGUF");

        // 基础模型
        let model = registry.get("qwen3.8b_base")?;
        assert_eq!(model.model_repo, "Qwen/Qwen3-8B");

        // 只给架构名：取 default
        let default_qwen3 = registry.get("qwen3")?;
        assert_eq!(default_qwen3.model_repo, "Qwen/Qwen3-8B-GGUF");
        assert!(default_qwen3.default);

        // 不存在的架构和变体均报错
        assert!(registry.get("unknown").is_err());
        assert!(registry.get("qwen3.NonExistent").is_err());

        Ok(())
    }

    /// tokenizer_repo 省略时回退到 model_repo；显式指定时不被覆盖
    #[test]
    fn test_tokenizer_repo_resolution() -> Result<()> {
        let registry = ModelRegistry::new()?;

        // 基础模型：没有单独 tokenizer，等于自身
        assert_eq!(
            registry.get("qwen3.8b_base")?.tokenizer_repo,
            "Qwen/Qwen3-8B"
        );

        // GGUF 量化：tokenizer 来自非 GGUF 仓库（显式配置）
        assert_eq!(registry.get("qwen3.8b_q4")?.tokenizer_repo, "Qwen/Qwen3-8B");
        assert_eq!(
            registry.get("qwen3.4b_q4")?.tokenizer_repo,
            "Qwen/Qwen3-4B-Instruct-2507"
        );

        // GGUF 仓库自带 tokenizer：回退为自身
        assert_eq!(
            registry.get("qwen3_5.4b_q4")?.tokenizer_repo,
            "unsloth/Qwen3.5-4B-GGUF"
        );

        Ok(())
    }
}
