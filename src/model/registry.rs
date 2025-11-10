use crate::model::hub::HubInfo;
use anyhow::{Error, Result};
use config::Config;
use serde::Deserialize;
use std::collections::HashMap;

#[derive(Debug, Deserialize)]
pub struct ModelRegistry {
    pub qwen2: HashMap<String, HubInfo>,
    pub qwen3: HashMap<String, HubInfo>,
    pub llama: Option<HashMap<String, HubInfo>>,
}

impl ModelRegistry {
    pub fn load() -> Result<Self> {
        Config::builder()
            .add_source(config::File::with_name("models.toml"))
            .build()?
            .try_deserialize()
            .map_err(Error::from)
    }

    /// 获取模型配置
    ///
    /// # 参数
    /// - `model_id`: 模型标识符
    ///   - 格式1: "qwen2.W25_14b" - 获取特定模型
    ///   - 格式2: "qwen2" - 获取默认模型
    ///
    /// # 示例
    /// ```
    /// let registry = ModelRegistry::load()?;
    /// let specific = registry.get("qwen2.W25_14b")?;  // 特定模型
    /// let default = registry.get("qwen2")?;           // 默认模型
    /// ```
    pub fn get(&self, model_id: &str) -> Option<&HubInfo> {
        match model_id.split_once('.') {
            Some((arch, variant)) => self.get_by_arch_and_variant(arch, variant),
            None => self.get_default_by_arch(model_id),
        }
    }

    /// 根据架构和变体获取特定模型
    fn get_by_arch_and_variant(&self, arch: &str, variant: &str) -> Option<&HubInfo> {
        match arch {
            "qwen2" => self.qwen2.get(variant),
            "qwen3" => self.qwen3.get(variant),
            "llama" => self.llama.as_ref()?.get(variant),
            _ => None,
        }
    }

    /// 根据架构获取默认模型
    fn get_default_by_arch(&self, arch: &str) -> Option<&HubInfo> {
        let models = match arch {
            "qwen2" => &self.qwen2,
            "qwen3" => &self.qwen3,
            "llama" => self.llama.as_ref()?,
            _ => return None,
        };

        models.values().find(|config| config.default)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_registry_parse() -> Result<()> {
        let registry = ModelRegistry::load()?;

        // 验证加载的配置
        assert_eq!(registry.qwen2.len(), 3);
        assert_eq!(registry.qwen3.len(), 4);

        Ok(())
    }

    #[test]
    fn test_get_specific_model() -> Result<()> {
        let registry = ModelRegistry::load()?;

        // 测试获取特定模型 - 使用新的格式 "arch.variant"
        let model = registry.get("qwen2.W25_14b").unwrap();
        assert_eq!(model.model_repo, "Qwen/Qwen2.5-14B-Instruct-GGUF");
        assert_eq!(model.model_file, "qwen2.5-14b-instruct-q4_0");
        assert_eq!(model.tokenizer_repo, "Qwen/Qwen2.5-14B-Instruct");
        assert!(!model.default);

        // 测试另一个特定模型
        let model = registry.get("qwen3.W3_14b").unwrap();
        assert_eq!(model.model_repo, "Qwen/Qwen3-14B-GGUF");

        Ok(())
    }

    #[test]
    fn test_get_default_model() -> Result<()> {
        let registry = ModelRegistry::load()?;

        // 测试获取默认模型 - 仅使用架构名
        let default_qwen2 = registry.get("qwen2").unwrap();
        assert_eq!(default_qwen2.model_repo, "Qwen/Qwen2.5-7B-Instruct-GGUF");
        assert!(default_qwen2.default);

        let default_qwen3 = registry.get("qwen3").unwrap();
        assert_eq!(default_qwen3.model_repo, "Qwen/Qwen3-8B-GGUF");
        assert!(default_qwen3.default);

        Ok(())
    }

    #[test]
    fn test_unified_get_method() -> Result<()> {
        let registry = ModelRegistry::load()?;

        // 测试统一的 get 方法
        // 1. 获取默认模型
        let default = registry.get("qwen2").unwrap();
        assert!(default.default);

        // 2. 获取特定模型
        let specific = registry.get("qwen2.W25_14b").unwrap();
        assert!(!specific.default);

        // 3. 测试不存在的架构
        assert!(registry.get("unknown").is_none());

        // 4. 测试不存在的变体
        assert!(registry.get("qwen2.NonExistent").is_none());

        Ok(())
    }
}
