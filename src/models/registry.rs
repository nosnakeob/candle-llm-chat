use anyhow::Result;
use config::Config;
use serde::Deserialize;
use std::collections::HashMap;

#[derive(Debug, Deserialize, Clone)]
pub struct HubModelInfo {
    pub model_repo: String,
    pub model_file: String,
    pub tokenizer_repo: String,
    #[serde(default)]
    pub default: bool,
}

#[derive(Debug, Deserialize)]
pub struct ModelRegistry {
    pub qwen2: HashMap<String, HubModelInfo>,
    pub qwen3: HashMap<String, HubModelInfo>,
    pub llama: Option<HashMap<String, HubModelInfo>>,
}

impl ModelRegistry {
    pub fn load() -> Result<Self> {
        let config = Config::builder()
            .add_source(config::File::with_name("models.toml"))
            .build()?;

        Ok(config.try_deserialize()?)
    }

    pub fn get_model(&self, family: &str, variant: &str) -> Option<&HubModelInfo> {
        match family {
            "qwen2" => self.qwen2.get(variant),
            "qwen3" => self.qwen3.get(variant),
            "llama" => self.llama.as_ref()?.get(variant),
            _ => None,
        }
    }

    pub fn get_default(&self, family: &str) -> Option<&HubModelInfo> {
        let models = match family {
            "qwen2" => &self.qwen2,
            "qwen3" => &self.qwen3,
            "llama" => self.llama.as_ref()?,
            _ => return None,
        };

        models.values().find(|config| config.default)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelFamily {
    Qwen2(String),
    Qwen3(String),
    Llama(String),
}

impl ModelFamily {
    pub fn qwen2(variant: &str) -> Self {
        Self::Qwen2(variant.to_string())
    }

    pub fn qwen3(variant: &str) -> Self {
        Self::Qwen3(variant.to_string())
    }

    pub fn default_qwen2() -> Self {
        Self::Qwen2("W25_7b".to_string())
    }

    pub fn default_qwen3() -> Self {
        Self::Qwen3("W3_8b".to_string())
    }

    /// 获取模型配置信息
    pub fn get_config(&self) -> Result<HubModelInfo> {
        let registry = ModelRegistry::load()?;

        let (family, variant) = match self {
            ModelFamily::Qwen2(v) => ("qwen2", v.as_str()),
            ModelFamily::Qwen3(v) => ("qwen3", v.as_str()),
            ModelFamily::Llama(v) => ("llama", v.as_str()),
        };

        let config = registry
            .get_model(family, variant)
            .or_else(|| registry.get_default(family))
            .ok_or_else(|| anyhow::anyhow!("Model not found: {}:{}", family, variant))?;

        Ok(config.clone())
    }

    /// 获取模型仓库信息
    pub fn model_repo(&self) -> Result<String> {
        Ok(self.get_config()?.model_repo)
    }

    /// 获取模型文件名
    pub fn model_file(&self) -> Result<String> {
        Ok(self.get_config()?.model_file)
    }

    /// 获取分词器仓库信息
    pub fn tokenizer_repo(&self) -> Result<String> {
        Ok(self.get_config()?.tokenizer_repo)
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

        // 验证具体配置
        let qwen2_7b = registry.qwen2.get("W25_7b").unwrap();
        assert_eq!(qwen2_7b.model_repo, "Qwen/Qwen2.5-7B-Instruct-GGUF");
        assert!(qwen2_7b.default);

        let qwen3_8b = registry.qwen3.get("W3_8b").unwrap();
        assert_eq!(qwen3_8b.model_repo, "Qwen/Qwen3-8B-GGUF");
        assert!(qwen3_8b.default);

        Ok(())
    }

    #[test]
    fn test_get_specific_model() -> Result<()> {
        let registry = ModelRegistry::load()?;

        // 测试获取特定模型
        let model = registry.get_model("qwen2", "W25_14b").unwrap();
        assert_eq!(model.model_repo, "Qwen/Qwen2.5-14B-Instruct-GGUF");
        assert_eq!(model.model_file, "qwen2.5-14b-instruct-q4_0");
        assert_eq!(model.tokenizer_repo, "Qwen/Qwen2.5-14B-Instruct");
        assert!(!model.default);

        Ok(())
    }

    #[test]
    fn test_get_default_model() -> Result<()> {
        let registry = ModelRegistry::load()?;

        // 测试获取默认模型
        let default_qwen2 = registry.get_default("qwen2").unwrap();
        assert_eq!(default_qwen2.model_repo, "Qwen/Qwen2.5-7B-Instruct-GGUF");
        assert!(default_qwen2.default);

        let default_qwen3 = registry.get_default("qwen3").unwrap();
        assert_eq!(default_qwen3.model_repo, "Qwen/Qwen3-8B-GGUF");
        assert!(default_qwen3.default);

        Ok(())
    }

    #[test]
    fn test_model_family_creation() {
        // 测试 ModelFamily 的创建方法
        let qwen2_model = ModelFamily::qwen2("W25_14b");
        assert_eq!(qwen2_model, ModelFamily::Qwen2("W25_14b".to_string()));

        let qwen3_model = ModelFamily::qwen3("W3_14b");
        assert_eq!(qwen3_model, ModelFamily::Qwen3("W3_14b".to_string()));

        let default_qwen2 = ModelFamily::default_qwen2();
        assert_eq!(default_qwen2, ModelFamily::Qwen2("W25_7b".to_string()));

        let default_qwen3 = ModelFamily::default_qwen3();
        assert_eq!(default_qwen3, ModelFamily::Qwen3("W3_8b".to_string()));
    }

    #[test]
    fn test_model_family_get_config() -> Result<()> {

        // 测试通过 ModelFamily 获取配置
        let model = ModelFamily::qwen2("W25_14b");
        let config = model.get_config()?;

        assert_eq!(config.model_repo, "Qwen/Qwen2.5-14B-Instruct-GGUF");
        assert_eq!(config.model_file, "qwen2.5-14b-instruct-q4_0");
        assert_eq!(config.tokenizer_repo, "Qwen/Qwen2.5-14B-Instruct");

        Ok(())
    }

    #[test]
    fn test_model_family_convenience_methods() {
        // 测试 ModelFamily 的创建和基本功能
        let model = ModelFamily::qwen2("W25_14b");

        // 测试模型创建
        assert_eq!(model, ModelFamily::Qwen2("W25_14b".to_string()));

        // 测试默认模型创建
        let default_qwen2 = ModelFamily::default_qwen2();
        assert_eq!(default_qwen2, ModelFamily::Qwen2("W25_7b".to_string()));

        let default_qwen3 = ModelFamily::default_qwen3();
        assert_eq!(default_qwen3, ModelFamily::Qwen3("W3_8b".to_string()));
    }

    #[test]
    fn test_fallback_to_default() -> Result<()> {
        // 测试当请求不存在的模型时，回退到默认模型
        let model = ModelFamily::qwen2("NonExistent");
        let config = model.get_config()?;

        // 应该回退到默认的 W25_7b
        assert_eq!(config.model_repo, "Qwen/Qwen2.5-7B-Instruct-GGUF");
        assert_eq!(config.model_file, "qwen2.5-7b-instruct-q4_0");

        Ok(())
    }

    #[test]
    fn test_usage_examples() -> Result<()> {
        // 使用示例 1: 创建特定模型配置
        let model_config = ModelFamily::qwen2("W25_14b");
        let config = model_config.get_config()?;
        println!("Model: {} -> {}", config.model_repo, config.model_file);

        // 使用示例 2: 使用默认模型
        let default_model = ModelFamily::default_qwen3();
        let default_config = default_model.get_config()?;
        println!(
            "Default Qwen3: {} -> {}",
            default_config.model_repo, default_config.model_file
        );

        // 使用示例 3: 动态选择模型
        let model_name = "W3_14b";
        let dynamic_model = ModelFamily::qwen3(model_name);
        let dynamic_config = dynamic_model.get_config()?;
        println!(
            "Dynamic: {} -> {}",
            dynamic_config.model_repo, dynamic_config.model_file
        );

        // 使用示例 4: 使用便利方法
        let model = ModelFamily::qwen2("W25_7b");
        println!("便利方法:");
        println!("  仓库: {}", model.model_repo()?);
        println!("  文件: {}", model.model_file()?);
        println!("  分词器: {}", model.tokenizer_repo()?);

        Ok(())
    }
}