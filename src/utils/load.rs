use anyhow::{Error, Result};
use candle::quantized::gguf_file::Content;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use futures_util::future::try_join_all;
use hf_hub::api::tokio::ApiBuilder;
use hf_hub::{Cache, Repo, api::tokio::Api};
use regex::Regex;
use std::{fs::File, path::PathBuf, process::Command};
use tokenizers::{FromPretrainedParameters, Tokenizer};

/// 从指定仓库下载GGUF模型文件,支持下载分片模型文件,会自动检测并合并分片
///
/// # 参数
/// * `repo` - 模型仓库名
/// * `filename` - 模型文件名(不带后缀)
pub async fn download_gguf(repo: &str, filename: &str) -> Result<PathBuf> {
    if let Some(path) = Cache::default().model(repo.to_string()).get(filename) {
        Ok(path)
    } else {
        let repo = ApiBuilder::from_env().build()?.model(repo.to_string());

        // 获取不带后缀的文件名前缀用于分片检测
        let filename_prefix = filename.strip_suffix(".gguf").unwrap_or(filename);

        // 模型可能分片, 收集前缀为 filename_prefix 的文件
        let split_filenames: Vec<_> = repo
            .info()
            .await?
            .siblings
            .into_iter()
            .map(|sibling| sibling.rfilename)
            .filter(|s| s.starts_with(filename_prefix))
            .collect();

        // 如果没有分片，直接下载完整文件
        if split_filenames.len() == 1 {
            return Ok(repo.get(filename).await?);
        }

        // 下载分片文件
        let split_paths = try_join_all(split_filenames.iter().map(|f| repo.get(f))).await?;

        let download_dir = split_paths[0].parent().unwrap();

        let merge_path = download_dir.join(format!("{filename_prefix}*"));

        let output = Command::new("gguf-utils")
            .arg("merge")
            .arg(merge_path)
            .arg("-o")
            .arg(download_dir)
            .output()?;

        let stdout = String::from_utf8(output.stdout)?;

        let re = Regex::new(r"\|\s*([^\|]+\.gguf)\s*\|")?;

        let merged_path = re
            .captures(&stdout)
            .and_then(|cap| cap.get(1))
            .map(|m| m.as_str().trim())
            .ok_or_else(|| anyhow!("Failed to extract file path from output"))?;

        // 重命名文件
        let new_path = download_dir.join(filename);
        std::fs::rename(merged_path, &new_path)?;

        Ok(new_path)
    }
}

pub fn load_tokenizer(repo: &str) -> Result<Tokenizer> {
    let mut params = FromPretrainedParameters::default();
    params.token = std::env::var("HF_TOKEN").ok();

    Tokenizer::from_pretrained(repo, Some(params)).map_err(Error::msg)
}

/// ApiRepo 的扩展 trait，提供 safetensors 加载功能
pub trait ApiRepoExt {
    /// 从 HuggingFace Hub 加载 safetensors 模型文件
    ///
    /// 根据 model.safetensors.index.json 文件加载所有分片的 safetensors 文件
    fn get_safetensors(&self) -> impl std::future::Future<Output = Result<Vec<PathBuf>>> + Send;
}

impl ApiRepoExt for hf_hub::api::tokio::ApiRepo {
    async fn get_safetensors(&self) -> Result<Vec<PathBuf>> {
        let json_file = "model.safetensors.index.json";
        // 自行下载 index.json 文件 
        // todo Header content-range is missing
        let json_path = self.get(json_file).await?;
        let json_file_handle = std::fs::File::open(json_path)?;
        let json: serde_json::Value = serde_json::from_reader(&json_file_handle)?;

        // 提取 weight_map
        let weight_map = match json.get("weight_map") {
            None => anyhow::bail!("no weight map in {json_file}"),
            Some(serde_json::Value::Object(map)) => map,
            Some(_) => anyhow::bail!("weight map in {json_file} is not a map"),
        };

        // 收集所有唯一的 safetensors 文件名
        let safetensors_files: Vec<String> = weight_map
            .values()
            .filter_map(|v| v.as_str().map(|s| s.to_string()))
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        // 并发下载所有文件
        let download_futures: Vec<_> = safetensors_files
            .iter()
            .map(|filename| self.get(filename))
            .collect();

        let paths = try_join_all(download_futures).await?;

        Ok(paths)
    }
}

mod tests {
    use super::*;
    use crate::model::registry::ModelRegistry;
    use crate::utils::{log_tensor_size, proxy::ProxyGuard};
    use candle_transformers::models::flux::model;
    use candle_transformers::models::hiera;
    use serde_json::Value;
    use std::io::BufReader;

    #[tokio::test]
    async fn test_load_gguf() -> Result<()> {
        tracing_subscriber::fmt::init();

        let registry = ModelRegistry::new()?;

        let model_id = "qwen3.4b_q4";

        let hub_info = registry.get(model_id).unwrap();

        let model_path = download_gguf(&hub_info.model_repo, &hub_info.model_file).await?;

        let mut file = File::open(&model_path)?;

        let api = ApiBuilder::from_env().build()?;

        // 构建模型
        let ct = Content::read(&mut file)?;

        let pth = api
            .model(hub_info.tokenizer_repo.clone())
            .get("tokenizer_config.json")
            .await?;

        let file = File::open(pth)?;
        let mut json: Value = serde_json::from_reader(BufReader::new(file))?;
        dbg!(&ct.metadata.keys());

        log_tensor_size(&ct);

        Ok(())
    }

    #[tokio::test]
    async fn test_hub_load_safetensors() -> Result<()> {
        // 测试加载分片的 safetensors 模型
        let api = ApiBuilder::from_env().build()?;
        let repo = api.model("Qwen/Qwen3-8B".to_string());

        let paths = repo.get_safetensors().await?;

        println!("加载了 {} 个 safetensors 文件:", paths.len());
        for path in &paths {
            println!("  - {:?}", path.file_name());
        }

        assert!(!paths.is_empty());
        Ok(())
    }

    #[tokio::test]
    async fn test_get_chat_template() -> Result<()> {
        let api = ApiBuilder::from_env().build()?;
        let repo = api.model("Qwen/Qwen3-4B-Instruct-2507".to_string());

        let info: Value = repo
            .info_request()
            .query(&[("chat_template", "default")])
            .send()
            .await?
            .json()
            .await?;
        dbg!(info);

        Ok(())
    }

    #[test]
    fn test_chat_template_parsing() -> Result<()> {
        // 测试 chat template 解析逻辑，不需要网络请求
        use serde_json::Value;

        let mock_config = r#"
        {
            "chat_template": "{{ bos_token }}{% for message in messages %}{% if message['role'] == 'user' %}{{ '<|im_start|>user\\n' + message['content'] + '<|im_end|>\\n<|im_start|>assistant\\n' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + '<|im_end|>\\n' }}{% endif %}{% endfor %}",
            "eos_token_id": 151643
        }
        "#;

        let json: Value = serde_json::from_str(mock_config)?;
        let chat_template = json["chat_template"].as_str().unwrap();

        assert!(chat_template.contains("im_start"));
        assert!(chat_template.contains("user"));
        assert!(chat_template.contains("assistant"));

        println!("✅ Chat template 解析测试通过");
        println!("Template: {}", &chat_template[..50]); // 显示前50个字符

        Ok(())
    }
}
