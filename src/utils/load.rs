use anyhow::{Error, Result};
use candle::quantized::gguf_file::Content;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use config;
use futures_util::future::try_join_all;
use hf_hub::api::tokio::ApiBuilder;
use hf_hub::{Cache, Repo, api::tokio::Api};
use regex::Regex;
use std::{fs::File, path::PathBuf, process::Command};
use tokenizers::Tokenizer;

/// 从指定仓库下载GGUF模型文件,支持下载分片模型文件,会自动检测并合并分片
///
/// # 参数
/// * `repo` - 模型仓库名
/// * `filename` - 模型文件名(不带后缀)
pub async fn download_gguf(repo: &str, filename: &str) -> Result<PathBuf> {
    // 添加.gguf后缀
    let filename_with_ext = format!("{}.gguf", filename);

    if let Some(path) = Cache::default()
        .model(repo.to_string())
        .get(&filename_with_ext)
    {
        Ok(path)
    } else {
        let repo = Api::new()?.model(repo.to_string());

        // 模型可能分片, 收集前缀为 filename 的文件
        let split_filenames: Vec<_> = repo
            .info()
            .await?
            .siblings
            .into_iter()
            .map(|sibling| sibling.rfilename)
            .filter(|s| s.starts_with(filename))
            .collect();

        // 如果没有分片，直接下载完整文件
        if split_filenames.len() == 1 {
            return Ok(repo.get(&filename_with_ext).await?);
        }

        // 下载分片文件
        let split_paths = try_join_all(split_filenames.iter().map(|f| repo.get(f))).await?;

        let download_dir = split_paths[0].parent().unwrap();

        let merge_path = download_dir.join(format!("{filename}*"));

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
        let new_path = download_dir.join(&filename_with_ext);
        std::fs::rename(merged_path, &new_path)?;

        Ok(new_path)
    }
}

pub async fn load_tokenizer(repo: &str) -> Result<Tokenizer> {
    let config = config::Config::builder()
        .add_source(config::File::with_name("config.toml"))
        .build()?;
    let token = config.get_string("huggingface.token")?;

    let pth = ApiBuilder::new()
        .with_token(Some(token))
        .build()?
        .model(repo.to_string())
        .get("tokenizer.json")
        .await?;

    Tokenizer::from_file(pth).map_err(Error::msg)
}

mod tests {
    use super::*;
    use crate::model::registry::ModelRegistry;
    use crate::utils::{log_tensor_size, proxy::ProxyGuard};
    use candle_transformers::models::hiera;
    use serde_json::Value;
    use std::io::BufReader;

    #[tokio::test]
    async fn test_load_gguf() -> Result<()> {
        tracing_subscriber::fmt::init();

        let registry = ModelRegistry::load()?;
        let hub_info = registry.get("qwen3").unwrap();

        let model_path = download_gguf(&hub_info.model_repo, &hub_info.model_file).await?;

        let mut file = File::open(&model_path)?;

        // 构建模型
        let ct = Content::read(&mut file)?;

        let pth = Api::new()?
            .model(hub_info.tokenizer_repo.to_string())
            .get("tokenizer_config.json")
            .await?;
        let file = File::open(pth)?;
        let mut json: Value = serde_json::from_reader(BufReader::new(file))?;
        dbg!(&ct.metadata.keys());

        log_tensor_size(&ct);

        Ok(())
    }
}
