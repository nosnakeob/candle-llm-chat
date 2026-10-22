use anyhow::{Error, Result, anyhow};
use hf_hub::repository::{HFRepository, RepoTypeModel};
use regex::Regex;
use std::{fs::File, path::PathBuf, process::Command};
use tokenizers::Tokenizer;

use crate::utils::hf::{fetch, fetch_many, list_files_with_prefix, model_repo};

/// 从指定仓库下载 GGUF 模型文件，支持下载分片模型文件，会自动检测并合并分片
///
/// # 参数
/// * `repo` - 模型仓库名
/// * `filename` - 模型文件名(不带后缀)
pub async fn download_gguf(repo: &str, filename: &str) -> Result<PathBuf> {
    let repo = model_repo(repo)?;

    // 获取不带后缀的文件名前缀用于分片检测
    let filename_prefix = filename.strip_suffix(".gguf").unwrap_or(filename);

    // 模型可能分片, 收集前缀为 filename_prefix 的文件
    // 走 tree 列表接口（单次分页请求），不拉整个仓库 info
    let split_filenames = list_files_with_prefix(&repo, filename_prefix).await?;

    // 如果没有分片，直接下载完整文件
    if split_filenames.len() == 1 {
        return fetch(&repo, &split_filenames[0]).await;
    }

    // 下载分片文件
    let split_paths = fetch_many(&repo, &split_filenames).await?;

    let download_dir = split_paths[0]
        .parent()
        .ok_or_else(|| anyhow!("GGUF 分片文件缺少父目录"))?;

    let merge_path = download_dir.join(format!("{filename_prefix}*"));

    let output = Command::new("gguf-utils")
        .arg("merge")
        .arg(merge_path)
        .arg("-o")
        .arg(download_dir)
        .output()
        .map_err(|e| {
            anyhow!("调用 gguf-utils 合并分片失败（需先 cargo install gguf-utils）: {e}")
        })?;

    if !output.status.success() {
        anyhow::bail!(
            "gguf-utils 合并失败: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }

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

/// 加载 tokenizer.json（走 hf-hub 1.0 的缓存与鉴权，不再使用 tokenizers 的 http feature）
pub async fn load_tokenizer(repo: &str) -> Result<Tokenizer> {
    let path = fetch(&model_repo(repo)?, "tokenizer.json").await?;
    Tokenizer::from_file(path).map_err(|e| Error::msg(e.to_string()))
}

/// 下载 safetensors 权重文件；单文件缺失时回退到分片 index.json
pub async fn download_safetensors(
    repo: &HFRepository<RepoTypeModel>,
    filename: &str,
) -> Result<Vec<PathBuf>> {
    match fetch(repo, filename).await {
        Ok(single) => Ok(vec![single]),
        Err(_) => repo.get_safetensors().await,
    }
}

/// HFRepository 的扩展 trait，提供 safetensors 分片加载
pub trait ApiRepoExt {
    /// 从 HuggingFace Hub 加载 safetensors 模型文件
    ///
    /// 根据 model.safetensors.index.json 文件加载所有分片的 safetensors 文件
    fn get_safetensors(&self) -> impl std::future::Future<Output = Result<Vec<PathBuf>>> + Send;
}

impl ApiRepoExt for HFRepository<RepoTypeModel> {
    async fn get_safetensors(&self) -> Result<Vec<PathBuf>> {
        let json_file = "model.safetensors.index.json";
        let json_path = fetch(self, json_file).await?;
        let json_file_handle = File::open(json_path)?;
        let json: serde_json::Value = serde_json::from_reader(&json_file_handle)?;

        // 提取 weight_map
        let weight_map = match json.get("weight_map") {
            None => anyhow::bail!("no weight map in {json_file}"),
            Some(serde_json::Value::Object(map)) => map,
            Some(_) => anyhow::bail!("weight map in {json_file} is not a map"),
        };

        // 收集所有唯一的 safetensors 文件名
        let mut safetensors_files: Vec<String> = weight_map
            .values()
            .filter_map(|v| v.as_str().map(|s| s.to_string()))
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        safetensors_files.sort();

        // 并发下载所有文件
        fetch_many(self, &safetensors_files).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 验证 chat template JSON 解析逻辑（不需要网络）
    #[test]
    fn test_chat_template_parsing() -> Result<()> {
        let mock_config = r#"
        {
            "chat_template": "{{ bos_token }}{% for message in messages %}{% if message['role'] == 'user' %}{{ '<|im_start|>user\\n' + message['content'] + '<|im_end|>\\n<|im_start|>assistant\\n' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + '<|im_end|>\\n' }}{% endif %}{% endfor %}",
            "eos_token_id": 151643
        }
        "#;

        let json: serde_json::Value = serde_json::from_str(mock_config)?;
        let chat_template = json["chat_template"].as_str().unwrap();

        assert!(chat_template.contains("im_start"));
        assert!(chat_template.contains("user"));
        assert!(chat_template.contains("assistant"));

        Ok(())
    }

    /// 验证 GGUF 模型下载（需要网络和磁盘空间）
    #[tokio::test]
    #[ignore]
    async fn test_load_gguf() -> Result<()> {
        let registry = crate::model::registry::ModelRegistry::new()?;
        let hub_info = registry.get("qwen3.4b_q4")?;
        let model_path = download_gguf(&hub_info.model_repo, &hub_info.model_file).await?;
        assert!(model_path.exists());
        Ok(())
    }

    /// 验证 safetensors 分片文件加载（需要网络和磁盘空间）
    #[tokio::test]
    #[ignore]
    async fn test_hub_load_safetensors() -> Result<()> {
        let repo = model_repo("Qwen/Qwen3-8B")?;
        let paths = repo.get_safetensors().await?;
        assert!(!paths.is_empty());
        Ok(())
    }
}
