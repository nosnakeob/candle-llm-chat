use crate::utils::format_size;
use anyhow::{Error, Result};
use candle::quantized::gguf_file::Content;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use config;
use futures_util::future::try_join_all;
use hf_hub::api::tokio::ApiBuilder;
use hf_hub::{Cache, Repo, api::tokio::Api};
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

        let merged_path = download_dir.join(&filename_with_ext);

        // 合并分片
        let output = Command::new(which::which("llama-gguf-split")?)
            .arg("--merge")
            .arg(split_paths[0].to_str().unwrap())
            .arg(merged_path.to_str().unwrap())
            .output()?;

        if !output.status.success() {
            bail!(
                "llama-gguf-split failed: {}",
                String::from_utf8_lossy(&output.stderr)
            )
        }

        Ok(merged_path)
    }
}

pub async fn load_gguf(repo: &str, filename: &str) -> Result<(File, Content)> {
    let model_path = download_gguf(repo, filename).await?;

    let mut file = File::open(&model_path)?;
    let start = std::time::Instant::now();

    // 构建模型
    let ct = Content::read(&mut file).map_err(|e| e.with_path(&model_path))?;
    let mut total_size_in_bytes = 0;
    for (_, tensor) in ct.tensor_infos.iter() {
        let elem_count = tensor.shape.elem_count();
        total_size_in_bytes +=
            elem_count * tensor.ggml_dtype.type_size() / tensor.ggml_dtype.block_size();
    }
    let formatted_size = format_size(total_size_in_bytes);
    info!(
        "loaded {:?} tensors ({}) in {:.2}s",
        ct.tensor_infos.len(),
        &formatted_size,
        start.elapsed().as_secs_f32(),
    );

    Ok((file, ct))
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
