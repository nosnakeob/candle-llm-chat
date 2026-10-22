//! hf-hub 1.0 访问层
//!
//! hf-hub 1.0 删掉了旧的 `Api` / `ApiRepo`，改成 `HFClient` + `HFRepository<T>`：
//! - 构造不需要 async，`HFClient::new()` 从环境变量读 HF_TOKEN / HF_ENDPOINT；
//! - 客户端内部持有一个 reqwest::Client（连接池），clone 很便宜 —— 全进程复用一个即可，
//!   不用每处 `ApiBuilder::from_env().build()` 各建一份；
//! - 文件名用 `download(...)`，返回值就是缓存目录里的 PathBuf（老 `get` 的替代）。

use std::sync::OnceLock;

use anyhow::{Context, Result};
use futures_util::TryStreamExt;
use hf_hub::repository::{HFRepository, RepoTreeEntry, RepoTypeModel};
use hf_hub::{HFClient, split_id};

static CLIENT: OnceLock<HFClient> = OnceLock::new();

/// 全局共享的 HF 客户端
///
/// 第一次调用时构建。注意：HF_ENDPOINT / HF_TOKEN 在首次构建时读取并固定，
/// 之后修改环境变量不会影响已构建的客户端。
pub fn client() -> Result<HFClient> {
    if let Some(client) = CLIENT.get() {
        return Ok(client.clone());
    }
    let client = build_client()?;
    // 并发首调时可能被别的线程抢先写入，取其值即可（两者配置一致）
    Ok(CLIENT.get_or_init(|| client.clone()).clone())
}

fn build_client() -> Result<HFClient> {
    // 镜像站（hf-mirror）不认 HF_TOKEN：带上 token 会直接 401，
    // 所以指向非官方 endpoint 时只保留匿名客户端，让缓存里的匿名文件继续可用。
    if let Ok(endpoint) = std::env::var("HF_ENDPOINT") {
        if !endpoint.is_empty() && !endpoint.contains("huggingface.co") {
            return HFClient::builder()
                .endpoint(endpoint)
                .token("")
                .build()
                .context("构建镜像站 HFClient 失败");
        }
    }
    HFClient::new().context("构建 HFClient 失败")
}

/// 取模型仓库句柄，`repo_id` 形如 `"Qwen/Qwen3-8B"`
pub fn model_repo(repo_id: &str) -> Result<HFRepository<RepoTypeModel>> {
    let (owner, name) = split_id(repo_id);
    Ok(client()?.model(owner, name))
}

/// 下载单个文件，返回本地缓存路径
///
/// hf-hub 1.0 的下载接口是 bon builder 风格，必须 `.send()` 才会真正发请求。
pub async fn fetch(
    repo: &HFRepository<RepoTypeModel>,
    filename: &str,
) -> Result<std::path::PathBuf> {
    Ok(repo.download_file().filename(filename).send().await?)
}

/// 并发下载多个文件，保持入参顺序
pub async fn fetch_many(
    repo: &HFRepository<RepoTypeModel>,
    filenames: &[String],
) -> Result<Vec<std::path::PathBuf>> {
    let futures = filenames
        .iter()
        .map(|f| repo.download_file().filename(f).send());
    Ok(futures_util::future::try_join_all(futures).await?)
}

/// 列出仓库中所有以 `prefix` 开头的文件路径（GGUF 分片探测）
pub async fn list_files_with_prefix(
    repo: &HFRepository<RepoTypeModel>,
    prefix: &str,
) -> Result<Vec<String>> {
    let entries = repo.list_tree().recursive(true).send()?;
    let mut entries = std::pin::pin!(entries);
    let mut paths = Vec::new();
    while let Some(entry) = entries.try_next().await? {
        if let RepoTreeEntry::File { path, .. } = entry {
            if path.starts_with(prefix) {
                paths.push(path);
            }
        }
    }
    Ok(paths)
}
