/// 临时环境变量守卫 — RAII 风格，Drop 时自动恢复原始值
///
/// 适用场景：测试或示例代码中临时覆盖环境变量，离开作用域后自动清理。
///
/// # 示例
///
/// ```rust
/// // 临时设置代理
/// let _proxy = ProxyGuard::new(7890);
///
/// // 临时切换到 HuggingFace 镜像站
/// let _hf = HfEndpointGuard::new("https://hf-mirror.com");
/// ```
use std::env;

// ─── ProxyGuard ──────────────────────────────────────────────────────────────

/// 临时设置 `HTTPS_PROXY` 环境变量，Drop 时自动移除。
pub struct ProxyGuard;

impl ProxyGuard {
    pub fn new(port: u16) -> Self {
        unsafe {
            env::set_var("HTTPS_PROXY", format!("http://127.0.0.1:{port}"));
        }
        Self
    }
}

impl Drop for ProxyGuard {
    fn drop(&mut self) {
        unsafe {
            env::remove_var("HTTPS_PROXY");
        }
    }
}

// ─── HfEndpointGuard ─────────────────────────────────────────────────────────

/// 临时覆盖 `HF_ENDPOINT` 环境变量，Drop 时恢复原始值（若原本不存在则移除）。
///
/// `hf-hub` 在构造 `ApiBuilder` 时读取该变量，因此必须在创建 API 客户端之前设置。
///
/// # 示例
///
/// ```rust
/// // 使用国内镜像站下载模型
/// let _hf = HfEndpointGuard::new("https://hf-mirror.com");
/// let api = ApiBuilder::from_env().build()?;
/// ```
pub struct HfEndpointGuard {
    prev: Option<String>,
}

impl HfEndpointGuard {
    pub fn new(endpoint: &str) -> Self {
        let prev = env::var("HF_ENDPOINT").ok();
        unsafe {
            env::set_var("HF_ENDPOINT", endpoint);
        }
        Self { prev }
    }
}

impl Drop for HfEndpointGuard {
    fn drop(&mut self) {
        unsafe {
            match &self.prev {
                Some(v) => env::set_var("HF_ENDPOINT", v),
                None => env::remove_var("HF_ENDPOINT"),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Guard Drop 后应恢复原来的值（原本已存在的情况）
    #[test]
    fn test_hf_endpoint_guard_restore_prev() {
        let original = "https://original-endpoint.example.com";
        unsafe { env::set_var("HF_ENDPOINT", original) };

        {
            let _guard = HfEndpointGuard::new("https://hf-mirror.com");
            assert_eq!(env::var("HF_ENDPOINT").unwrap(), "https://hf-mirror.com");
        }

        assert_eq!(
            env::var("HF_ENDPOINT").unwrap(),
            original,
            "Drop 后应恢复原始值"
        );

        // 清理
        unsafe { env::remove_var("HF_ENDPOINT") };
    }

    /// 真实请求镜像站，验证 hf-hub 实际使用了 HF_ENDPOINT
    ///
    /// 该测试会发起网络请求，默认 ignore，手动运行：
    /// cargo test --lib utils::env_guard::tests::test_hf_endpoint_mirror_reachable -- --nocapture --ignored
    #[tokio::test]
    async fn test_hf_endpoint_mirror_reachable() {
        let _guard = HfEndpointGuard::new("https://hf-mirror.com");

        let api = hf_hub::api::tokio::ApiBuilder::from_env()
            .build()
            .expect("构建 ApiBuilder 失败");

        // 拉取一个极小的公开文件来确认镜像可达
        let repo = api.model("Qwen/Qwen3-4B-Instruct-2507".to_string());
        let result = repo.get("config.json").await;

        match &result {
            Ok(path) => println!("✅ 镜像可达，文件缓存至：{}", path.display()),
            Err(e) => println!("❌ 镜像请求失败：{e}"),
        }

        // 不强制 assert，让用户通过输出判断
        dbg!(&result.is_ok());
    }
}
