//! 网页抓取工具
//!
//! 提供访问指定 URL 并提取页面文本内容的能力。
//!
//! ## 处理流程
//! 1. 验证 URL 格式（必须是 http/https）
//! 2. 发送 GET 请求，设置超时和 User-Agent
//! 3. 提取响应 HTML 中的纯文本（去除标签、脚本、样式）
//! 4. 超过 `max_chars` 时截断并提示
//!
//! ## 使用限制
//! - 仅支持 http / https 协议
//! - 响应内容上限为 20000 字符，超过时截断
//! - 请求超时 15 秒

use anyhow::{bail, Context, Result};
use regex::Regex;

use super::{Tool, ToolDef};

/// 网页抓取工具
pub struct WebFetchTool {
    /// 提取文本的最大字符数
    max_chars: usize,
    /// 请求超时（秒）
    timeout_secs: u64,
}

impl WebFetchTool {
    pub fn new() -> Self {
        Self {
            max_chars: 20_000,
            timeout_secs: 15,
        }
    }

    pub fn with_max_chars(mut self, max_chars: usize) -> Self {
        self.max_chars = max_chars;
        self
    }

    pub fn with_timeout_secs(mut self, secs: u64) -> Self {
        self.timeout_secs = secs;
        self
    }

    fn validate_url(&self, url: &str) -> Result<String> {
        let url = url.trim();
        if url.is_empty() {
            bail!("URL 不能为空");
        }
        if !url.starts_with("http://") && !url.starts_with("https://") {
            bail!("仅支持 http / https 协议，收到: {}", url);
        }
        Ok(url.to_string())
    }

    fn fetch(&self, url: &str) -> Result<String> {
        let url = url.to_string();
        let timeout = self.timeout_secs;

        // reqwest::blocking 内部会创建 tokio runtime，在异步上下文中直接调用会 panic。
        // 通过 std::thread::spawn 在独立线程中执行，彻底脱离当前 tokio 上下文。
        let handle = std::thread::spawn(move || -> Result<String> {
            let client = reqwest::blocking::Client::builder()
                .timeout(std::time::Duration::from_secs(timeout))
                .user_agent("Mozilla/5.0 (compatible; candle-llm-chat/1.0)")
                .build()
                .context("构建 HTTP 客户端失败")?;

            let response = client
                .get(&url)
                .send()
                .context(format!("请求失败: {}", url))?;

            let status = response.status();
            if !status.is_success() {
                bail!("HTTP 请求返回错误状态码: {}", status);
            }

            response.text().context("读取响应内容失败")
        });

        handle
            .join()
            .map_err(|_| anyhow::anyhow!("HTTP 请求线程 panic"))?
    }
}

impl Default for WebFetchTool {
    fn default() -> Self {
        Self::new()
    }
}

impl Tool for WebFetchTool {
    fn name(&self) -> &str {
        "web_fetch"
    }

    fn definition(&self) -> ToolDef {
        ToolDef::new(
            "web_fetch",
            "访问指定 URL 并返回页面的纯文本内容。当用户需要查看网页内容、获取在线资料或访问链接时调用此工具。仅支持 http/https 协议。",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "要访问的网页 URL，必须以 http:// 或 https:// 开头"
                    }
                },
                "required": ["url"]
            }),
        )
    }

    fn execute(&self, arguments: &serde_json::Value) -> Result<String> {
        let url = arguments["url"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("url 参数必须是字符串"))?;

        let url = self.validate_url(url)?;
        let html = self.fetch(&url)?;
        let text = extract_text(&html, self.max_chars);

        Ok(text)
    }
}

/// 从 HTML 中提取纯文本
///
/// 处理步骤：
/// 1. 移除 `<script>` 和 `<style>` 块（含内容）
/// 2. 移除所有 HTML 标签
/// 3. 解码常见 HTML 实体
/// 4. 合并多余空白行
/// 5. 超长时截断
fn extract_text(html: &str, max_chars: usize) -> String {
    // 移除 <script>...</script>
    let re_script = Regex::new(r"(?si)<script[^>]*>.*?</script>").expect("valid regex");
    let text = re_script.replace_all(html, " ");

    // 移除 <style>...</style>
    let re_style = Regex::new(r"(?si)<style[^>]*>.*?</style>").expect("valid regex");
    let text = re_style.replace_all(&text, " ");

    // 移除所有 HTML 标签
    let re_tags = Regex::new(r"<[^>]+>").expect("valid regex");
    let text = re_tags.replace_all(&text, " ");

    // 解码常见 HTML 实体
    let text = text
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&nbsp;", " ")
        .replace("&apos;", "'");

    // 合并多余空白（多个空格/制表符 → 单个空格）
    let re_spaces = Regex::new(r"[ \t]+").expect("valid regex");
    let text = re_spaces.replace_all(&text, " ");

    // 合并多余空行（3 行以上 → 2 行）
    let re_newlines = Regex::new(r"\n{3,}").expect("valid regex");
    let text = re_newlines.replace_all(&text, "\n\n");

    let text = text.trim().to_string();

    // 截断
    if text.chars().count() > max_chars {
        let truncated: String = text.chars().take(max_chars).collect();
        // 在最后一个换行处截断，避免截断到行中间
        let cut = truncated
            .rfind('\n')
            .map(|p| p + 1)
            .unwrap_or(truncated.len());
        format!(
            "{}\n\n[内容过长，已截断前 {} 字符]",
            &truncated[..cut],
            max_chars
        )
    } else {
        text
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_url_ok() {
        let tool = WebFetchTool::new();
        assert!(tool.validate_url("https://example.com").is_ok());
        assert!(tool.validate_url("http://example.com/path?q=1").is_ok());
    }

    #[test]
    fn test_validate_url_err() {
        let tool = WebFetchTool::new();
        assert!(tool.validate_url("").is_err());
        assert!(tool.validate_url("ftp://example.com").is_err());
        assert!(tool.validate_url("example.com").is_err());
    }

    #[test]
    fn test_extract_text_removes_tags() {
        let html = "<html><body><h1>标题</h1><p>正文内容</p></body></html>";
        let text = extract_text(html, 10000);
        assert!(!text.contains('<'));
        assert!(text.contains("标题"));
        assert!(text.contains("正文内容"));
    }

    #[test]
    fn test_extract_text_removes_script() {
        let html = "<p>可见文本</p><script>var x = 1;</script><p>另一段</p>";
        let text = extract_text(html, 10000);
        assert!(!text.contains("var x"));
        assert!(text.contains("可见文本"));
    }

    #[test]
    fn test_extract_text_decodes_entities() {
        let html = "<p>a &amp; b &lt;c&gt; &quot;d&quot;</p>";
        let text = extract_text(html, 10000);
        assert!(text.contains("a & b <c> \"d\""));
    }

    #[test]
    fn test_extract_text_truncates() {
        let html = format!("<p>{}</p>", "中".repeat(500));
        let text = extract_text(&html, 100);
        assert!(text.contains("已截断"));
        assert!(text.chars().count() < 200);
    }

    #[test]
    fn test_execute_missing_url() {
        let tool = WebFetchTool::new();
        let result = tool.execute(&serde_json::json!({}));
        assert!(result.is_err());
    }

    #[test]
    fn test_execute_invalid_protocol() {
        let tool = WebFetchTool::new();
        let result = tool.execute(&serde_json::json!({"url": "ftp://example.com"}));
        assert!(result.is_err());
    }

    #[test]
    #[ignore] // 需要网络
    fn test_fetch_real_url() {
        let tool = WebFetchTool::new();
        let result = tool.execute(&serde_json::json!({"url": "https://example.com"}));
        dbg!(&result);
        assert!(result.is_ok());
        let text = result.unwrap();
        assert!(text.contains("Example"));
    }
}
