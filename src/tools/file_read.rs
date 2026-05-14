//! 文件读取工具
//!
//! 提供读取本地文件内容的能力，支持纯文本文件和 .eml 邮件文件。
//!
//! 模型通过 `read_file` 工具读取用户指定的文件，
//! 系统将文件内容作为结果返回，模型基于此内容回答用户问题。
//!
//! ## 使用限制
//! - 目前仅支持读取项目目录下的文件，防止越权访问
//! - 读取内容上限为 100KB，超过时截断并提示
//! - .eml 文件自动提取邮件头和正文，纯文本直接返回

use std::path::Path;

use anyhow::{bail, Context, Result};

use super::{Tool, ToolDef};

/// 文件读取工具
pub struct FileReadTool {
    /// 允许读取的根目录（不含末尾斜杠）
    /// 若为 None，则不限制路径
    root_dir: Option<String>,
    /// 最大读取字节数
    max_bytes: usize,
}

impl FileReadTool {
    pub fn new() -> Self {
        Self {
            root_dir: None,
            max_bytes: 100 * 1024, // 100KB
        }
    }

    /// 限制读取路径在指定目录下
    pub fn with_root_dir(mut self, root_dir: impl Into<String>) -> Self {
        self.root_dir = Some(root_dir.into());
        self
    }

    /// 限制最大读取字节数
    pub fn with_max_bytes(mut self, max_bytes: usize) -> Self {
        self.max_bytes = max_bytes;
        self
    }

    fn validate_path(&self, path: &str) -> Result<String> {
        // 防止路径遍历攻击
        let path = path.trim();

        if path.is_empty() {
            bail!("路径不能为空");
        }

        if let Some(ref root) = self.root_dir {
            let abs_root = Path::new(root).canonicalize()
                .context("根目录不存在")?;
            let abs_path = Path::new(path).canonicalize()
                .context(format!("路径不存在或无法访问: {}", path))?;

            // 确保路径在根目录内
            if !abs_path.starts_with(&abs_root) {
                bail!("禁止访问根目录之外的文件: {}", path);
            }
        }

        Ok(path.to_string())
    }

    fn do_read(&self, path: &Path) -> Result<String> {
        match path.extension().and_then(|e| e.to_str()) {
            Some("eml") => read_eml(path),
            _ => read_text(path, self.max_bytes),
        }
    }
}

impl Default for FileReadTool {
    fn default() -> Self {
        Self::new()
    }
}

impl Tool for FileReadTool {
    fn name(&self) -> &str {
        "read_file"
    }

    fn definition(&self) -> ToolDef {
        ToolDef::new(
            "read_file",
            "读取指定路径的本地文件内容。当用户需要查看、总结、分析文件内容时，必须调用此工具。",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "文件的完整路径，例如 /path/to/file.txt 或 C:\\path\\to\\file.txt"
                    }
                },
                "required": ["path"]
            }),
        )
    }

    fn execute(&self, arguments: &serde_json::Value) -> Result<String> {
        let path_str = arguments["path"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("path 参数必须是字符串"))?;

        let validated = self.validate_path(path_str)?;
        let path = Path::new(&validated);

        self.do_read(path)
    }
}

/// 读取纯文本文件，超长时截断
fn read_text(path: &Path, max_bytes: usize) -> Result<String> {
    let bytes = std::fs::read(path)?;
    let len = bytes.len();

    let content = if len > max_bytes {
        let truncated = &bytes[..max_bytes];
        // 尝试找最后一个完整的行，避免截断行首
        let end = truncated
            .iter()
            .rposition(|&b| b == b'\n')
            .map(|p| p + 1)
            .unwrap_or(max_bytes);

        let s = String::from_utf8_lossy(&truncated[..end]);
        format!(
            "{}\n\n[文件过长，已截断前 {} 字节，共 {} 字节]",
            s, max_bytes, len
        )
    } else {
        String::from_utf8(bytes).context("文件不是有效的 UTF-8 文本")?
    };

    Ok(content)
}

/// 解析 .eml 文件，提取邮件头和正文
fn read_eml(path: &Path) -> Result<String> {
    let raw = std::fs::read_to_string(path)?;
    let mut result = String::new();
    let mut in_body = false;
    let mut body_lines: Vec<&str> = Vec::new();

    for line in raw.lines() {
        if !in_body {
            if line.is_empty() {
                in_body = true;
                continue;
            }
            if let Some(v) = line.strip_prefix("Subject:") {
                result.push_str(&format!("主题: {}\n", v.trim()));
            } else if let Some(v) = line.strip_prefix("From:") {
                result.push_str(&format!("发件人: {}\n", v.trim()));
            } else if let Some(v) = line.strip_prefix("To:") {
                result.push_str(&format!("收件人: {}\n", v.trim()));
            } else if let Some(v) = line.strip_prefix("Date:") {
                result.push_str(&format!("日期: {}\n", v.trim()));
            }
        } else {
            body_lines.push(line);
        }
    }

    result.push_str("\n正文:\n");
    result.push_str(&body_lines.join("\n"));

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_file(name: &str, content: &str) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(name);
        std::fs::write(&path, content).unwrap();
        path
    }

    #[test]
    fn test_read_text() {
        let path = tmp_file("tool_test_read_text.txt", "Hello, world!\n这是一段测试文本。");
        let content = read_text(&path, 1024).unwrap();
        assert!(content.contains("Hello"));
        assert!(content.contains("测试文本"));
    }

    #[test]
    fn test_read_truncate() {
        let long_content = "x".repeat(300);
        let path = tmp_file("tool_test_truncate.txt", &long_content);
        let content = read_text(&path, 100).unwrap();
        assert!(content.contains("已截断"));
        assert!(content.len() < 200);
    }

    #[test]
    fn test_path_traversal_blocked() {
        let tool = FileReadTool::new().with_root_dir("/safe/dir");
        let result = tool.validate_path("/safe/dir/../../../etc/passwd");
        assert!(result.is_err());
    }

    #[test]
    fn test_path_within_root() {
        // 使用实际存在的临时目录和文件，canonicalize() 要求路径必须存在
        let tmp = std::env::temp_dir();
        let file = tmp_file("tool_test_within_root.txt", "test");
        let tool = FileReadTool::new().with_root_dir(tmp.to_string_lossy().as_ref());
        let validated = tool.validate_path(&file.to_string_lossy());
        assert!(validated.is_ok());
    }

    #[test]
    fn test_read_eml() {
        let path = std::env::temp_dir().join("tool_test.eml");
        std::fs::write(
            &path,
            "From: sender@example.com\nTo: receiver@example.com\nSubject: 测试邮件\nDate: Mon, 1 Jan 2025 10:00:00 +0800\n\n这是邮件正文内容。",
        )
        .unwrap();

        let content = read_eml(&path).unwrap();
        assert!(content.contains("测试邮件"));
        assert!(content.contains("邮件正文内容"));
    }

    #[test]
    fn test_tool_execute() {
        let tool = FileReadTool::new();
        let path = tmp_file("tool_test_execute.txt", "文件内容ABC");

        let args = serde_json::json!({ "path": path.to_string_lossy() });
        let result = tool.execute(&args).unwrap();
        assert!(result.contains("文件内容ABC"));
    }
}
