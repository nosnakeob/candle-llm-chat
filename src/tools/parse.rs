//! Tool Call 解析器
//!
//! 负责从模型输出中提取 `<tool_call>...</tool_call>` 块。
//!
//! 模型输出示例：
//! ```text
//! 我来帮你读取这个文件。
//! <tool_call>{"name": "read_file", "arguments": {"path": "/path/to/file.txt"}}</tool_call>
//! 好的，让我把文件内容展示给你看。
//! ```
//!
//! 解析器会：
//! 1. 用正则提取 `<tool_call>` 标签内的 JSON
//! 2. 解析 JSON 中的 `name` 和 `arguments` 字段
//! 3. 支持在同一个响应中包含多个工具调用

use anyhow::{bail, Result};
use regex::Regex;
use serde::Deserialize;

/// ToolCallParser 从模型输出中提取 tool_call 块
///
/// 这是一个零大小类型，所有方法均为关联函数。
pub struct ToolCallParser;

/// 从模型原始输出中解析出的 JSON 结构
#[derive(Debug, Deserialize)]
struct RawToolCall {
    name: String,
    arguments: serde_json::Value,
}

impl ToolCallParser {
    /// 从模型输出文本中提取所有 tool_call 块
    ///
    /// 支持两种格式（按优先级顺序尝试）：
    /// 1. XML 包裹格式（推荐）：`<tool_call>{"name": "...", "arguments": {...}}</tool_call>`
    /// 2. 裸 JSON 格式（兼容）：`{"name": "...", "arguments": {...}}`（整个输出就是一个 JSON 对象）
    ///
    /// 返回按出现顺序排列的所有工具调用。
    pub fn parse(output: &str) -> Result<Vec<crate::tools::ParsedToolCall>> {
        // 优先尝试 <tool_call>...</tool_call> 包裹格式
        let re = Regex::new(r"<tool_call>\s*(\{.*?\})\s*</tool_call>")
            .map_err(|e| anyhow::anyhow!("regex error: {}", e))?;

        let mut results = Vec::new();

        for cap in re.captures_iter(output) {
            let json_str = &cap[1];
            if let Some(call) = Self::parse_json_call(json_str)? {
                results.push(call);
            }
        }

        // 若 XML 格式未匹配到，尝试将整个输出作为裸 JSON 解析
        // 兼容模型直接输出 {"name": "...", "arguments": {...}} 的情况
        if results.is_empty() {
            let trimmed = output.trim();
            if trimmed.starts_with('{') && trimmed.ends_with('}') {
                if let Some(call) = Self::parse_json_call(trimmed)? {
                    results.push(call);
                }
            }
        }

        Ok(results)
    }

    /// 解析单个 JSON 字符串为 ParsedToolCall，若字段不符合工具调用格式则返回 None
    fn parse_json_call(json_str: &str) -> Result<Option<crate::tools::ParsedToolCall>> {
        let raw: RawToolCall = match serde_json::from_str(json_str) {
            Ok(v) => v,
            Err(_) => return Ok(None),
        };

        // arguments 必须是对象
        let arguments = match raw.arguments {
            serde_json::Value::Object(obj) => serde_json::Value::Object(obj),
            other => bail!("arguments 必须是 JSON 对象，实际类型: {:?}", other),
        };

        Ok(Some(crate::tools::ParsedToolCall {
            name: raw.name,
            arguments,
        }))
    }

    /// 从模型输出中移除所有 tool_call 块（XML 格式和裸 JSON 格式）
    ///
    /// 用于在显示给用户之前清理模型的原始输出。
    pub fn strip(output: &str) -> String {
        // 移除 XML 包裹格式
        let re = Regex::new(r"<tool_call>\s*\{.*?\}\s*</tool_call>\n?")
            .expect("valid regex");
        let cleaned = re.replace_all(output, "");

        // 若整个输出是裸 JSON 工具调用，清空
        let trimmed = cleaned.trim();
        let cleaned = if trimmed.starts_with('{') && trimmed.ends_with('}') {
            if serde_json::from_str::<RawToolCall>(trimmed).is_ok() {
                std::borrow::Cow::Borrowed("")
            } else {
                cleaned
            }
        } else {
            cleaned
        };

        let re_empty_lines = Regex::new(r"\n{3,}").expect("valid regex");
        re_empty_lines.replace_all(&cleaned, "\n\n").to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_single_call() {
        let output = "我来读取这个文件。\n<tool_call>{\"name\": \"read_file\", \"arguments\": {\"path\": \"/a.txt\"}}</tool_call>\n";
        let calls = ToolCallParser::parse(output).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "read_file");
        assert_eq!(calls[0].arguments["path"], "/a.txt");
    }

    #[test]
    fn test_parse_multiple_calls() {
        let output = "<tool_call>{\"name\": \"read_file\", \"arguments\": {\"path\": \"/a.txt\"}}</tool_call>\n<tool_call>{\"name\": \"read_file\", \"arguments\": {\"path\": \"/b.txt\"}}</tool_call>";
        let calls = ToolCallParser::parse(output).unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[1].arguments["path"], "/b.txt");
    }

    #[test]
    fn test_parse_with_spaces() {
        let output = "<tool_call>  {\"name\": \"read_file\", \"arguments\": {\"path\": \"/a.txt\"}}  </tool_call>";
        let calls = ToolCallParser::parse(output).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "read_file");
    }

    #[test]
    fn test_parse_invalid_json() {
        // 非 JSON 格式的内容不匹配正则 \{.*?\}，返回空列表而非错误
        let output = "<tool_call>not json</tool_call>";
        let result = ToolCallParser::parse(output).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_parse_no_calls() {
        let output = "普通回复，不调用任何工具。";
        let calls = ToolCallParser::parse(output).unwrap();
        assert!(calls.is_empty());
    }

    /// 兼容模型直接输出裸 JSON 的情况（不带 <tool_call> 标签）
    #[test]
    fn test_parse_bare_json() {
        let output = r#"{"name": "read_file", "arguments": {"path": "/a.txt"}}"#;
        let calls = ToolCallParser::parse(output).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "read_file");
        assert_eq!(calls[0].arguments["path"], "/a.txt");
    }

    /// 裸 JSON 格式的 strip 应清空输出
    #[test]
    fn test_strip_bare_json() {
        let output = r#"{"name": "read_file", "arguments": {"path": "/a.txt"}}"#;
        let cleaned = ToolCallParser::strip(output);
        assert!(cleaned.trim().is_empty());
    }

    #[test]
    fn test_strip_tool_calls() {
        let output = "我来读文件。\n<tool_call>{\"name\": \"read_file\", \"arguments\": {\"path\": \"/a.txt\"}}</tool_call>\n这是文件内容。";
        let cleaned = ToolCallParser::strip(output);
        assert!(!cleaned.contains("tool_call"));
        assert!(cleaned.contains("我来读文件"));
        assert!(cleaned.contains("这是文件内容"));
    }
}
