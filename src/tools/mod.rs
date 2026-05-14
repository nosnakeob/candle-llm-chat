//! 工具调用（Tool Calling）模块
//!
//! 通过为 Agent 注册工具，让大模型能够主动调用外部函数来完成任务。
//!
//! # 工具调用流程
//! 1. Agent 接收用户消息
//! 2. 模型在输出中嵌入 `<tool_call>{"name": "...", "arguments": {...}}</tool_call>`
//! 3. Agent 解析出 ToolCall，查找对应 Tool 并执行
//! 4. 将执行结果注入到对话上下文
//! 5. 循环直到模型不再调用工具，最终输出回答
//!
//! # 格式约定
//! 使用 XML 标签包裹 JSON，保证解析可靠且不容易被模型截断：
//! ```text
//! <tool_call>{"name": "read_file", "arguments": {"path": "/path/to/file.txt"}}</tool_call>
//! ```
//!
//! 该格式在 system prompt 中告知模型，模型（特别是 Qwen3-instruct）对该格式有良好的指令遵循能力。

mod file_read;
mod registry;
mod parse;
mod web_fetch;

pub use file_read::FileReadTool;
pub use parse::ToolCallParser;
pub use registry::ToolRegistry;
pub use web_fetch::WebFetchTool;

use anyhow::Result;
use serde::{Deserialize, Serialize};

/// 工具元信息（暴露给模型的描述）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDef {
    /// 工具唯一标识名，模型通过此名称选择调用
    pub name: String,
    /// 工具用途描述，供模型理解何时该调用
    pub description: String,
    /// JSON Schema 格式的参数定义
    pub parameters: serde_json::Value,
}

impl ToolDef {
    pub fn new(name: impl Into<String>, description: impl Into<String>, parameters: serde_json::Value) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
        }
    }
}

/// 工具接口，所有可用工具都要实现此 trait
pub trait Tool: Send + Sync {
    /// 工具名称，必须与 ToolDef.name 一致
    fn name(&self) -> &str;

    /// 工具元信息（描述 + 参数 schema）
    fn definition(&self) -> ToolDef;

    /// 执行工具，返回执行结果的文本描述
    fn execute(&self, arguments: &serde_json::Value) -> Result<String>;
}

/// 单次工具调用的解析结果
#[derive(Debug)]
pub struct ParsedToolCall {
    /// 被调用的工具名称
    pub name: String,
    /// 解析后的参数（JSON 对象）
    pub arguments: serde_json::Value,
}

impl ParsedToolCall {
    /// 将解析结果转换为工具可用的 JSON
    pub fn arguments(&self) -> &serde_json::Value {
        &self.arguments
    }
}
