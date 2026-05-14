//! 工具注册表
//!
//! 负责管理所有可用工具，提供工具查找、执行、JSON Schema 导出功能。

use anyhow::{bail, Result};
use std::collections::HashMap;

use super::{Tool, ToolDef};

/// 工具注册表
pub struct ToolRegistry {
    /// 工具集合，key 为工具名称
    tools: HashMap<String, Box<dyn Tool>>,
}

impl ToolRegistry {
    /// 创建一个空的工具注册表
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    /// 注册一个工具
    ///
    /// 同一名称的工具只能注册一次，重复注册会报错。
    pub fn register<T: Tool + 'static>(&mut self, tool: T) -> Result<()> {
        let name = tool.name().to_string();
        if self.tools.contains_key(&name) {
            bail!("工具 '{}' 已存在，不能重复注册", name);
        }
        self.tools.insert(name, Box::new(tool));
        Ok(())
    }

    /// 执行指定名称的工具
    pub fn execute(&self, name: &str, arguments: &serde_json::Value) -> Result<String> {
        let tool = self
            .tools
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("未找到工具: {}", name))?;

        tool.execute(arguments)
    }

    /// 检查是否包含指定名称的工具
    pub fn has(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }

    /// 获取所有工具的名称列表
    pub fn tool_names(&self) -> Vec<&str> {
        self.tools.keys().map(|s| s.as_str()).collect()
    }

    /// 遍历所有注册的工具，生成 JSON Schema 格式的 tools 数组
    ///
    /// 生成的 JSON 符合 OpenAI tools API 格式，可直接注入到 system prompt。
    pub fn to_openai_schema(&self) -> serde_json::Value {
        let tools: Vec<serde_json::Value> = self
            .tools
            .values()
            .map(|tool| {
                serde_json::json!({
                    "type": "function",
                    "function": tool.definition()
                })
            })
            .collect();

        serde_json::json!({ "tools": tools })
    }

    /// 生成为模型编写的 system prompt 片段
    ///
    /// 包含：
    /// 1. 工具使用说明
    /// 2. 可用工具列表（名称、描述、参数 schema）
    /// 3. 输出格式规范
    pub fn system_prompt_snippet(&self) -> String {
        if self.tools.is_empty() {
            return String::new();
        }

        let mut lines = vec![
            String::from("你是一个可以调用工具的助手。当用户的请求需要使用工具才能完成时，你必须调用对应的工具，不能假装自己无法执行或直接编造结果。"),
            String::from("\n## 可用工具\n"),
        ];

        for tool in self.tools.values() {
            let def = tool.definition();
            lines.push(format!("### {}\n{}", def.name, def.description));
            lines.push(format!(
                "参数: {}",
                serde_json::to_string_pretty(&def.parameters).unwrap()
            ));
            lines.push(String::new());
        }

        lines.push(String::from(
            "## 调用格式\n\
             需要调用工具时，在回复中按以下格式输出（禁止放在 <think> 标签内）：\n\
             <tool_call>{\"name\": \"工具名称\", \"arguments\": {\"参数名\": \"参数值\"}}</tool_call>\n\
             \n\
             示例：\n\
             <tool_call>{\"name\": \"read_file\", \"arguments\": {\"path\": \"/path/to/file.txt\"}}</tool_call>\n\
             \n\
             工具执行结果会以 <tool_result> 标签注入上下文，你再基于结果继续回答用户。\n\
             重要：如果用户提供了文件路径，你必须先调用工具读取，不能说「我无法访问文件系统」。",
        ));

        lines.join("\n")
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct DummyTool;
    impl Tool for DummyTool {
        fn name(&self) -> &str {
            "dummy"
        }
        fn definition(&self) -> ToolDef {
            ToolDef::new(
                "dummy",
                "一个测试工具",
                serde_json::json!({"type": "object", "properties": {}}),
            )
        }
        fn execute(&self, _args: &serde_json::Value) -> anyhow::Result<String> {
            Ok("dummy result".to_string())
        }
    }

    #[test]
    fn test_register_and_execute() {
        let mut registry = ToolRegistry::new();
        registry.register(DummyTool).unwrap();

        assert!(registry.has("dummy"));
        assert_eq!(registry.tool_names(), vec!["dummy"]);

        let result = registry
            .execute("dummy", &serde_json::json!({}))
            .unwrap();
        assert_eq!(result, "dummy result");
    }

    #[test]
    fn test_duplicate_register() {
        let mut registry = ToolRegistry::new();
        registry.register(DummyTool).unwrap();
        let result = registry.register(DummyTool);
        assert!(result.is_err());
    }

    #[test]
    fn test_unknown_tool() {
        let registry = ToolRegistry::new();
        let result = registry.execute("unknown", &serde_json::json!({}));
        assert!(result.is_err());
    }

    #[test]
    fn test_system_prompt_not_empty() {
        let mut registry = ToolRegistry::new();
        registry.register(DummyTool).unwrap();

        let snippet = registry.system_prompt_snippet();
        assert!(!snippet.is_empty());
        assert!(snippet.contains("read_file"));
        assert!(snippet.contains("tool_call"));
    }
}
