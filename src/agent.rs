/// Agent - 支持文件读取和工具调用的智能体
///
/// 当前支持：
/// - .eml 邮件文件读取
/// - 通用文件读取（via Tool Calling）
///
/// # Tool Calling 流程
/// 1. 接收用户消息，推入对话上下文
/// 2. 调用模型获取完整响应
/// 3. 检测响应中的 `<tool_call>...</tool_call>` 块
/// 4. 若有工具调用：执行工具，将结果注入上下文，重复步骤 2-4
/// 5. 若无工具调用：清理输出中的 tool_call 块后流式输出给用户
use anyhow::Result;
use async_stream::try_stream;
use futures_core::stream::Stream;
use std::pin::Pin;
use tracing::info;

use crate::pipe::TextGeneration;
use crate::tools::{FileReadTool, ToolCallParser, ToolRegistry, WebFetchTool};

/// 最大工具调用轮次，防止模型陷入无限循环
const MAX_TOOL_ROUNDS: usize = 10;

pub struct Agent {
    pipe: TextGeneration,
    tools: ToolRegistry,
    /// 工具调用系统的 system prompt 片段
    tool_system_prompt: String,
}

impl Agent {
    /// 创建 Agent（无工具）
    pub async fn new() -> Result<Self> {
        Self::with_default_tools().await
    }

    /// 使用指定模型创建 Agent（无工具）
    pub async fn with_model(model_id: &str) -> Result<Self> {
        Self::with_model_and_tools(model_id, ToolRegistry::new()).await
    }

    /// 创建 Agent（带默认工具：文件读取）
    pub async fn with_default_tools() -> Result<Self> {
        Self::with_model_and_tools("qwen3", ToolRegistry::new()).await?.register_default_tools()
    }

    /// 使用指定模型和工具注册表创建 Agent
    pub async fn with_model_and_tools(model_id: &str, tools: ToolRegistry) -> Result<Self> {
        let tool_system_prompt = tools.system_prompt_snippet();
        Ok(Self {
            pipe: TextGeneration::with_default_config(model_id).await?,
            tools,
            tool_system_prompt,
        })
    }

    /// 注册默认工具（read_file + web_fetch）
    pub fn register_default_tools(mut self) -> Result<Self> {
        self.tools.register(FileReadTool::new())?;
        self.tools.register(WebFetchTool::new())?;
        self.tool_system_prompt = self.tools.system_prompt_snippet();
        Ok(self)
    }

    /// 注册自定义工具
    pub fn register_tool<T: crate::tools::Tool + 'static>(&mut self, tool: T) -> Result<()> {
        self.tools.register(tool)?;
        self.tool_system_prompt = self.tools.system_prompt_snippet();
        Ok(())
    }

    /// 对话入口（统一接口）
    ///
    /// - 若已注册工具：模型可主动调用工具，系统自动执行并注入结果，循环直到输出最终回答
    /// - 若未注册工具：退化为普通对话
    ///
    /// 返回清理后的流式输出（不含 tool_call 块）。
    pub fn chat<'a>(
        &'a mut self,
        prompt: &'a str,
    ) -> Pin<Box<dyn Stream<Item = Result<String>> + 'a>> {
        // 如果没有注册任何工具，退化为普通对话
        if self.tools.tool_names().is_empty() {
            return Box::pin(self.pipe.chat(prompt));
        }

        Box::pin(try_stream! {
            // 注入工具描述 system prompt（只注入一次）
            self.pipe.inject_system_prompt(&self.tool_system_prompt)?;
            // 推送用户消息
            self.pipe.push_user_message(prompt);

            let mut tool_rounds: usize = 0;

            loop {
                tool_rounds += 1;
                if tool_rounds > MAX_TOOL_ROUNDS {
                    let warning = format!(
                        "\n[警告：已达到最大工具调用轮次 ({})，强制停止工具调用循环]",
                        MAX_TOOL_ROUNDS
                    );
                    yield warning;
                    break;
                }

                // 获取模型完整响应
                let raw_response = self.pipe.chat_full()?;

                // 提取并执行工具调用
                let tool_calls: Vec<_> = ToolCallParser::parse(&raw_response)?;

                if tool_calls.is_empty() {
                    // 没有工具调用，流式输出清理后的内容并结束
                    let cleaned = ToolCallParser::strip(&raw_response);
                    for c in cleaned.chars() {
                        yield c.to_string();
                    }
                    break;
                }

                // 有工具调用，执行并注入结果
                for call in &tool_calls {
                    info!("执行工具调用: {} {:?}", call.name, call.arguments);

                    match self.tools.execute(&call.name, &call.arguments) {
                        Ok(result) => {
                            let tool_result_msg = format!(
                                "<tool_result name=\"{}\">{}</tool_result>",
                                call.name, result
                            );
                            self.pipe.push_tool_result(&tool_result_msg)?;
                            yield format!("[工具 {} 调用成功]\n", call.name);
                        }
                        Err(e) => {
                            let err_msg = format!("<tool_result name=\"{}\">错误: {}</tool_result>", call.name, e);
                            self.pipe.push_tool_result(&err_msg)?;
                            yield format!("[工具 {} 调用失败: {}]\n", call.name, e);
                        }
                    }
                }

                // 通知模型继续生成（注入空白 assistant 标记，让模型继续）
                self.pipe.push_assistant_continuation()?;
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::{pin_mut, StreamExt};

    /// 集成测试：模型通过 tool calling 读取 eml 文件并回答问题
    ///
    /// 流程：用户提供 eml 路径 → 模型调用 read_file → 系统读取文件 → 结果注入 → 模型总结邮件内容
    #[tokio::test]
    #[ignore]
    async fn test_read_eml_with_tools() -> Result<()> {
        let _proxy = crate::utils::env_guard::ProxyGuard::new(7897);

        let eml_path = std::env::temp_dir().join("test_agent.eml");
        std::fs::write(
            &eml_path,
            "From: boss@company.com\nTo: employee@company.com\nSubject: 项目进度汇报\nDate: Mon, 1 Jan 2025 10:00:00 +0800\n\n你好，请在本周五前提交Q1季度的项目进度报告，包括已完成的功能、遇到的问题以及下一步计划。谢谢。",
        ).unwrap();

        let mut agent = Agent::with_default_tools().await?;

        let prompt = format!(
            "请读取这封邮件并告诉我它要求我做什么，路径是：{}",
            eml_path.to_string_lossy()
        );
        let stream = agent.chat(&prompt);
        pin_mut!(stream);

        let mut answer = String::new();
        while let Some(Ok(token)) = stream.next().await {
            print!("{token}");
            answer.push_str(&token);
        }
        println!();

        dbg!(&answer);
        assert!(!answer.is_empty());
        assert!(
            answer.contains("报告") || answer.contains("进度") || answer.contains("周五"),
            "回答未体现邮件内容：\n{}",
            answer
        );

        let _ = std::fs::remove_file(&eml_path);
        Ok(())
    }

    /// 集成测试：验证 web_fetch tool calling
    ///
    /// 流程：用户问 → 模型输出 web_fetch tool_call → 系统抓取页面 → 结果注入 → 模型输出总结
    #[tokio::test]
    #[ignore] // 需要网络
    async fn test_chat_with_web_fetch() -> Result<()> {
        // let _proxy = crate::utils::env_guard::ProxyGuard::new(7890);

        let mut agent = Agent::with_default_tools().await?;

        let stream = agent.chat(
            "请访问 https://example.com 并告诉我这个页面的主要内容是什么",
        );
        pin_mut!(stream);

        let mut answer = String::new();
        while let Some(Ok(token)) = stream.next().await {
            print!("{token}");
            answer.push_str(&token);
        }
        println!();

        dbg!(&answer);
        assert!(!answer.is_empty());
        assert!(
            answer.contains("Example") || answer.contains("示例") || answer.contains("域名"),
            "回答未体现页面内容：\n{}",
            answer
        );

        Ok(())
    }
}
