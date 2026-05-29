//! 统一推理管线枚举
//!
//! 将 TextGeneration（纯文本）和 Qwen3VL（多模态）统一为一个接口，
//! 上层无需关心底层是哪个管线。

use anyhow::Result;
use futures_core::stream::Stream;
use std::pin::Pin;

use crate::model::config::InferenceConfig;
use crate::pipe::TextGeneration;
use crate::qwen3_vl::Qwen3VL;

pub enum Pipeline {
    Text(TextGeneration),
    VL(Qwen3VL),
}

impl Pipeline {
    /// 根据 model_id 自动创建对应管线
    ///
    /// `model_id` 示例：`"qwen3"`, `"qwen3.8b_q4"`, `"qwen3_vl"`, `"qwen3_vl.2b_base"`
    pub async fn new(model_id: &str, config: InferenceConfig) -> Result<Self> {
        let arch = model_id.split_once('.').map(|(a, _)| a).unwrap_or(model_id);
        match arch {
            "qwen3_vl" => Ok(Pipeline::VL(Qwen3VL::new(model_id, config).await?)),
            _ => Ok(Pipeline::Text(TextGeneration::new(model_id, config).await?)),
        }
    }

    pub async fn default() -> Result<Self> {
        Self::new("qwen3", InferenceConfig::default()).await
    }

    /// 流式对话
    pub fn chat<'a>(
        &'a mut self,
        prompt: &'a str,
    ) -> Pin<Box<dyn Stream<Item = Result<String>> + 'a>> {
        match self {
            Pipeline::Text(p) => Box::pin(p.chat(prompt)),
            Pipeline::VL(vl) => vl.chat(prompt, None),
        }
    }

    /// 完整响应对话
    pub fn chat_full(&mut self, prompt: &str) -> Result<String> {
        match self {
            Pipeline::Text(p) => {
                p.push_user_message(prompt);
                p.chat_full()
            }
            Pipeline::VL(vl) => vl.chat_full(prompt, None),
        }
    }
}
