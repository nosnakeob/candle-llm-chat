#[macro_use]
extern crate anyhow;

use anyhow::{Error, Result};
use derive_new::new;
use hf_hub::api::tokio::Api;
use minijinja::Environment;
use parking_lot::RwLock;
use serde::Serialize;
use serde_json::Value;
use std::fs::File;
use std::io::BufReader;
use std::sync::LazyLock;

static TEMPLATE_ENV: LazyLock<RwLock<Environment>> =
    LazyLock::new(|| RwLock::new(Environment::new()));

async fn load_template(tokenizer_repo: &str) -> Result<Value> {
    let pth = Api::new()?
        .model(tokenizer_repo.to_string())
        .get("tokenizer_config.json")
        .await?;

    let file = File::open(pth)?;
    let mut json: Value = serde_json::from_reader(BufReader::new(file))?;

    Ok(json["chat_template"].take())
}

#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    System,
    User,
    Assistant,
}

#[derive(Debug, Clone, Serialize, new, PartialEq)]
pub struct Message {
    pub role: Role,
    #[new(into)]
    pub content: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatContext {
    pub messages: Vec<Message>,
    pub add_generation_prompt: bool,

    #[serde(skip_serializing)]
    pub tokenizer_repo: String,
}

impl ChatContext {
    pub async fn new(tokenizer_repo: &str) -> Result<Self> {
        let tokenizer_repo = tokenizer_repo.to_string();

        if TEMPLATE_ENV.read().get_template(&tokenizer_repo).is_err() {
            let template = load_template(&tokenizer_repo).await?;

            TEMPLATE_ENV.write().add_template_owned(
                tokenizer_repo.clone(),
                template.as_str().unwrap().to_string(),
            )?;
        }

        Ok(Self {
            messages: vec![],
            add_generation_prompt: true,
            tokenizer_repo,
        })
    }

    /// 添加消息到对话上下文中  
    /// 发送消息角色根据上一条消息自动切换  
    /// User->Assistant->User->...
    pub fn push_msg(&mut self, content: &str) {
        let role = match self.messages.last() {
            None => Role::User,
            Some(msg) => match msg.role {
                Role::User => Role::Assistant,
                _ => Role::User,
            },
        };
        self.messages.push(Message::new(role, content));
    }

    pub fn render(&self) -> Result<String> {
        if self.messages.is_empty() {
            bail!("no messages")
        }

        let ctx = serde_json::to_value(self)?;

        TEMPLATE_ENV
            .read()
            .get_template(&self.tokenizer_repo)?
            .render(&ctx)
            .map_err(Error::msg)
    }

    pub async fn set_tokenizer_repo(&mut self, tokenizer_repo: &str) -> Result<()> {
        self.tokenizer_repo = tokenizer_repo.to_string();

        if TEMPLATE_ENV.read().get_template(tokenizer_repo).is_err() {
            let template = load_template(tokenizer_repo).await?;
            TEMPLATE_ENV.write().add_template_owned(
                tokenizer_repo.to_string(),
                template.as_str().unwrap().to_string(),
            )?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn t_push_msg() -> Result<()> {
        let mut template = ChatContext::new("Qwen/Qwen2.5-7B-Instruct").await?;

        template.push_msg("hello");
        template.push_msg("hi");
        template.push_msg("how are you");

        assert_eq!(
            template.messages,
            vec![
                Message::new(Role::User, "hello"),
                Message::new(Role::Assistant, "hi"),
                Message::new(Role::User, "how are you"),
            ]
        );

        Ok(())
    }

    #[tokio::test]
    async fn t_ctx2prompt() -> Result<()> {
        let mut template = ChatContext::new("Qwen/Qwen2.5-7B-Instruct").await?;

        template.push_msg("hello");
        template.push_msg("hi");
        template.push_msg("how are you");

        assert_eq!(
            template.render()?,
            "<|im_start|>system\n\
            You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n\
            <|im_start|>user\n\
            hello<|im_end|>\n\
            <|im_start|>assistant\n\
            hi<|im_end|>\n\
            <|im_start|>user\n\
            how are you<|im_end|>\n\
            <|im_start|>assistant\n"
        );
        template
            .set_tokenizer_repo("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
            .await?;
        assert_eq!(
            template.render()?,
            "<｜User｜>hello<｜Assistant｜>hi<｜end▁of▁sentence｜><｜User｜>how are you<｜Assistant｜><think>\n"
        );

        Ok(())
    }
}
