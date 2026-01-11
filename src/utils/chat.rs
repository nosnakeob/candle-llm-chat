use anyhow::{Error, Result, bail};
use derive_new::new;
use hf_hub::api::tokio::{Api, ApiBuilder};
use minijinja::{Environment, Template};
use minijinja_contrib::pycompat;
use serde::Serialize;
use serde_json::Value;
use std::fs::File;
use std::io::BufReader;
use std::ops::{Deref, DerefMut};
use std::sync::LazyLock;

/// Environment存在生命周期标注，放置全局避免在ChatContext中处理生命周期问题
static TEMPLATE_ENV: LazyLock<Environment> = LazyLock::new(|| {
    let mut env = Environment::new();
    env.set_unknown_method_callback(pycompat::unknown_method_callback);
    env
});

pub async fn load_template(tokenizer_repo: &str) -> Result<Value> {
    let pth = ApiBuilder::from_env()
        .build()?
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
    add_generation_prompt: bool,
    // qwen3特有
    pub enable_thinking: bool,
    #[serde(skip_serializing)]
    template: Template<'static, 'static>,
}

impl Deref for ChatContext {
    type Target = Vec<Message>;

    fn deref(&self) -> &Self::Target {
        &self.messages
    }
}

impl DerefMut for ChatContext {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.messages
    }
}

impl ChatContext {
    /// 从tokenizer repo创建ChatContext
    pub async fn from_repo(tokenizer_repo: &str) -> Result<Self> {
        let template_str = load_template(&tokenizer_repo)
            .await?
            .as_str()
            .unwrap()
            .to_string();
        Self::from_template(&template_str)
    }

    /// 从模板字符串创建ChatContext
    pub fn from_template(template_str: &str) -> Result<Self> {
        Ok(Self {
            messages: vec![],
            add_generation_prompt: true,
            enable_thinking: false,
            template: TEMPLATE_ENV
                .template_from_str(Box::leak(template_str.to_string().into_boxed_str()))?,
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
        self.messages.push(Message::new(
            role,
            // 带思考过程只取回答
            content.split("</think>").last().unwrap(),
        ));
    }

    /// 手动添加指定角色的消息
    pub fn push_message(&mut self, role: Role, content: &str) {
        self.messages.push(Message::new(role, content));
    }

    /// 渲染为模板字符串
    pub fn render(&self) -> Result<String> {
        if self.messages.is_empty() {
            bail!("no messages");
        }
        let ctx = serde_json::to_value(self)?;
        self.template.render(&ctx).map_err(Error::msg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_push_msg() -> Result<()> {
        let mut ctx = ChatContext::from_repo("Qwen/Qwen3-4B-Instruct-2507").await?;
        ctx.push_msg("hello");
        ctx.push_msg("hi");
        ctx.push_msg("how are you");

        assert_eq!(
            ctx.messages,
            vec![
                Message::new(Role::User, "hello"),
                Message::new(Role::Assistant, "hi"),
                Message::new(Role::User, "how are you"),
            ]
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_manual_push() -> Result<()> {
        let mut ctx = ChatContext::from_repo("Qwen/Qwen3-4B-Instruct-2507").await?;
        ctx.push_message(Role::System, "You are a helpful assistant");
        ctx.push_message(Role::User, "hello");
        ctx.push_message(Role::Assistant, "hi there!");

        assert_eq!(ctx.len(), 3);
        assert_eq!(ctx.messages[0].role, Role::System);
        assert_eq!(ctx.messages[1].role, Role::User);
        assert_eq!(ctx.messages[2].role, Role::Assistant);
        Ok(())
    }

    #[tokio::test]
    async fn test_from_repo() -> Result<()> {
        let mut ctx = ChatContext::from_repo("Qwen/Qwen3-4B-Instruct-2507").await?;
        ctx.push_msg("hello");
        ctx.push_msg("hi");

        dbg!(ctx.render()?);

        Ok(())
    }

    #[tokio::test]
    async fn test_from_template() -> Result<()> {
        let template_str = r#"
{%- for message in messages %}
    {%- if message.role == 'user' %}
<|user|>{{ message.content }}<|end|>
    {%- elif message.role == 'assistant' %}
<|assistant|>{{ message.content }}<|end|>
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
<|assistant|>
{%- endif %}"#;

        let mut ctx = ChatContext::from_template(template_str)?;
        ctx.push_msg("hello");
        ctx.push_msg("hi");

        assert_eq!(
            ctx.render()?,
            r#"
<|user|>hello<|end|>
<|assistant|>hi<|end|>
<|assistant|>"#
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_thinking_content() -> Result<()> {
        let mut ctx = ChatContext::from_repo("Qwen/Qwen3-4B-Instruct-2507").await?;
        ctx.push_msg("hello");
        ctx.push_msg("<think>let me think about this</think>hi there!");

        assert_eq!(ctx.messages[1].content, "hi there!");
        Ok(())
    }
}
