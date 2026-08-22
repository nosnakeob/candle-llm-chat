use anyhow::{Error, Result, bail};
use derive_new::new;
use hf_hub::api::tokio::{Api, ApiBuilder};
use minijinja::Environment;
use minijinja_contrib::pycompat;
use serde::Serialize;
use serde_json::Value;
use std::fs::File;
use std::io::BufReader;
use std::ops::{Deref, DerefMut};

/// 构造对齐 HF 模板约定的 Jinja 环境：
/// - pycompat 回调支持模板中的 Python 风格方法调用（切片等）
/// - raise_exception 函数（HF 官方模板常用）
fn template_env() -> Environment<'static> {
    let mut env = Environment::new();
    env.set_unknown_method_callback(pycompat::unknown_method_callback);
    env.add_function("raise_exception", |msg: String| -> Result<String, minijinja::Error> {
        Err(minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, msg))
    });
    env
}

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

#[derive(Clone)]
pub struct ChatContext {
    pub messages: Vec<Message>,
    add_generation_prompt: bool,
    // qwen3 特有，渲染时作为模板上下文变量传入
    pub enable_thinking: bool,
    env: Environment<'static>,
}

impl std::fmt::Debug for ChatContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChatContext")
            .field("messages", &self.messages.len())
            .field("add_generation_prompt", &self.add_generation_prompt)
            .field("enable_thinking", &self.enable_thinking)
            .finish()
    }
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
        let mut env = template_env();
        // 传 String 使 Cow::Owned，环境持有模板所有权（'static）
        env.add_template_owned("chat", template_str.to_string())
            .map_err(Error::msg)?;
        Ok(Self {
            messages: vec![],
            add_generation_prompt: true,
            enable_thinking: false,
            env,
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

    /// 推送 system 消息（工具描述等固定上下文）
    pub fn push_system(&mut self, content: &str) {
        self.messages.push(Message::new(Role::System, content));
    }

    /// 以 system 角色推送消息（不改变后续自动切换逻辑）
    ///
    /// 用于注入工具执行结果，结果会作为 system 消息出现在历史中，
    /// 不会被后续 `push_msg` 的角色切换逻辑影响。
    pub fn push_msg_system(&mut self, content: &str) {
        self.messages.push(Message::new(Role::System, content));
    }

    /// 推送 assistant 消息（直接指定，不走自动切换）
    pub fn push_assistant(&mut self, content: &str) {
        self.messages.push(Message::new(Role::Assistant, content));
    }

    /// 清空消息历史，保留 system prompt
    pub fn clear_history(&mut self) {
        let system_messages: Vec<_> = self
            .messages
            .iter()
            .filter(|m| m.role == Role::System)
            .cloned()
            .collect();
        self.messages = system_messages;
    }

    /// 渲染为模板字符串
    ///
    /// 上下文变量对齐 HF `apply_chat_template` 约定：
    /// messages / add_generation_prompt / enable_thinking / bos_token / eos_token
    pub fn render(&self) -> Result<String> {
        if self.messages.is_empty() {
            bail!("no messages");
        }
        let template = self.env.get_template("chat").map_err(Error::msg)?;
        template
            .render(minijinja::context! {
                messages => &self.messages,
                add_generation_prompt => self.add_generation_prompt,
                enable_thinking => self.enable_thinking,
                bos_token => "",
                eos_token => "",
            })
            .map_err(Error::msg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 验证 push_msg 的角色自动切换逻辑（User → Assistant → User）
    #[tokio::test]
    async fn test_push_msg_role_switching() -> Result<()> {
        let mut ctx = ChatContext::from_repo("Qwen/Qwen3-4B-Instruct-2507").await?;

        ctx.push_msg("hello");
        ctx.push_msg("hi");
        ctx.push_msg("how are you");

        assert_eq!(ctx.messages[0].role, Role::User);
        assert_eq!(ctx.messages[1].role, Role::Assistant);
        assert_eq!(ctx.messages[2].role, Role::User);

        // push_message 手动指定角色
        ctx.push_message(Role::System, "system instruction");
        assert_eq!(ctx.messages[3].role, Role::System);
        assert_eq!(ctx.len(), 4);

        Ok(())
    }

    /// 验证 from_template 渲染结果正确（不依赖网络）
    #[tokio::test]
    async fn test_from_template_render() -> Result<()> {
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

    /// 验证 <think> 标签内容被过滤，只保留回答部分
    #[tokio::test]
    async fn test_thinking_content_stripped() -> Result<()> {
        let mut ctx = ChatContext::from_repo("Qwen/Qwen3-4B-Instruct-2507").await?;
        ctx.push_msg("hello");
        ctx.push_msg("<think>let me think about this</think>hi there!");

        assert_eq!(ctx.messages[1].content, "hi there!");
        Ok(())
    }
}
