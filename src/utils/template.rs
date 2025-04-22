use anyhow::{Error, Result};
use derive_new::new;
use minijinja::{context, Environment};
use serde::Serialize;
use serde_json::json;
use std::sync::LazyLock;

static TEMPLATE_ENV: LazyLock<Environment> = LazyLock::new(|| {
    let mut env = Environment::new();
    env.add_template(
        "qwen_template",
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/templates/qwen.jinja")),
    )
    .unwrap();
    env.add_template(
        "ds_template",
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/templates/ds.jinja")),
    )
    .unwrap();
    env
});

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

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ChatContext {
    pub messages: Vec<Message>,
    pub add_generation_prompt: bool,
    #[serde(skip_serializing)]
    pub template_type: TemplateType,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TemplateType {
    Qwen,
    DeepSeek,
}

impl ChatContext {
    pub fn new(template_type: TemplateType) -> Self {
        Self {
            messages: vec![],
            add_generation_prompt: true,
            template_type,
        }
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

        let template_name = match self.template_type {
            TemplateType::Qwen => "qwen_template",
            TemplateType::DeepSeek => "ds_template",
        };
        let template = TEMPLATE_ENV.get_template(template_name)?;
        template.render(&ctx).map_err(Error::msg)
    }

    pub fn set_template_type(&mut self, template_type: TemplateType) {
        self.template_type = template_type;
    }
}
