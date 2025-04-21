use anyhow::{Error, Result};
use derive_new::new;
use minijinja::{context, Environment};
use serde::Serialize;
use serde_json::json;
use tera::to_value;
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
    Tool,
}

#[derive(Debug, Clone, Serialize, new, PartialEq)]
pub struct Message {
    pub role: Role,
    #[new(into)]
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[new(default)]
    pub tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ToolCall {
    name: String,
    arguments: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct Tool {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

#[derive(Debug, Clone, new, Serialize, PartialEq)]
pub struct ChatContext {
    pub messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    pub add_generation_prompt: bool,
}

impl Default for ChatContext {
    fn default() -> Self {
        Self {
            messages: vec![],
            tools: None,
            add_generation_prompt: true,
        }
    }
}

#[derive(Debug, Clone)]
pub enum TemplateType {
    Qwen,
    DeepSeek,
}

impl ChatContext {
    pub fn push_msg(&mut self, content: &str) {
        let role = match self.messages.last() {
            None => Role::User,
            Some(msg) => match msg.role {
                Role::User => Role::Assistant,
                _ => Role::User,
            }
        };
        self.messages.push(Message::new(role, content));
    }

    pub fn render(&self, template_type: TemplateType) -> Result<String> {
        if self.messages.is_empty() {
            bail!("no messages")
        }

        let ctx = serde_json::to_value(self)?;
        
        let template_name = match template_type {
            TemplateType::Qwen => "qwen_template",
            TemplateType::DeepSeek => "ds_template",
        };
        let template = TEMPLATE_ENV.get_template(template_name)?;
        template.render(&ctx).map_err(Error::msg)
    }
}
