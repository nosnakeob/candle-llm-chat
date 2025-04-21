use crate::utils::template::{ChatContext, Message, Role, TemplateType};
use anyhow::{Ok, Result};

#[test]
fn t_ctx2prompt() -> Result<()> {
    let mut ctx = ChatContext::default();

    ctx.push_msg("hello");
    ctx.push_msg("hi");
    ctx.push_msg("how are you");

    assert_eq!(
        ctx.render(TemplateType::Qwen)?,
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
    assert_eq!(
        ctx.render(TemplateType::DeepSeek)?,
        "<｜User｜>hello<｜Assistant｜>hi<｜end▁of▁sentence｜><｜User｜>how are you<｜Assistant｜>"
    );

    Ok(())
}

#[test]
fn t_push_msg() -> Result<()> {
    let mut ctx = ChatContext::default();

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
