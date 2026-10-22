//! Agent 示例：模型自主调用工具（读文件 / 抓网页）
//!
//! 运行：
//!   cargo run --release --example agent
//!   cargo run --release --example agent -- "读取 C:/tmp/a.txt 并总结内容"
//!
//! 流程：模型输出 <tool_call> → 框架执行工具 → 结果注入上下文 → 模型继续作答。

use anyhow::Result;
use candle_llm_chat::agent::Agent;
use futures_util::{StreamExt, pin_mut};
use std::io::Write;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("candle_llm_chat=info")
        .init();

    let prompt = std::env::args().skip(1).collect::<Vec<_>>().join(" ");
    let prompt = if prompt.is_empty() {
        "请访问 https://example.com 并总结页面内容".to_string()
    } else {
        prompt
    };

    // Agent::new() 使用默认模型 + 默认工具（read_file、web_fetch）
    let mut agent = Agent::new().await?;

    let stream = agent.chat(&prompt);
    pin_mut!(stream);

    let mut out = std::io::stdout();
    while let Some(chunk) = stream.next().await {
        print!("{}", chunk?);
        out.flush()?;
    }
    println!();

    Ok(())
}
