//! 最简对话示例：流式打印模型输出
//!
//! 运行：
//!   cargo run --release --example chat
//!   cargo run --release --example chat -- "介绍一下你自己"
//!
//! 第一次运行会从 HuggingFace 下载模型（国内建议先设 HF_ENDPOINT=https://hf-mirror.com）。

use anyhow::Result;
use candle_llm_chat::pipe::ChatPipeline;
use futures_util::{StreamExt, pin_mut};
use std::io::Write;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("candle_llm_chat=info")
        .init();

    let prompt = std::env::args().skip(1).collect::<Vec<_>>().join(" ");
    let prompt = if prompt.is_empty() {
        "介绍一下你自己".to_string()
    } else {
        prompt
    };

    // 默认模型是 models.toml 里 [qwen3] 下 default = true 的那个
    let mut pipe = ChatPipeline::default().await?;

    let stream = pipe.chat(&prompt);
    pin_mut!(stream);

    let mut out = std::io::stdout();
    while let Some(chunk) = stream.next().await {
        print!("{}", chunk?);
        out.flush()?;
    }
    println!();

    Ok(())
}
