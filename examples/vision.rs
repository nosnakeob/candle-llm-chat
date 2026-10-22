//! 多模态示例：让 Qwen3-VL 看图回答
//!
//! 运行：
//!   cargo run --release --example vision -- 图片路径 你的问题
//!
//! 默认用 models.toml 里的 qwen3_vl.2b_base。

use anyhow::Result;
use candle_llm_chat::model::config::InferenceConfig;
use candle_llm_chat::pipe::ChatPipeline;
use futures_util::{StreamExt, pin_mut};
use std::io::Write;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("candle_llm_chat=info")
        .init();

    let mut args = std::env::args().skip(1);
    let image = args.next().unwrap_or_else(|| {
        eprintln!("用法: cargo run --release --example vision -- <图片路径> [问题]");
        std::process::exit(2)
    });
    let question = args.next().unwrap_or_else(|| "请描述这张图片".to_string());

    let mut pipe = ChatPipeline::new("qwen3_vl.2b_base", InferenceConfig::default()).await?;

    let images = [image.as_str()];
    let stream = pipe.chat_with_images(&question, &images);
    pin_mut!(stream);

    let mut out = std::io::stdout();
    while let Some(chunk) = stream.next().await {
        print!("{}", chunk?);
        out.flush()?;
    }
    println!();

    Ok(())
}
