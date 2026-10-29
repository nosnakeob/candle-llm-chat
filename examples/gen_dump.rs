//! 贪心解码的确定性对照：同一 prompt、temperature=0、固定长度，把生成结果整段打印。
//!
//! 用途：改 KV cache / attention / RoPE 这类"应当行为等价"的实现后，用它在新旧两版
//! 二进制上各跑一次，直接 diff 输出。贪心解码 + 固定长度 ⇒ 逐 token 一致才说明等价，
//! 比"看着通顺"可靠。
//!
//! 运行：
//!   export PATH="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3/bin/x64:$PATH"
//!   cargo run --release --example gen_dump
//!   GEN_MODEL=qwen3.5.0_8b_q4 GEN_LEN=64 GEN_PROMPT="你好" cargo run --release --example gen_dump
use anyhow::Result;
use candle_llm_chat::model::config::InferenceConfig;
use candle_llm_chat::pipe::ChatPipeline;
use futures_util::{StreamExt, pin_mut};

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

#[tokio::main]
async fn main() -> Result<()> {
    let _ = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::WARN)
        .try_init();
    let model_id = std::env::var("GEN_MODEL").unwrap_or_else(|_| "qwen3_5.0_8b_q4".into());
    let prompt = std::env::var("GEN_PROMPT")
        .unwrap_or_else(|_| "请用一句话说明长江发源于哪里。".to_string());
    let len = env_usize("GEN_LEN", 64);
    let conf = InferenceConfig {
        sample_len: len,
        temperature: 0.0, // 贪心：去掉采样随机性
        ..Default::default()
    };
    let mut pipe = ChatPipeline::new(&model_id, conf).await?;
    let mut out = String::new();
    {
        let stream = pipe.chat(&prompt);
        pin_mut!(stream);
        while let Some(t) = stream.next().await {
            out.push_str(&t?);
        }
    }
    println!("--- model={model_id} len={len} prompt={prompt}");
    println!("{out}");
    println!("--- bytes={}", out.len());
    Ok(())
}
