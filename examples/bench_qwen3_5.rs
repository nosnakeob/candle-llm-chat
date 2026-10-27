//! 固定长度的 decode 基准（不依赖模型自己决定何时停）
//!
//! 运行：
//!   cargo run --release --example bench_qwen3_5
//!   BENCH_MODEL=qwen3.0_8b_q4 cargo run --release --example bench_qwen3_5
//!
//! 与 e2e 测试的区别：这里把 sample_len 固定住，输出长度不随模型心情变化，
//! 于是不同改动之间的 tok/s 可以直接比。
//!
//! 放在 examples/ 而不是 tests/：它不是断言型测试，是测量工具；
//! 留在 tests/ 只会让 `cargo test` 多编译一份、再多一条恒被 #[ignore] 的条目。
//!
//! 速率取自 `ChatPipeline::last_decode_stats()` 而不是数流里的 chunk——
//! 流吐的是解码后的字符串片段，一个片段可能含多个 token，按片段计数会把
//! tok/s 算低一倍以上。
use anyhow::Result;
use candle_llm_chat::model::config::InferenceConfig;
use candle_llm_chat::pipe::{ChatPipeline, DecodeStats};
use futures_util::{StreamExt, pin_mut};

const PROMPT: &str = "请写一段不少于 300 字的说明文，介绍长江的地理特征与流域经济，不要分点。";
const SAMPLE_LEN: usize = 256;

/// 跑一轮固定长度解码，返回生成循环自己统计的实测数据
async fn run_once(pipe: &mut ChatPipeline) -> Result<DecodeStats> {
    // 流借用 pipe 到结束，用块限制作用域后才能读统计
    {
        let stream = pipe.chat(PROMPT);
        pin_mut!(stream);
        while let Some(t) = stream.next().await {
            t?;
        }
    }
    pipe.last_decode_stats()
        .ok_or_else(|| anyhow::anyhow!("生成未产生统计信息"))
}

#[tokio::main]
async fn main() -> Result<()> {
    let _ = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::WARN)
        .try_init();
    let conf = InferenceConfig {
        sample_len: SAMPLE_LEN,
        ..Default::default()
    };
    // 允许用环境变量换模型，用来对比不同规模下每 token 固定开销被摊薄的程度
    let model_id = std::env::var("BENCH_MODEL").unwrap_or_else(|_| "qwen3_5.0_8b_q4".into());
    println!("model = {model_id}");
    let mut pipe = ChatPipeline::new(&model_id, conf).await?;

    // 预热一次（首轮包含 CUDA 上下文初始化与 kernel 编译）
    let warm = run_once(&mut pipe).await?;
    println!(
        "warmup: {} tok | prefill {:.3}s | decode {:.2}s = {:.2} tok/s",
        warm.tokens,
        warm.prefill_secs,
        warm.decode_secs,
        warm.tok_per_sec()
    );

    let mut rates = Vec::new();
    for round in 0..3 {
        // 每轮清空历史：否则上一轮的长回答会留在 prompt 里，长度逐轮翻倍，
        // 测到的是"上下文越来越长"，而不是稳定的 decode 速率。
        pipe.clear_history()?;
        let s = run_once(&mut pipe).await?;
        println!(
            "round {round}: {} tok | prefill {:.3}s | decode {:.2}s = {:.2} tok/s",
            s.tokens,
            s.prefill_secs,
            s.decode_secs,
            s.tok_per_sec()
        );
        rates.push(s.tok_per_sec());
    }
    rates.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!("median decode = {:.2} tok/s", rates[rates.len() / 2]);
    Ok(())
}