//! 固定长度的 decode 基准（不依赖模型自己决定何时停）
//!
//! cargo test --release --test qwen3_5_bench -- --ignored --nocapture
//!
//! 与 e2e 测试的区别：这里把 sample_len 固定住，输出长度不随模型心情变化，
//! 于是不同改动之间的 tok/s 可以直接比。

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use candle_llm_chat::model::config::InferenceConfig;
    use candle_llm_chat::pipe::ChatPipeline;
    use futures_util::{StreamExt, pin_mut};
    use std::time::Instant;

    const PROMPT: &str = "请写一段不少于 300 字的说明文，介绍长江的地理特征与流域经济，不要分点。";
    const SAMPLE_LEN: usize = 256;

    /// 跑一轮固定长度解码，返回 (token 数, prefill 秒, 稳态 decode 秒)
    ///
    /// 用"首 token 到达时刻"切分两段：之前是 prefill，之后是逐 token 解码。
    /// 把两段混在一起算平均速率是之前基准误导自己的原因。
    async fn run_once(pipe: &mut ChatPipeline) -> Result<(usize, f64, f64)> {
        let start = Instant::now();
        let stream = pipe.chat(PROMPT);
        pin_mut!(stream);
        let mut tokens = 0usize;
        let mut first_at = None;
        while let Some(t) = stream.next().await {
            t?;
            tokens += 1;
            if first_at.is_none() {
                first_at = Some(start.elapsed().as_secs_f64());
            }
        }
        let total = start.elapsed().as_secs_f64();
        let prefill = first_at.unwrap_or(total);
        Ok((tokens, prefill, total - prefill))
    }

    #[tokio::test]
    #[ignore] // 需要模型文件 + GPU
    async fn bench_decode_fixed_len() -> Result<()> {
        let _ = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::INFO)
            .try_init();
        // 预热用的短长度：避免把模型加载时间算进基准
        let conf = InferenceConfig {
            sample_len: SAMPLE_LEN,
            ..Default::default()
        };
        // 允许用环境变量换模型，用来对比不同规模下每 token 固定开销被摊薄的程度
        let model_id = std::env::var("BENCH_MODEL").unwrap_or_else(|_| "qwen3_5.0_8b_q4".into());
        println!("model = {model_id}");
        let mut pipe = ChatPipeline::new(&model_id, conf).await?;

        // 预热一次（首轮包含 CUDA 上下文初始化与 kernel 编译）
        let _ = run_once(&mut pipe).await?;

        let mut rates = Vec::new();
        for round in 0..3 {
            // 每轮清空历史：否则上一轮的长回答会留在 prompt 里，长度逐轮翻倍，
            // 测到的是"上下文越来越长"，而不是稳定的 decode 速率。
            pipe.clear_history()?;
            let (tokens, prefill, decode) = run_once(&mut pipe).await?;
            let rate = if decode > 0.0 {
                tokens as f64 / decode
            } else {
                0.0
            };
            println!(
                "round {round}: {tokens} tok | prefill {prefill:.3}s | decode {decode:.2}s = {rate:.2} tok/s"
            );
            rates.push(rate);
        }
        rates.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!("median decode = {:.2} tok/s", rates[rates.len() / 2]);
        Ok(())
    }
}
