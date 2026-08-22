//! Qwen3.5（Gated DeltaNet 混合架构）真实推理验证
//!
//! 运行: cargo test --lib -- --ignored qwen3_5
//! 首次运行会从 HF 下载模型（走 HF_ENDPOINT=hf-mirror 可加速）
#[cfg(test)]
mod tests {
    use candle_llm_chat::pipe::ChatPipeline;
    use anyhow::Result;
    use futures_util::{pin_mut, StreamExt};

    /// 验证 GGUF 量化版 Qwen3.5-0.8B 的完整对话链路：
    /// 模型加载 → chat template 渲染 → GDN 线性注意力前向 → 流式输出 → 多轮上下文
    /// 这是 qwen3_5 实现正确性的最小验证闸门。
    #[tokio::test]
    #[ignore] // 需要下载模型 + GPU
    async fn qwen3_5_0_8b_chat() -> Result<()> {
        tracing_subscriber::fmt()
            .with_max_level(tracing::Level::INFO)
            .init();
        let mut pipe = ChatPipeline::with_default_config("qwen3_5.0_8b_q4").await?;

        let prompts = vec![
            "你好，请用一句话介绍你自己",
            "我刚才让你做什么了？", // 考察多轮记忆
        ];

        let mut full_answer = String::new();
        for prompt in prompts {
            println!("\n=== 用户: {prompt}");
            let stream = pipe.chat(prompt);
            pin_mut!(stream);

            let mut answer = String::new();
            while let Some(token) = stream.next().await {
                let t = token?;
                print!("{t}");
                answer.push_str(&t);
            }
            println!();
            assert!(!answer.trim().is_empty(), "模型输出为空");
            full_answer.push_str(&answer);
        }

        // 基本合理性：回答应为可读文本而非乱码/空转
        let printable_ratio = full_answer
            .chars()
            .filter(|c| !c.is_control())
            .count() as f64
            / full_answer.chars().count() as f64;
        assert!(
            printable_ratio > 0.9,
            "输出疑似乱码（不可打印字符占比过高）: {full_answer}"
        );

        Ok(())
    }
}
