use crate::config::BaseConfig;
use crate::models::Forward;
use crate::models::HubInfo;
use crate::models::{q_llama, q_qwen2};
use crate::utils::get_user_prompt;
use crate::TextGeneration;
use anyhow::{Error, Result};
use candle::Tensor;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::{quantized_llama, quantized_qwen2};
use candle_transformers::utils::apply_repeat_penalty;
use futures_util::{pin_mut, StreamExt};
use std::io::Write;
use std::{env, io};
use tokenizers::Tokenizer;

mod quant;
mod qwen2;

fn str2tokens(string: &str, tokenizer: &Tokenizer) -> Result<Vec<u32>> {
    let tokens = tokenizer.encode(string, true).map_err(Error::msg)?;
    let tokens = tokens.get_ids().to_vec();

    Ok(tokens)
}

fn gen_next_token<W: Forward, Wi>(
    ctx_tokens: &[u32],
    idx_pos: usize,
    model: &mut W,
    logits_processor: &mut LogitsProcessor,
    config: &BaseConfig<Wi>,
    ans_start_idx: Option<usize>,
) -> Result<u32> {
    let input = match ans_start_idx {
        Some(_) => Tensor::new(&[*ctx_tokens.last().unwrap()], &config.device)?,
        // 首个字符
        None => Tensor::new(ctx_tokens, &config.device)?,
    }
    .unsqueeze(0)?;

    let mut logits = model.forward(&input, idx_pos)?.squeeze(0)?;

    if let Some(ans_start_idx) = ans_start_idx {
        if config.repeat_penalty != 1. {
            let ans_tokens = &ctx_tokens[ans_start_idx..];
            let start_at = ans_tokens.len().saturating_sub(config.repeat_last_n);
            logits = apply_repeat_penalty(&logits, config.repeat_penalty, &ans_tokens[start_at..])?;
        }
    }

    // 采样下一个token
    logits_processor.sample(&logits).map_err(Error::msg)
}

#[tokio::test]
async fn test_chat() -> Result<()> {
    tracing_subscriber::fmt::init();

    unsafe {
        env::set_var("HTTPS_PROXY", "http://127.0.0.1:10808");
    }
    let config = BaseConfig::<q_qwen2::Which>::default();
    println!("{config:?}");

    let mut text_gen = TextGeneration::<quantized_qwen2::ModelWeights, _>::new(config).await?;

    loop {
        // 获取用户输入
        let prompt_str = get_user_prompt();

        // 创建 stream 并 pin 它
        let stream = text_gen.chat(&prompt_str);
        pin_mut!(stream); // 使用 pin_mut! 宏来固定 stream

        while let Some(Ok(t)) = stream.next().await {
            print!("{t}");
            io::stdout().flush()?;
        }
    }
}
