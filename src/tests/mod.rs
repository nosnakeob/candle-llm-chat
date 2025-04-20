use crate::models::quantized_qwen2::Which;
use crate::models::BaseConfig;
use anyhow::{Error, Result};
use candle::Tensor;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::{quantized_llama, quantized_qwen2};
use candle_transformers::utils::apply_repeat_penalty;
use tokenizers::Tokenizer;

mod qwen2;

enum ModelWeight<'a> {
    Qwen(&'a mut quantized_qwen2::ModelWeights),
    Quant(&'a mut quantized_llama::ModelWeights),
}

fn gen_next_token<W>(
    ctx_tokens: &[u32],
    idx_pos: usize,
    model: ModelWeight,
    logits_processor: &mut LogitsProcessor,
    config: &BaseConfig<W>,
    ans_start_idx: Option<usize>,
) -> Result<u32> {
    let input = match ans_start_idx {
        Some(_) => Tensor::new(&[*ctx_tokens.last().unwrap()], &config.device)?,
        // 首个字符
        None => Tensor::new(ctx_tokens, &config.device)?,
    }.unsqueeze(0)?;

    let mut logits = if let ModelWeight::Qwen(model) = model {
        model.forward(&input, idx_pos)?.squeeze(0)?
    } else if let ModelWeight::Quant(model) = model {
        model.forward(&input, idx_pos)?.squeeze(0)?
    } else {
        unreachable!()
    };

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
