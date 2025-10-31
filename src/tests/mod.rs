use crate::config::BaseConfig;
use crate::models::{Forward, HubInfo, q_llama, q_qwen2, q_qwen3};
use crate::pipe::TextGeneration;
use crate::utils::ProxyGuard;
use crate::utils::get_user_prompt;
use anyhow::{Error, Result};
use candle::Tensor;
use candle_examples::token_output_stream::TokenOutputStream;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::utils::apply_repeat_penalty;
use futures_util::{StreamExt, pin_mut};
use crate::chat::ChatContext;
use std::io;
use std::io::Write;
use tokenizers::Tokenizer;

mod quant;
mod qwen2;

fn str2tokens(string: &str, tokenizer: &Tokenizer) -> Result<Vec<u32>> {
    let tokens = tokenizer.encode(string, true).map_err(Error::msg)?;
    let tokens = tokens.get_ids().to_vec();

    Ok(tokens)
}

fn gen_next_token<Wi: HubInfo>(
    ctx_tokens: &[u32],
    idx_pos: usize,
    model: &mut Wi::ModelWeight,
    logits_processor: &mut LogitsProcessor,
    config: &BaseConfig<Wi>,
    ans_start_idx: Option<usize>,
) -> Result<u32> {
    let input = match ans_start_idx {
        Some(_) => Tensor::new(&[*ctx_tokens.last().unwrap()], &config.device)?,
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
async fn test_pipeline() -> Result<()> {
    tracing_subscriber::fmt::init();
    let _proxy = ProxyGuard::new("http://127.0.0.1:10808");

    let config = BaseConfig::<q_qwen2::Which>::default();
    println!("{config:?}");

    let mut text_gen = TextGeneration::new(config).await?;

    for _ in 0..3 {
        // 获取用户输入
        let prompt_str = get_user_prompt();

        // 创建 stream 并 pin 它
        let stream = text_gen.chat(&prompt_str);
        pin_mut!(stream); // 固定 stream

        while let Some(Ok(t)) = stream.next().await {
            print!("{t}");
            io::stdout().flush()?;
        }
    }

    Ok(())
}

#[tokio::test]
async fn test_prompt() -> Result<()> {
    let _proxy = ProxyGuard::new("http://127.0.0.1:10808");

    let config = BaseConfig::<q_qwen3::Which>::default();
    println!("{config:?}");

    let info = config.which.info();

    // 初始化模型、分词器和logits处理器
    let (mut model, eos_token_id) = config.setup_model().await?;
    let mut tos = TokenOutputStream::new(config.setup_tokenizer().await?);
    let mut logits_processor =
        LogitsProcessor::new(config.seed, Some(config.temperature), config.top_p);
    let mut ctx = ChatContext::new(info.tokenizer_repo).await?;

    // 初始化上下文token列表
    let mut ctx_tokens = vec![];

    let prompts = vec![
        "我是snake，你给我记住了",
        "还记得我是谁吗",
        "你是谁",
        "给我笑一笑",
    ];
    let mut answer = String::with_capacity(1024);

    for prompt_str in prompts {
        ctx.push_msg(prompt_str);
        let prompt = ctx.render()?;
        // println!("prompt: {}", prompt);
        ctx_tokens = str2tokens(&prompt, tos.tokenizer())?;

        let start = std::time::Instant::now();

        let ans_start_idx = ctx_tokens.len();

        // 统一处理token生成和输出
        for index in 0..config.sample_len {
            let next_token = gen_next_token(
                &ctx_tokens,
                if index == 0 { 0 } else { ans_start_idx + index - 1 },
                &mut model,
                &mut logits_processor,
                &config,
                if index == 0 { None } else { Some(ans_start_idx) },
            )?;
            ctx_tokens.push(next_token);

            if let Some(t) = tos.next_token(next_token)? {
                print!("{}", t);
                answer.push_str(&t);
                io::stdout().flush()?;
            }

            if next_token == eos_token_id {
                break;
            }
        }

        if let Some(t) = tos.decode_rest()? {
            print!("{}", t);
            answer.push_str(&t);
            io::stdout().flush()?;
        }

        ctx.push_msg(&answer);

        tos.clear();
        answer.clear();

        let dt = start.elapsed();

        println!(
            "\n\nspeed: {:.2} token/s",
            (ctx_tokens.len() - ans_start_idx) as f64 / dt.as_secs_f64(),
        );
    }

    Ok(())
}
