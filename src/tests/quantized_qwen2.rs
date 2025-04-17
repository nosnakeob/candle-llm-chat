use crate::models::quantized_qwen2::{fmt_prompt, TextGeneration, Which};
use crate::models::{BaseConfig, Setup};
use crate::utils::get_user_prompt;
use crate::utils::load::load_logits_processor;
use anyhow::{Error, Result};
use candle::Tensor;
use candle_examples::token_output_stream::TokenOutputStream;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::quantized_qwen2::ModelWeights;
use candle_transformers::utils::apply_repeat_penalty;
use futures_util::{pin_mut, StreamExt};
use std::io::Write;
use std::{env, io};
use tokenizers::Tokenizer;

fn process_prompt(prompt: &str, tokenizer: &Tokenizer) -> Result<Vec<u32>> {
    // 格式化提示词
    let prompt = fmt_prompt(&prompt);

    // 将提示词转换为token
    let tokens = tokenizer.encode(prompt, true).map_err(Error::msg)?;
    let tokens = tokens.get_ids().to_vec();

    Ok(tokens)
}

fn gen_next_token(
    ctx_tokens: &[u32],
    idx_pos: usize,
    model: &mut ModelWeights,
    logits_processor: &mut LogitsProcessor,
    config: &BaseConfig<Which>,
    ans_start_idx: Option<usize>,
) -> Result<u32> {
    // 根据是否有答案起始索引选择输入张量
    let input = match ans_start_idx {
        Some(_) => Tensor::new(&[*ctx_tokens.last().unwrap()], &config.device)?,
        None => Tensor::new(ctx_tokens, &config.device)?,
    };

    // 获取模型输出并压缩维度
    let mut logits = model.forward(&input.unsqueeze(0)?, idx_pos)?.squeeze(0)?;

    // 如果有答案起始索引且需要重复惩罚，则应用惩罚
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
async fn test_tokenizer() -> Result<()> {
    let tokenizer = BaseConfig::<Which>::default().setup_tokenizer().await?;

    let eos_token = "<|im_end|>";
    // 单个
    let id = tokenizer.token_to_id(eos_token).unwrap();
    // 批量
    assert_eq!(*tokenizer.get_vocab(true).get(eos_token).unwrap(), id);

    assert_eq!(
        tokenizer.decode(&[id], true).map_err(Error::msg)?,
        tokenizer
            .id_to_token(id)
            .map(|t| if t == eos_token { String::new() } else { t })
            .unwrap(),
    );

    Ok(())
}

// 用tos多轮输出奇怪
#[tokio::test]
async fn test_prompt() -> Result<()> {
    unsafe {
        env::set_var("HTTPS_PROXY", "http://127.0.0.1:10808");
    }
    let config = BaseConfig::<Which>::default();
    println!("{config:?}");

    // 初始化模型、分词器和logits处理器
    let mut model = config.setup_model().await?;
    let mut tos = TokenOutputStream::new(config.setup_tokenizer().await?);
    let mut logits_processor =
        load_logits_processor(config.temperature, config.seed, config.top_k, config.top_p);

    // 初始化上下文token列表
    let mut ctx_tokens = vec![];
    let eos_token = tos
        .tokenizer()
        .token_to_id(config.which.eos_token())
        .unwrap();
    let to_sample = config.sample_len.saturating_sub(1);

    let prompts = vec![
        "我是snake，你给我记住了",
        "还记得我是谁吗",
        "你是谁",
        "给我笑一笑",
    ];

    for prompt_str in prompts {
        let prompt_tokens = process_prompt(&prompt_str, tos.tokenizer())?;
        ctx_tokens.extend_from_slice(&prompt_tokens);

        let start = std::time::Instant::now();

        // 生成第一个token
        let mut next_token = gen_next_token(
            &ctx_tokens,
            0,
            &mut model,
            &mut logits_processor,
            &config,
            None,
        )?;
        let ans_start_idx = ctx_tokens.len();
        ctx_tokens.push(next_token);

        print!("{}", tos.next_token(next_token)?.unwrap());
        io::stdout().flush()?;

        // 循环生成回答
        for index in 0..to_sample {
            next_token = gen_next_token(
                &ctx_tokens,
                ans_start_idx + index,
                &mut model,
                &mut logits_processor,
                &config,
                Some(ans_start_idx),
            )?;
            ctx_tokens.push(next_token);

            print!("{}", tos.next_token(next_token)?.unwrap());
            io::stdout().flush()?;

            if next_token == eos_token {
                break;
            }
        }

        let dt = start.elapsed();

        println!(
            "\n\n生成速度: {:.2} token/s",
            (ctx_tokens.len() - ans_start_idx) as f64 / dt.as_secs_f64(),
        );
    }

    unsafe {
        env::remove_var("HTTPS_PROXY");
    }
    Ok(())
}

#[tokio::test]
async fn test_prompt_wo_tos() -> Result<()> {
    candle::cuda::set_gemm_reduced_precision_f16(true);
    candle::cuda::set_gemm_reduced_precision_bf16(true);
    unsafe {
        env::set_var("HTTPS_PROXY", "http://127.0.0.1:10808");
    }

    let config = BaseConfig::<Which>::default();
    println!("{config:?}");

    // 初始化模型、分词器和logits处理器
    let mut model = config.setup_model().await?;
    let tokenizer = config.setup_tokenizer().await?;
    let mut logits_processor =
        load_logits_processor(config.temperature, config.seed, config.top_k, config.top_p);

    // 初始化上下文token列表
    let mut ctx_tokens = vec![];
    let eos_token = tokenizer.token_to_id(config.which.eos_token()).unwrap();
    let to_sample = config.sample_len.saturating_sub(1);

    let prompts = vec![
        "我是snake，你给我记住了",
        "还记得我是谁吗",
        "你是谁",
        "给我笑一笑",
    ];

    for prompt_str in prompts {
        // 将提示词转换为token并添加到上下文
        let prompt_tokens = process_prompt(&prompt_str, &tokenizer)?;
        ctx_tokens.extend_from_slice(&prompt_tokens);

        let start = std::time::Instant::now();

        // 生成第一个token
        let mut next_token = gen_next_token(
            &ctx_tokens,
            0,
            &mut model,
            &mut logits_processor,
            &config,
            None,
        )?;
        let ans_start_idx = ctx_tokens.len();
        ctx_tokens.push(next_token);

        print!(
            "{}",
            tokenizer.decode(&[next_token], true).map_err(Error::msg)?
        );
        io::stdout().flush()?;

        // 循环生成回答
        for index in 0..to_sample {
            next_token = gen_next_token(
                &ctx_tokens,
                ans_start_idx + index,
                &mut model,
                &mut logits_processor,
                &config,
                Some(ans_start_idx),
            )?;
            ctx_tokens.push(next_token);

            print!(
                "{}",
                tokenizer.decode(&[next_token], true).map_err(Error::msg)?
            );
            io::stdout().flush()?;

            if next_token == eos_token {
                break;
            }
        }

        // 将回答添加到上下文
        let dt = start.elapsed();

        println!(
            "\n\n生成速度: {:.2} token/s",
            (ctx_tokens.len() - ans_start_idx) as f64 / dt.as_secs_f64(),
        );
    }

    unsafe {
        env::remove_var("HTTPS_PROXY");
    }
    Ok(())
}

#[tokio::test]
async fn test_chat() -> Result<()> {
    tracing_subscriber::fmt::init();

    unsafe {
        env::set_var("HTTPS_PROXY", "http://127.0.0.1:10808");
    }
    let config = BaseConfig::default();
    println!("{config:?}");

    let mut text_gen = TextGeneration::new(config).await?;

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
