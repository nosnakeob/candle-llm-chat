use crate::models::quantized_qwen2::{TextGeneration, Which};
use crate::models::{BaseConfig, Setup};
use crate::tests::{gen_next_token, ModelWeight};
use crate::utils::get_user_prompt;
use crate::utils::load::load_logits_processor;
use anyhow::{Error, Result};
use candle_examples::token_output_stream::TokenOutputStream;
use futures_util::{pin_mut, StreamExt};
use std::io::Write;
use std::{env, io};
use tokenizers::Tokenizer;

fn process_prompt(prompt: &str, which: &Which, tokenizer: &Tokenizer) -> Result<Vec<u32>> {
    // 格式化提示词
    let prompt = which.fmt_prompt(&prompt);

    // 将提示词转换为token
    let tokens = tokenizer.encode(prompt, true).map_err(Error::msg)?;
    let tokens = tokens.get_ids().to_vec();

    Ok(tokens)
}

#[tokio::test]
async fn test_tokenizer() -> Result<()> {
    let config = BaseConfig::<Which>::default();
    let tokenizer = config.setup_tokenizer().await?;

    let eos_token = config.which.eos_token();
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

    let prompt = config.which.fmt_prompt("我是snake，你给我记住了");
    println!("{prompt}");
    
    let tokens = tokenizer
        .encode(prompt, true)
        .map_err(Error::msg)?
        .get_ids()
        .to_vec();
    println!("tokens: {tokens:?}");
    
    println!(
        "1st word: {}",
        tokenizer.decode(&[tokens[0]], false).map_err(Error::msg)?
    );
    println!("{}", tokenizer.decode(&tokens, true).map_err(Error::msg)?);

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
        let prompt_tokens = process_prompt(&prompt_str, &config.which, tos.tokenizer())?;
        ctx_tokens.extend_from_slice(&prompt_tokens);

        let start = std::time::Instant::now();

        // 生成第一个token
        let mut next_token = gen_next_token(
            &ctx_tokens,
            0,
            ModelWeight::Qwen(&mut model),
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
                ModelWeight::Qwen(&mut model),
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
        let prompt_tokens = process_prompt(&prompt_str, &config.which, &tokenizer)?;
        ctx_tokens.extend_from_slice(&prompt_tokens);

        let start = std::time::Instant::now();

        // 生成第一个token
        let mut next_token = gen_next_token(
            &ctx_tokens,
            0,
            ModelWeight::Qwen(&mut model),
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
                ModelWeight::Qwen(&mut model),
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
