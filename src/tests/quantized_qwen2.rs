use crate::utils::{get_user_prompt, setup_logits_processor};
use candle_examples::token_output_stream::TokenOutputStream;
use futures_util::{pin_mut, StreamExt};
use std::io::Write;
use std::{env, io};
use anyhow::Error;
use candle::Tensor;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::quantized_qwen2::ModelWeights;
use candle_transformers::utils::apply_repeat_penalty;
use tokenizers::Tokenizer;
use crate::models::quantized_qwen2::{Args, setup_tokenizer, fmt_prompt, setup_model, token2str, TextGeneration};

async fn setup_tos(args: &Args) -> anyhow::Result<TokenOutputStream> {
    let tokenizer = setup_tokenizer(args).await?;
    Ok(TokenOutputStream::new(tokenizer))
}

fn process_prompt(prompt: &str, tokenizer: &Tokenizer) -> anyhow::Result<Vec<u32>> {
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
    args: &Args,
    ans_start_idx: Option<usize>,
) -> anyhow::Result<u32> {
    let logits = if let Some(ans_start_idx) = ans_start_idx {
        let input = Tensor::new(&[*ctx_tokens.last().unwrap()], &args.device)?.unsqueeze(0)?;
        let mut logits = model.forward(&input, idx_pos)?;
        logits = logits.squeeze(0)?;

        if args.repeat_penalty != 1. {
            let ans_tokens = &ctx_tokens[ans_start_idx..];
            let start_at = ans_tokens.len().saturating_sub(args.repeat_last_n);
            logits =
                apply_repeat_penalty(&logits, args.repeat_penalty, &ans_tokens[start_at..])?
        };

        logits
    } else {
        let input = Tensor::new(ctx_tokens, &args.device)?.unsqueeze(0)?;
        let logits = model.forward(&input, idx_pos)?;
        logits.squeeze(0)?
    };

    logits_processor.sample(&logits).map_err(Error::msg)
}

// 用tos输出奇怪
#[tokio::test]
async fn test_prompt() -> anyhow::Result<()> {
    env::set_var("HTTPS_PROXY", "http://127.0.0.1:10808");
    let args = Args::default();
    println!("{args:?}");

    // 初始化模型、分词器和logits处理器
    let mut model = setup_model(&args).await?;
    let mut tos = setup_tos(&args).await?;
    let mut logits_processor =
        setup_logits_processor(args.temperature, args.seed, args.top_k, args.top_p);

    // 初始化上下文token列表
    let mut ctx_tokens = vec![];
    let eos_token = *tos.tokenizer().get_vocab(true).get("<|im_end|>").unwrap();
    let to_sample = args.sample_len.saturating_sub(1);

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
            &args,
            None,
        )?;
        let ans_start_idx = ctx_tokens.len();
        ctx_tokens.push(next_token);
        if let Some(t) = tos.next_token(next_token)? {
            print!("{t}");
            io::stdout().flush()?;
        }

        // 循环生成回答
        for index in 0..to_sample {
            next_token = gen_next_token(
                &ctx_tokens,
                ans_start_idx + index,
                &mut model,
                &mut logits_processor,
                &args,
                Some(ans_start_idx),
            )?;
            ctx_tokens.push(next_token);

            if let Some(t) = tos.next_token(next_token)? {
                print!("{t}");
                io::stdout().flush()?;
            }

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

    env::remove_var("HTTPS_PROXY");
    Ok(())
}

#[tokio::test]
async fn test_prompt_wo_tos() -> anyhow::Result<()> {
    env::set_var("HTTPS_PROXY", "http://127.0.0.1:10808");
    let args = Args::default();
    println!("{args:?}");

    // 初始化模型、分词器和logits处理器
    let mut model = setup_model(&args).await?;
    let tokenizer = setup_tokenizer(&args).await?;
    let mut logits_processor =
        setup_logits_processor(args.temperature, args.seed, args.top_k, args.top_p);

    // 初始化上下文token列表
    let mut ctx_tokens = vec![];
    let eos_token = *tokenizer.get_vocab(true).get("<|im_end|>").unwrap();
    let to_sample = args.sample_len.saturating_sub(1);

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
            &args,
            None,
        )?;
        let ans_start_idx = ctx_tokens.len();
        ctx_tokens.push(next_token);
        if let Ok(t) = token2str(next_token, &tokenizer) {
            print!("{t}");
            io::stdout().flush()?;
        }

        // 循环生成回答
        for index in 0..to_sample {
            next_token = gen_next_token(
                &ctx_tokens,
                ans_start_idx + index,
                &mut model,
                &mut logits_processor,
                &args,
                Some(ans_start_idx),
            )?;
            ctx_tokens.push(next_token);

            if let Ok(t) = token2str(next_token, &tokenizer) {
                print!("{t}");
                io::stdout().flush()?;
            }

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

    env::remove_var("HTTPS_PROXY");
    Ok(())
}

#[tokio::test]
async fn test_chat() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    env::set_var("HTTPS_PROXY", "http://127.0.0.1:10808");
    let args = Args::default();
    println!("{args:?}");

    let mut text_gen = TextGeneration::new(args).await?;

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
