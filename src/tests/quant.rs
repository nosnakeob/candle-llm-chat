use crate::config::BaseConfig;
use crate::models::q_llama::Which;
use crate::models::HubInfo;
use crate::tests::{gen_next_token, str2tokens};
use crate::utils::load::load_logits_processor;
use anyhow::{Error, Result};
use candle::Tensor;
use candle_examples::token_output_stream::TokenOutputStream;
use candle_transformers::models::quantized_llama::ModelWeights;
use hf_chat_template::ChatContext;
use std::env;
use std::io::{self, Write};

#[tokio::test]
async fn test_tokenizer() -> Result<()> {
    let config = BaseConfig::<Which>::default();
    println!("{:?}", config);

    let tokenizer = config.setup_tokenizer().await?;
    println!("{:#?}", tokenizer.get_added_vocabulary());

    // token -> id
    let eos_token = config.which.info().eos_token;
    let eos_id = tokenizer.token_to_id(&eos_token).unwrap();

    // id -> token
    assert_eq!(
        tokenizer.decode(&[eos_id], false).map_err(Error::msg)?,
        tokenizer.id_to_token(eos_id).unwrap(),
    );

    let tokens = [128011, 57668, 53901, 128012, 128013, 198];
    println!("{}", tokenizer.decode(&tokens, false).map_err(Error::msg)?);
    // 单个token-/>字符
    tokens.iter().for_each(|t| {
        print!("{}", tokenizer.decode(&[*t], true).unwrap());
    });
    println!();

    let mut tos = TokenOutputStream::new(tokenizer);
    tokens.iter().for_each(|t| {
        if let Some(t) = tos.next_token(*t).unwrap() {
            print!("{}", t);
        }
    });
    if let Some(t) = tos.decode_rest()? {
        print!("{}", t);
    }
    println!();

    Ok(())
}

#[tokio::test]
async fn test_prompt() -> Result<()> {
    candle::cuda::set_gemm_reduced_precision_f16(true);
    candle::cuda::set_gemm_reduced_precision_bf16(true);
    unsafe {
        env::set_var("HTTPS_PROXY", "http://127.0.0.1:10808");
    }
    let config = BaseConfig::<Which>::default();
    println!("{config:?}");

    let info = config.which.info();
    // 初始化模型、分词器和logits处理器
    let mut model = config.setup_model::<ModelWeights>().await?;
    let mut tos = TokenOutputStream::new(config.setup_tokenizer().await?);
    let mut logits_processor =
        load_logits_processor(config.temperature, config.seed, config.top_k, config.top_p);
    let mut ctx = ChatContext::new(info.tokenizer_repo).await?;

    // 初始化上下文token列表
    let mut ctx_tokens = vec![];
    let eos_token = tos.tokenizer().token_to_id(info.eos_token).unwrap();
    let to_sample = config.sample_len.saturating_sub(1);

    let prompts = vec![
        // "你好",
        "我是snake，你给我记住了",
        "还记得我是谁吗",
        "你是谁",
        "给我笑一笑",
    ];
    let mut answer = String::new();

    for prompt_str in prompts {
        ctx.push_msg(prompt_str);
        let prompt = ctx.render()?;
        // println!("prompt: {}", prompt);
        ctx_tokens = str2tokens(&prompt, tos.tokenizer())?;
        // println!("{:?}", ctx_tokens);

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

        // print!("{next_token} ");
        if let Some(t) = tos.next_token(next_token)? {
            print!("{}", t);
            answer.push_str(&t);
            io::stdout().flush()?;
        }

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

            // print!("{next_token} ");
            if let Some(t) = tos.next_token(next_token)? {
                print!("{}", t);
                answer.push_str(&t);
                io::stdout().flush()?;
            }

            if next_token == eos_token {
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

    unsafe {
        env::remove_var("HTTPS_PROXY");
    }
    Ok(())
}
