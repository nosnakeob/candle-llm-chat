use crate::config::BaseConfig;
use crate::models::q_qwen2::Which;
use crate::models::HubInfo;
use crate::tests::{gen_next_token, str2tokens};
use crate::utils::load::load_logits_processor;
use anyhow::{Error, Result};
use candle_examples::token_output_stream::TokenOutputStream;
use candle_transformers::models::quantized_qwen2::ModelWeights;
use hf_chat_template::ChatContext;
use std::io::Write;
use std::{env, io};

#[tokio::test]
async fn test_tokenizer() -> Result<()> {
    let config = BaseConfig::<Which>::default();
    println!("{:?}", config);

    let tokenizer = config.setup_tokenizer().await?;
    // println!("{:#?}", tokenizer.get_added_vocabulary());

    // token -> id
    let eos_token = config.which.info().eos_token;
    let eos_id = tokenizer.token_to_id(&eos_token).unwrap();

    // id -> token
    assert_eq!(
        tokenizer.decode(&[eos_id], false).map_err(Error::msg)?,
        tokenizer.id_to_token(eos_id).unwrap(),
    );

    let tokens = [
        103942, 73670, 3837, 112735, 113195, 102936, 101036, 11319, 87752, 99639, 97084, 103358,
        48443, 144236, 84141, 106, 48738, 198, 144185, 48840, 239, 61138, 198, 144736, 8908, 227,
        117, 56652, 48738, 198, 144251, 10236, 230, 109, 63109, 198, 136879, 38433, 104, 100979,
        105626, 198, 144538, 6567, 223, 233, 99242, 101047, 99396, 107261, 198, 144927, 90476, 251,
        77598, 198, 145048, 4891, 241, 255, 103738, 198, 144848, 68739, 115, 109959, 198, 144656,
        40666, 100, 48738, 198, 145379, 18137, 248, 122, 38182, 198, 145491, 86009, 101265, 271,
        102056, 18830, 100398, 104378, 100631, 103945, 108086, 3837, 73670, 106525, 104170, 6313,
        151645,
    ];
    println!("{}", tokenizer.decode(&tokens, true).map_err(Error::msg)?);
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
    let mut model = config.setup_model().await?;
    let mut tos = TokenOutputStream::new(config.setup_tokenizer().await?);
    let mut logits_processor =
        load_logits_processor(config.temperature, config.seed, config.top_k, config.top_p);
    let mut ctx = ChatContext::new(info.tokenizer_repo).await?;

    // 初始化上下文token列表
    let mut ctx_tokens = vec![];
    let eos_token = tos.tokenizer().token_to_id(info.eos_token).unwrap();
    let to_sample = config.sample_len.saturating_sub(1);

    let prompts = vec![
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

