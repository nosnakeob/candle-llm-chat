use crate::config::BaseConfig;
use crate::models::HubInfo;
use crate::models::q_qwen2::Which;
use crate::tests::{gen_next_token, str2tokens};
use crate::utils::ProxyGuard;
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
