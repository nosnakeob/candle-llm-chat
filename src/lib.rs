#[macro_use]
extern crate anyhow;
#[macro_use]
extern crate tracing;

use crate::config::BaseConfig;
use crate::models::{Forward, FromGGUF, HubInfo, HubModelInfo};
use crate::utils::load::{load_gguf, load_tokenizer};
use anyhow::{Error, Result};
use async_stream::try_stream;
use candle::quantized::gguf_file::Content;
use candle::{Device, Tensor};
use candle_examples::token_output_stream::TokenOutputStream;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::quantized_qwen2::ModelWeights;
use candle_transformers::utils::apply_repeat_penalty;
use futures_core::stream::Stream;
use hf_chat_template::{ChatContext, Message, Role};
use std::io::{Read, Seek};
use std::ops::Deref;
use tokenizers::Tokenizer;

mod config;
mod models;
#[cfg(test)]
mod tests;
mod utils;

// 一次对话
pub struct TextGeneration<Wi: HubInfo> {
    model: Wi::ModelWeight,
    tos: TokenOutputStream,
    logits_processor: LogitsProcessor,
    config: BaseConfig<Wi>,
    eos_token_id: u32,
    ctx: ChatContext,
}

impl<Wi: HubInfo> TextGeneration<Wi> {
    pub async fn new(config: BaseConfig<Wi>) -> Result<Self> {
        let tokenizer = config.setup_tokenizer().await?;
        let (model, eos_token_id) = config.setup_model().await?;

        Ok(Self {
            model,
            tos: TokenOutputStream::new(tokenizer),
            logits_processor: LogitsProcessor::new(
                config.seed,
                Some(config.temperature),
                config.top_p,
            ),
            ctx: ChatContext::new(config.which.info().tokenizer_repo).await?,
            config,
            eos_token_id,
        })
    }


    pub fn chat<'a>(&'a mut self, prompt: &'a str) -> impl Stream<Item = Result<String>> + 'a {
        let mut answer = String::with_capacity(1024);
        self.ctx.push_msg(prompt);

        try_stream!({
            let prompt = self.ctx.render()?;
            // println!("prompt: {}", prompt);
            let mut ctx_tokens = self.str2tokens(&prompt)?;

            let start = std::time::Instant::now();

            let ans_start_idx = ctx_tokens.len();

            // 循环生成回答
            for index in 0..self.config.sample_len {
                let next_token = if index == 0 {
                    self.gen_next_token(&ctx_tokens, 0, None)?
                } else {
                    self.gen_next_token(&ctx_tokens, ans_start_idx + index - 1, Some(ans_start_idx))?
                };
                ctx_tokens.push(next_token);

                if let Some(t) = self.tos.next_token(next_token)? {
                    answer.push_str(&t);
                    yield t;
                }

                if next_token == self.eos_token_id {
                    break;
                }
            }

            if let Some(t) = self.tos.decode_rest()? {
                answer.push_str(&t);
                yield t;
            }

            self.ctx.push_msg(&answer);
            self.tos.clear();

            info!(
                "speed: {:.2} token/s, total tokens: {}",
                (ctx_tokens.len() - ans_start_idx) as f64 / start.elapsed().as_secs_f64(),
                ctx_tokens.len()
            );
        })
    }

    fn str2tokens(&mut self, string: &str) -> Result<Vec<u32>> {
        let tokens = self
            .tos
            .tokenizer()
            .encode(string, true)
            .map_err(Error::msg)?;
        let tokens = tokens.get_ids().to_vec();

        Ok(tokens)
    }

    fn gen_next_token(
        &mut self,
        ctx_tokens: &Vec<u32>,
        idx_pos: usize,
        ans_start_idx: Option<usize>,
    ) -> Result<u32> {
        let input_arr = match ans_start_idx {
            Some(_) => &[*ctx_tokens.last().unwrap()],
            // 首个字符
            None => &**ctx_tokens,
        };

        let input = Tensor::new(input_arr, &self.config.device)?.unsqueeze(0)?;

        // 获取模型输出并压缩维度
        let mut logits = self.model.forward(&input, idx_pos)?.squeeze(0)?;

        // 非首个字符应用惩罚
        if let Some(ans_start_idx) = ans_start_idx {
            if self.config.repeat_penalty != 1. {
                let ans_tokens = &ctx_tokens[ans_start_idx..];
                let start_at = ans_tokens.len().saturating_sub(self.config.repeat_last_n);
                logits = apply_repeat_penalty(
                    &logits,
                    self.config.repeat_penalty,
                    &ans_tokens[start_at..],
                )?;
            }
        }

        // 采样下一个token
        self.logits_processor.sample(&logits).map_err(Error::msg)
    }
}
