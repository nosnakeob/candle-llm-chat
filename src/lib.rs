#[macro_use]
extern crate tracing;

use std::sync::{Arc, Mutex};
use std::time;

use anyhow::Result;
use candle_core::Tensor;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::quantized_llama::ModelWeights;
use candle_transformers::utils::apply_repeat_penalty;
use tokio::sync::mpsc;
use tokio::sync::mpsc::Receiver;

use crate::arg::Args;
use crate::token_output_stream::TokenOutputStream;

mod token_output_stream;
mod arg;
mod utils;

pub struct ChatBot {
    model: Arc<Mutex<ModelWeights>>,
    tos: Arc<Mutex<TokenOutputStream>>,
    logits_processor: Arc<Mutex<LogitsProcessor>>,

    args: Args,
    eos_token: u32,
}

impl ChatBot {
    pub async fn from_args(args: Args) -> Result<Self> {
        let model = args.model().await?;

        let tokenizer = args.tokenizer().await?;
        let tos = TokenOutputStream::new(tokenizer);

        let eos_token = tos.tokenizer().token_to_id("<|end_of_turn|>").unwrap();

        let logits_processor = LogitsProcessor::new(args.seed, Some(args.temperature), None);

        Ok(Self {
            model: Arc::new(Mutex::new(model)),
            tos: Arc::new(Mutex::new(tos)),
            logits_processor: Arc::new(Mutex::new(logits_processor)),
            args,
            eos_token,
        })
    }

    pub async fn from_default_args() -> Result<Self> {
        Self::from_args(Args::default()).await
    }

    /// 创建子任务让模型不断预测(耗时任务)下一个token放入channel
    /// 用户可以从channel中获取回答
    pub fn chat(&mut self, input: String) -> Receiver<String> {
        let (tx, rx) = mpsc::channel(self.args.sample_len);

        let model = self.model.clone();
        let tos = self.tos.clone();
        let logits_processor = self.logits_processor.clone();
        let args = self.args.clone();
        let eos_token = self.eos_token;

        tokio::spawn(async move {
            let prompt = format!("User: {} <|end_of_turn|> Assistant: ", input.trim());

            let tokens = tos.lock().unwrap()
                .tokenizer()
                .encode(prompt, true).unwrap();

            let prompt_tokens = tokens.get_ids();
            let mut all_tokens = vec![];

            let start_prompt_processing = time::Instant::now();

            let mut next_token = {
                let input = Tensor::new(prompt_tokens, &args.device).unwrap()
                    .unsqueeze(0).unwrap();
                let logits = model.lock().unwrap()
                    .forward(&input, 0).unwrap();
                let logits = logits.squeeze(0).unwrap();
                logits_processor.lock().unwrap()
                    .sample(&logits).unwrap()
            };
            let prompt_dt = start_prompt_processing.elapsed();

            // 第一个单词
            let first_word = tos.lock().unwrap().next_token(next_token).unwrap();
            if let Some(t) = first_word {
                tx.send(t).await.unwrap();
            }

            let start_post_prompt = time::Instant::now();
            let to_sample = args.sample_len.saturating_sub(1);
            let mut sampled = 1;
            while sampled < to_sample {
                let input = Tensor::new(&[next_token], &args.device).unwrap()
                    .unsqueeze(0).unwrap();
                let logits = model.lock().unwrap()
                    .forward(&input, prompt_tokens.len() + sampled).unwrap();
                let logits = logits.squeeze(0).unwrap();
                let logits = if args.repeat_penalty == 1. {
                    logits
                } else {
                    let start_at = all_tokens.len().saturating_sub(args.repeat_last_n);
                    apply_repeat_penalty(
                        &logits,
                        args.repeat_penalty,
                        &all_tokens[start_at..],
                    ).unwrap()
                };
                next_token = logits_processor.lock().unwrap().sample(&logits).unwrap();
                all_tokens.push(next_token);
                let word = tos.lock().unwrap()
                    .next_token(next_token).unwrap();

                if let Some(t) = word {
                    tx.send(t).await.unwrap();
                }

                sampled += 1;
                if next_token == eos_token {
                    break;
                };
            }
            let rest = tos.lock().unwrap().decode_rest().unwrap();
            if let Some(rest) = rest {
                tx.send(rest).await.unwrap();
            }

            let dt = start_post_prompt.elapsed();

            info!("{} prompt tokens processed: {:.2} token/s",
            prompt_tokens.len(),
            prompt_tokens.len() as f64 / prompt_dt.as_secs_f64(),
        );
            info!("{sampled} tokens generated: {:.2} token/s",
            sampled as f64 / dt.as_secs_f64(),
        );
        });

        rx
    }
}

#[cfg(test)]
mod tests {
    use candle_core::utils::{with_avx, with_f16c, with_neon, with_simd128};
    use super::*;

    #[tokio::test]
    async fn t_chatbot_recv() -> Result<()> {
        tracing_subscriber::fmt::init();

        let mut chatbot = ChatBot::from_default_args().await?;

        let mut input = "hi".to_string();
        let mut rx = chatbot.chat(input);

        print!("output: ");
        // 一个一个token接收
        while let Some(word) = rx.recv().await {
            print!("{}", word);
        }
        println!();

        input = "what can you do?".to_string();
        rx = chatbot.chat(input);

        print!("output: ");
        // 一次接收一组, 可能模型来不及生成
        let cap = 68;
        let mut buf = Vec::with_capacity(cap);
        while rx.recv_many(&mut buf, cap).await > 0 {
            print!("{}", buf.join(""));
            buf.clear();
        }

        Ok(())
    }
}