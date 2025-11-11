use crate::model::Forward;
use crate::model::config::{InferenceConfig, ModelLoader};
use crate::model::hub::ModelInfo;
use crate::utils::chat::ChatContext;
use anyhow::{Error, Result};
use async_stream::try_stream;
use candle::Tensor;
use candle_examples::token_output_stream::TokenOutputStream;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::utils::apply_repeat_penalty;
use futures_core::stream::Stream;
use tracing::info;

pub struct TextGeneration {
    model: Box<dyn Forward>,
    tos: TokenOutputStream,
    logits_processor: LogitsProcessor,
    ctx: ChatContext,
    infer_conf: InferenceConfig,
    model_info: ModelInfo,
}

impl TextGeneration {
    pub async fn new(model_id: &str, config: InferenceConfig) -> Result<Self> {
        let (model, tokenizer, model_info) = ModelLoader::load(model_id, &config.device).await?;

        let logits_processor =
            LogitsProcessor::new(config.seed, Some(config.temperature), config.top_p);

        let ctx = ChatContext::from_template(&model_info.chat_template)?;

        Ok(Self {
            model,
            tos: TokenOutputStream::new(tokenizer),
            logits_processor,
            ctx,
            infer_conf: config,
            model_info,
        })
    }

    /// 便利构造函数 - 使用默认配置
    pub async fn with_default_config(model_id: &str) -> Result<Self> {
        Self::new(model_id, InferenceConfig::default()).await
    }

    /// 便利构造函数
    pub async fn default() -> Result<Self> {
        Self::with_default_config("qwen3").await
    }

    pub fn chat<'a>(&'a mut self, prompt: &'a str) -> impl Stream<Item = Result<String>> + 'a {
        let mut answer = String::with_capacity(1024);
        self.ctx.push_msg(prompt);

        try_stream!({
            let prompt = self.ctx.render()?;
            let mut ctx_tokens = self.str2tokens(&prompt)?;

            let start = std::time::Instant::now();
            let ans_start_idx = ctx_tokens.len();

            // 循环生成回答
            for index in 0..self.infer_conf.sample_len {
                let next_token = if index == 0 {
                    self.gen_next_token(&ctx_tokens, 0, None)?
                } else {
                    self.gen_next_token(
                        &ctx_tokens,
                        ans_start_idx + index - 1,
                        Some(ans_start_idx),
                    )?
                };
                ctx_tokens.push(next_token);

                if let Some(t) = self.tos.next_token(next_token)? {
                    answer.push_str(&t);
                    yield t;
                }

                if next_token == self.model_info.eos_token_id {
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
            None => &**ctx_tokens,
        };

        let input = Tensor::new(input_arr, &self.infer_conf.device)?.unsqueeze(0)?;

        // 获取模型输出并压缩维度
        let mut logits = self.model.forward(&input, idx_pos)?.squeeze(0)?;

        // 非首个字符应用惩罚
        if let Some(ans_start_idx) = ans_start_idx {
            if self.infer_conf.repeat_penalty != 1. {
                let ans_tokens = &ctx_tokens[ans_start_idx..];
                let start_at = ans_tokens
                    .len()
                    .saturating_sub(self.infer_conf.repeat_last_n);
                logits = apply_repeat_penalty(
                    &logits,
                    self.infer_conf.repeat_penalty,
                    &ans_tokens[start_at..],
                )?;
            }
        }

        // 采样下一个token
        self.logits_processor.sample(&logits).map_err(Error::msg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Forward;
    use crate::pipe::TextGeneration;
    use crate::utils::chat::ChatContext;
    use crate::utils::{get_user_prompt, proxy::ProxyGuard};
    use anyhow::{Error, Result};
    use candle::Tensor;
    use candle_transformers::generation::LogitsProcessor;
    use candle_transformers::utils::apply_repeat_penalty;
    use futures_util::{StreamExt, pin_mut};
    use std::io;
    use std::io::Write;
    use tokenizers::Tokenizer;

    fn str2tokens(string: &str, tokenizer: &Tokenizer) -> Result<Vec<u32>> {
        let tokens = tokenizer.encode(string, true).map_err(Error::msg)?;
        let tokens = tokens.get_ids().to_vec();

        Ok(tokens)
    }

    fn gen_next_token(
        ctx_tokens: &[u32],
        idx_pos: usize,
        model: &mut Box<dyn Forward>,
        logits_processor: &mut LogitsProcessor,
        config: &InferenceConfig,
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
                logits =
                    apply_repeat_penalty(&logits, config.repeat_penalty, &ans_tokens[start_at..])?;
            }
        }

        // 采样下一个token
        logits_processor.sample(&logits).map_err(Error::msg)
    }

    #[tokio::test]
    async fn test_prompt() -> Result<()> {
        let _proxy = ProxyGuard::new(7890);

        let (mut model, tokenizer, model_info) =
            ModelLoader::load("qwen3", &candle::Device::cuda_if_available(0)?).await?;
        let config = InferenceConfig::default();

        // 初始化模型、分词器和logits处理器
        let mut tos = TokenOutputStream::new(tokenizer);
        let mut logits_processor =
            LogitsProcessor::new(config.seed, Some(config.temperature), config.top_p);
        let mut ctx = ChatContext::from_template(&model_info.chat_template)?;

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
                    if index == 0 {
                        0
                    } else {
                        ans_start_idx + index - 1
                    },
                    &mut model,
                    &mut logits_processor,
                    &config,
                    if index == 0 {
                        None
                    } else {
                        Some(ans_start_idx)
                    },
                )?;
                ctx_tokens.push(next_token);

                if let Some(t) = tos.next_token(next_token)? {
                    print!("{}", t);
                    answer.push_str(&t);
                    io::stdout().flush()?;
                }

                if next_token == model_info.eos_token_id {
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

    #[tokio::test]
    async fn test_pipeline() -> Result<()> {
        tracing_subscriber::fmt::init();
        let _proxy = ProxyGuard::new(7890);

        let mut text_gen = TextGeneration::default().await?;

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
}
