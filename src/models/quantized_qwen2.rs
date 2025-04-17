extern crate intel_mkl_src;

use crate::models::{BaseConfig, Setup};
use crate::utils::load::{load_gguf, load_logits_processor, load_tokenizer};
use anyhow::{Error, Result};
use async_stream::try_stream;
use candle::Tensor;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::quantized_qwen2::ModelWeights;
use candle_transformers::utils::apply_repeat_penalty;
use futures_core::stream::Stream;
use std::ops::Deref;
use tokenizers::Tokenizer;

#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub enum Which {
    // gpu推理报错
    W2_0_5b,
    W2_1_5b,
    W2_7b,
    W2_72b,
    W25_0_5b,
    W25_1_5b,
    #[default]
    W25_7b,
    W25_14b,
    W25_32b,
}

impl Which {
    pub fn model(&self) -> (&'static str, &'static str) {
        match self {
            Which::W2_0_5b => ("Qwen/Qwen2-0.5B-Instruct-GGUF", "qwen2-0_5b-instruct-q4_0"),
            Which::W2_1_5b => ("Qwen/Qwen2-1.5B-Instruct-GGUF", "qwen2-1_5b-instruct-q4_0"),
            Which::W2_7b => ("Qwen/Qwen2-7B-Instruct-GGUF", "qwen2-7b-instruct-q4_0"),
            Which::W2_72b => ("Qwen/Qwen2-72B-Instruct-GGUF", "qwen2-72b-instruct-q4_0"),
            Which::W25_0_5b => (
                "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
                "qwen2.5-0.5b-instruct-q4_0",
            ),
            Which::W25_1_5b => (
                "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
                "qwen2.5-1_5b-instruct-q4_0",
            ),
            Which::W25_7b => ("Qwen/Qwen2.5-7B-Instruct-GGUF", "qwen2.5-7b-instruct-q4_0"),
            Which::W25_14b => (
                "Qwen/Qwen2.5-14B-Instruct-GGUF",
                "qwen2.5-14b-instruct-q4_0",
            ),
            Which::W25_32b => (
                "Qwen/Qwen2.5-32B-Instruct-GGUF",
                "qwen2.5-32b-instruct-q4_0",
            ),
        }
    }

    pub fn tokenizer_repo(&self) -> &'static str {
        match self {
            Which::W2_0_5b => "Qwen/Qwen2-0.5B-Instruct",
            Which::W2_1_5b => "Qwen/Qwen2-1.5B-Instruct",
            Which::W2_7b => "Qwen/Qwen2-7B-Instruct",
            Which::W2_72b => "Qwen/Qwen2-72B-Instruct",
            Which::W25_0_5b => "Qwen/Qwen2.5-0.5B-Instruct",
            Which::W25_1_5b => "Qwen/Qwen2.5-1.5B-Instruct",
            Which::W25_7b => "Qwen/Qwen2.5-7B-Instruct",
            Which::W25_14b => "Qwen/Qwen2.5-14B-Instruct",
            Which::W25_32b => "Qwen/Qwen2.5-32B-Instruct",
        }
    }

    pub fn eos_token(&self) -> &'static str {
        "<|im_end|>"
    }
}

impl Setup for BaseConfig<Which> {
    type Weight = ModelWeights;

    async fn setup_model(&self) -> Result<Self::Weight> {
        let (repo, filename) = self.which.model();
        let (mut file, model) = load_gguf(repo, filename).await?;
        let model = ModelWeights::from_gguf(model, &mut file, &self.device)?;

        Ok(model)
    }

    async fn setup_tokenizer(&self) -> Result<Tokenizer> {
        load_tokenizer(self.which.tokenizer_repo()).await
    }
}

pub(crate) fn fmt_prompt(prompt: &str) -> String {
    format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n")
}

// 一次对话
pub struct TextGeneration {
    model: ModelWeights,
    tokenizer: Tokenizer,
    logits_processor: LogitsProcessor,
    config: BaseConfig<Which>,
    ctx_tokens: Vec<u32>,
    eos_token: u32,
}

impl TextGeneration {
    pub async fn new(config: BaseConfig<Which>) -> Result<Self> {
        let tokenizer = config.setup_tokenizer().await?;
        let eos_token = tokenizer.token_to_id(config.which.eos_token()).unwrap();

        Ok(Self {
            model: config.setup_model().await?,
            tokenizer,
            logits_processor: load_logits_processor(
                config.temperature,
                config.seed,
                config.top_k,
                config.top_p,
            ),
            config,
            ctx_tokens: Vec::with_capacity(1024),
            eos_token,
        })
    }

    pub fn chat<'a>(&'a mut self, prompt: &'a str) -> impl Stream<Item = Result<String>> + 'a {
        try_stream! {
            let prompt_tokens = self.process_prompt(prompt)?;
            self.ctx_tokens.extend_from_slice(&prompt_tokens);

            let start = std::time::Instant::now();

            // 生成第一个token
            let mut next_token = self.gen_next_token(0, None)?;
            let ans_start_idx = self.ctx_tokens.len();
            self.ctx_tokens.push(next_token);

            yield self.tokenizer.decode(&[next_token], true).map_err(Error::msg)?;

            // 循环生成回答
            for index in 0..self.config.sample_len.saturating_sub(1) {
                next_token = self.gen_next_token(ans_start_idx + index, Some(ans_start_idx))?;
                self.ctx_tokens.push(next_token);

                yield self.tokenizer.decode(&[next_token], true).map_err(Error::msg)?;

                if next_token == self.eos_token {
                    break;
                }
            }

            // 在生成速度统计前添加换行
            yield "\n".to_string();

            let dt = start.elapsed();
            info!("speed: {:.2} token/s",
                (self.ctx_tokens.len() - ans_start_idx) as f64 / dt.as_secs_f64(),
            );
        }
    }

    fn process_prompt(&self, prompt: &str) -> Result<Vec<u32>> {
        // 格式化提示词
        let prompt = fmt_prompt(&prompt);

        // 将提示词转换为token
        let tokens = self.tokenizer.encode(prompt, true).map_err(Error::msg)?;
        let tokens = tokens.get_ids().to_vec();

        Ok(tokens)
    }

    fn gen_next_token(&mut self, idx_pos: usize, ans_start_idx: Option<usize>) -> Result<u32> {
        let input = match ans_start_idx {
            Some(_) => Tensor::new(&[*self.ctx_tokens.last().unwrap()], &self.config.device)?,
            // 首个字符
            None => Tensor::new(&*self.ctx_tokens, &self.config.device)?,
        }
        .unsqueeze(0)?;

        // 获取模型输出并压缩维度
        let mut logits = self.model.forward(&input, idx_pos)?.squeeze(0)?;

        // 非首个字符应用惩罚
        if let Some(ans_start_idx) = ans_start_idx {
            if self.config.repeat_penalty != 1. {
                let ans_tokens = &self.ctx_tokens[ans_start_idx..];
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
