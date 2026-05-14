use crate::model::ModelInference;
use crate::model::config::{InferenceConfig, ModelLoader};
use crate::model::registry::ModelRegistry;
use crate::utils::chat::ChatContext;
use anyhow::{Error, Result};
use async_stream::try_stream;
use candle::Tensor;
use candle_examples::token_output_stream::TokenOutputStream;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::utils::apply_repeat_penalty;
use futures_core::stream::Stream;
use hf_hub::api::tokio::ApiBuilder;
use serde_json::Value;
use std::fs;
use tracing::info;

pub struct TextGeneration {
    model: Box<dyn ModelInference>,
    tos: TokenOutputStream,
    logits_processor: LogitsProcessor,
    ctx: ChatContext,
    infer_conf: InferenceConfig,
    eos_token_id: u32,
}

impl TextGeneration {
    pub async fn new(model_id: &str, config: InferenceConfig) -> Result<Self> {
        let registry = ModelRegistry::new()?;
        let hub_info = registry.get(model_id)?;
        let (model, tokenizer) = ModelLoader::load(hub_info, &config.device).await?;

        let logits_processor =
            LogitsProcessor::new(config.seed, Some(config.temperature), config.top_p);

        let ctx = ChatContext::from_repo(&hub_info.tokenizer_repo).await?;

        let pth = ApiBuilder::from_env()
            .build()?
            .model(hub_info.tokenizer_repo.clone())
            .get("config.json")
            .await?;
        let v: Value = serde_json::from_str(&fs::read_to_string(pth)?)?;
        let eos_token_id = v
            .get("eos_token_id")
            .and_then(|x| x.as_u64())
            .ok_or_else(|| anyhow!("eos_token_id not found"))? as u32;

        Ok(Self {
            model,
            tos: TokenOutputStream::new(tokenizer),
            logits_processor,
            ctx,
            infer_conf: config,
            eos_token_id,
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

    /// 流式对话（用于直接展示给用户）
    ///
    /// 适用于普通对话，不需要检测工具调用。
    pub fn chat<'a>(&'a mut self, prompt: &'a str) -> impl Stream<Item = Result<String>> + 'a {
        let mut answer = String::with_capacity(1024);
        self.ctx.push_msg(prompt);
        // 开始新的推理时清空 KV 缓存
        self.model.clr_kv_cache();

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

    /// 完整响应对话（用于 Agent tool calling 循环）
    ///
    /// 与 `chat()` 不同，此方法返回完整的响应字符串，而不是流式。
    /// 调用方负责将 user message 推入 context。
    ///
    /// 典型使用方式：
    /// ```ignore
    /// ctx.push_msg(user_message);
    /// model.clr_kv_cache();
    /// let response = pipe.chat_full()?;
    /// ctx.push_msg(&response);  // 助手回复入上下文
    /// ```
    pub fn chat_full(&mut self) -> Result<String> {
        let mut answer = String::with_capacity(1024);
        self.model.clr_kv_cache();

        let prompt = self.ctx.render()?;
        let mut ctx_tokens = self.str2tokens(&prompt)?;

        let start = std::time::Instant::now();
        let ans_start_idx = ctx_tokens.len();

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
            }

            if next_token == self.eos_token_id {
                break;
            }
        }

        if let Some(t) = self.tos.decode_rest()? {
            answer.push_str(&t);
        }

        self.ctx.push_msg(&answer);
        self.tos.clear();

        info!(
            "speed: {:.2} token/s, total tokens: {}",
            (ctx_tokens.len() - ans_start_idx) as f64 / start.elapsed().as_secs_f64(),
            ctx_tokens.len()
        );

        Ok(answer)
    }

    /// 向对话上下文注入 system prompt（工具调用时使用）
    ///
    /// 将 system 消息加入 ChatContext，工具描述会通过 chat_template 渲染。
    pub fn inject_system_prompt(&mut self, system_prompt: &str) -> Result<()> {
        self.ctx.push_system(system_prompt);
        Ok(())
    }

    /// 向对话上下文推送用户消息
    pub fn push_user_message(&mut self, message: &str) {
        self.ctx.push_msg(message);
    }

    /// 向对话上下文推送工具执行结果（作为 system 消息注入）
    pub fn push_tool_result(&mut self, result: &str) -> Result<()> {
        // 工具结果以 system 角色注入，表示"工具返回的内容"
        self.ctx.push_msg_system(result);
        Ok(())
    }

    /// 推送空白的 assistant 消息，通知模型继续生成
    ///
    /// 在工具结果注入后调用，模型会在空消息后继续输出内容。
    pub fn push_assistant_continuation(&mut self) -> Result<()> {
        self.ctx.push_assistant("");
        Ok(())
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
        let mut logits = self
            .model
            .forward(&input, idx_pos)?
            .squeeze(0)?
            .squeeze(0)?;

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
    use crate::model::ModelInference;
    use crate::pipe::TextGeneration;
    use crate::utils::chat::ChatContext;
    use crate::utils::{get_user_prompt, env_guard::ProxyGuard};
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
        model: &mut Box<dyn ModelInference>,
        logits_processor: &mut LogitsProcessor,
        config: &InferenceConfig,
        ans_start_idx: Option<usize>,
    ) -> Result<u32> {
        let input = match ans_start_idx {
            Some(_) => Tensor::new(&[*ctx_tokens.last().unwrap()], &config.device)?,
            None => Tensor::new(ctx_tokens, &config.device)?,
        }
        .unsqueeze(0)?;

        let mut logits = model.forward(&input, idx_pos)?.squeeze(0)?.squeeze(0)?;

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
    #[ignore] // 需要下载模型文件
    async fn test_prompt() -> Result<()> {
        // let _proxy = ProxyGuard::new(7890);

        let registry = ModelRegistry::new()?;
        let hub_info = registry.get("qwen3.4b_q4")?;

        let (mut model, tokenizer) =
            ModelLoader::load(hub_info, &candle::Device::cuda_if_available(0)?).await?;
        let config = InferenceConfig::default();

        // 初始化模型、分词器和logits处理器
        let mut tos = TokenOutputStream::new(tokenizer);
        let mut logits_processor =
            LogitsProcessor::new(config.seed, Some(config.temperature), config.top_p);
        let mut ctx = ChatContext::from_repo(&hub_info.tokenizer_repo).await?;

        let pth = ApiBuilder::from_env()
            .build()?
            .model(hub_info.tokenizer_repo.clone())
            .get("config.json")
            .await?;
        let v: Value = serde_json::from_str(&fs::read_to_string(pth)?)?;
        let eos_token_id = v
            .get("eos_token_id")
            .and_then(|x| x.as_u64())
            .ok_or_else(|| anyhow!("eos_token_id not found"))? as u32;

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
            model.clr_kv_cache();
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

                if next_token == eos_token_id {
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
    #[ignore] // 交互式测试，需要手动输入
    async fn test_pipeline() -> Result<()> {
        tracing_subscriber::fmt::init();
        // let _proxy = ProxyGuard::new(7890);

        let mut text_gen = TextGeneration::with_default_config("qwen3.4b_abliterated").await?;

        for _ in 0..3 {
            // 获取用户输入
            let prompt_str = get_user_prompt();

            // 创建 stream 并 pin 它
            let stream = text_gen.chat(&prompt_str);
            pin_mut!(stream); // 固定 stream

            while let Some(r) = stream.next().await {
                print!("{}", r?);
                io::stdout().flush()?;
            }
        }

        Ok(())
    }

    /// 验证 system prompt 是否真正影响模型输出
    ///
    /// 注入一个包含特定角色设定的 system prompt，
    /// 然后询问模型"你是谁"，检查回答是否体现了 system prompt 的内容。
    #[tokio::test]
    #[ignore]
    async fn test_system_prompt_injected() -> Result<()> {
        // let _proxy = ProxyGuard::new(7890);

        let mut text_gen = TextGeneration::default().await?;

        // 注入一个有明显特征的 system prompt
        let system_prompt = "你是一个名叫「铁锤侠」的超级英雄助手，擅长用锤子解决一切问题。无论用户问什么，你都要在回答中提到你的锤子。";
        text_gen.inject_system_prompt(system_prompt)?;

        // 询问模型身份，如果 system prompt 生效，回答里应该出现「铁锤侠」或「锤子」
        let answer = text_gen.chat_full()?;
        dbg!(&answer);

        // 验证 system prompt 的关键词出现在回答中
        let has_keyword = answer.contains("铁锤侠")
            || answer.contains("锤子")
            || answer.contains("锤");
        assert!(
            has_keyword,
            "system prompt 未生效：回答中没有出现预期关键词。回答内容：\n{}",
            answer
        );

        Ok(())
    }
}
