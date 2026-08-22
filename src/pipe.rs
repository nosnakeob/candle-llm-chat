use crate::model::ModelInference;
use crate::model::config::{InferenceConfig, ModelLoader};
use crate::model::VisionInput;
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
use std::pin::Pin;
use tracing::info;

use crate::model::qwen3_vl::{self, Qwen3VL};

pub struct ChatPipeline {
    model: Box<dyn ModelInference>,
    tos: TokenOutputStream,
    logits_processor: LogitsProcessor,
    ctx: ChatContext,
    infer_conf: InferenceConfig,
    eos_token_id: u32,
    /// 额外需要停止的 token id（如 <|im_end|>，Qwen3.5 config 的 eos 是 <|endoftext|>
    /// 但 chat 模板实际以 <|im_end|> 结尾）
    stop_tokens: Vec<u32>,
}

impl ChatPipeline {
    pub async fn new(model_id: &str, config: InferenceConfig) -> Result<Self> {
        let registry = ModelRegistry::new()?;
        let hub_info = registry.get(model_id)?;
        let device = &config.device;
        let arch = model_id.split_once('.').map(|(a, _)| a).unwrap_or(model_id);

        let logits_processor =
            LogitsProcessor::new(config.seed, Some(config.temperature), config.top_p);

        let ctx = ChatContext::from_repo(&hub_info.tokenizer_repo).await?;

        let (model, tokenizer, eos_token_id, stop_tokens) = if arch == "qwen3_vl" {
            let (vl, tok, eos) = Qwen3VL::load(&hub_info, device).await?;
            (Box::new(vl) as Box<dyn ModelInference>, tok, eos, vec![])
        } else {            let pth = ApiBuilder::from_env()
                .build()?
                .model(hub_info.tokenizer_repo.clone())
                .get("config.json")
                .await?;
            let v: Value = serde_json::from_str(&fs::read_to_string(pth)?)?;
            // eos_token_id 可能在顶层或嵌套在 text_config（Qwen3.5+ 多模态配置格式）
            let eos = v
                .get("eos_token_id")
                .or_else(|| v.get("text_config").and_then(|tc| tc.get("eos_token_id")))
                .and_then(|x| x.as_u64())
                .ok_or_else(|| anyhow!("eos_token_id not found"))? as u32;

            // 从 tokenizer 解析 <|im_end|> 等额外停止 token
            let (m, tok) = ModelLoader::load(&hub_info, device).await?;
            let stop_tokens: Vec<u32> = ["<|im_end|>", "<|endoftext|>"]
                .iter()
                .filter_map(|s| tok.token_to_id(s))
                .filter(|&id| id != eos)
                .collect();
            (m, tok, eos, stop_tokens)
        };

        Ok(Self {
            model,
            tos: TokenOutputStream::new(tokenizer),
            logits_processor,
            ctx,
            infer_conf: config,
            eos_token_id,
            stop_tokens,
        })
    }

    pub async fn with_default_config(model_id: &str) -> Result<Self> {
        Self::new(model_id, InferenceConfig::default()).await
    }

    pub async fn default() -> Result<Self> {
        Self::with_default_config("qwen3").await
    }

    /// 流式对话（纯文本，用于直接展示给用户）
    pub fn chat<'a>(&'a mut self, prompt: &'a str) -> impl Stream<Item = Result<String>> + 'a {
        let mut answer = String::with_capacity(1024);
        self.ctx.push_msg(prompt);
        self.model.clr_kv_cache();

        try_stream!({
            let prompt = self.ctx.render()?;
            let mut ctx_tokens = self.str2tokens(&prompt)?;

            let start = std::time::Instant::now();
            let ans_start_idx = ctx_tokens.len();

            let mut decode_start: Option<std::time::Instant> = None;
            let mut decoded = 0usize;
            for index in 0..self.infer_conf.sample_len {
                let t0 = std::time::Instant::now();
                let next_token = if index == 0 {
                    self.gen_next_token(&ctx_tokens, 0, None, None)?
                } else {
                    if decode_start.is_none() { decode_start = Some(std::time::Instant::now()); }
                    self.gen_next_token(
                        &ctx_tokens,
                        ans_start_idx + index - 1,
                        Some(ans_start_idx),
                        None,
                    )?
                };
                let dt = t0.elapsed();
                tracing::info!("[q35-perf] idx={index} gen={dt:.1?}");
                ctx_tokens.push(next_token);

                if let Some(t) = self.tos.next_token(next_token)? {
                    answer.push_str(&t);
                    yield t;
                }

                if self.is_stop_token(next_token) {
                    break;
                }
                decoded += 1;
            }
            if let Some(ds) = decode_start {
                info!("[q35-perf] decode total: {} tok in {:.1?} = {:.2} tok/s", decoded, ds.elapsed(), decoded as f64 / ds.elapsed().as_secs_f64());
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

    /// 流式对话（多模态，支持图片）
    pub fn chat_with_images<'a>(
        &'a mut self,
        prompt: &'a str,
        images: &'a [&'a str],
    ) -> Pin<Box<dyn Stream<Item = Result<String>> + 'a>> {
        let vision_config = match self.model.vision_config() {
            Some(vc) => vc.clone(),
            None => {
                return Box::pin(try_stream! {
                    yield "[模型不支持视觉输入]".to_string();
                });
            }
        };

        let preprocessed = match qwen3_vl::preprocess_images(images, &vision_config, &self.infer_conf.device) {
            Ok(pp) => pp,
            Err(e) => {
                return Box::pin(try_stream! {
                    yield format!("[图像预处理失败: {}]", e);
                });
            }
        };
        let full_prompt = qwen3_vl::build_vision_prompt(prompt, preprocessed.num_placeholders);
        self.ctx.push_msg(&full_prompt);

        Box::pin(try_stream! {
            let rendered = self.ctx.render()?;
            let mut ctx_tokens = self.str2tokens(&rendered)?;
            let span = qwen3_vl::find_image_token_span(&ctx_tokens, vision_config.image_token_id);
            let vision = VisionInput {
                pixel_values: preprocessed.pixel_values,
                image_grid_thw: preprocessed.image_grid_thw,
                image_token_span: span,
            };

            let mut answer = String::with_capacity(1024);
            self.model.clr_kv_cache();

            let start = std::time::Instant::now();
            let ans_start_idx = ctx_tokens.len();

            for index in 0..self.infer_conf.sample_len {
                let v_ref = if index == 0 { Some(&vision) } else { None };
                let next_token = if index == 0 {
                    self.gen_next_token(&ctx_tokens, 0, None, v_ref)?
                } else {
                    self.gen_next_token(
                        &ctx_tokens,
                        ans_start_idx + index - 1,
                        Some(ans_start_idx),
                        None,
                    )?
                };
                ctx_tokens.push(next_token);

                if let Some(t) = self.tos.next_token(next_token)? {
                    answer.push_str(&t);
                    yield t;
                }

                if self.is_stop_token(next_token) {
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
    /// 调用方负责将 user message 推入 context。
    pub fn chat_full(&mut self) -> Result<String> {
        let answer = self.generate_inner(None)?;
        self.ctx.push_msg(&answer);
        Ok(answer)
    }

    /// 完整响应对话（多模态，支持图片）
    pub fn chat_full_with_images(&mut self, prompt: &str, images: &[&str]) -> Result<String> {
        let vision_config = self
            .model
            .vision_config()
            .ok_or_else(|| anyhow!("model does not support vision"))?
            .clone();

        let preprocessed =
            qwen3_vl::preprocess_images(images, &vision_config, &self.infer_conf.device)?;
        let full_prompt = qwen3_vl::build_vision_prompt(prompt, preprocessed.num_placeholders);
        self.ctx.push_msg(&full_prompt);

        let vision_data = (
            preprocessed.pixel_values,
            preprocessed.image_grid_thw,
            vision_config.image_token_id,
        );
        let answer = self.generate_inner(Some(vision_data))?;

        self.ctx.push_msg(&answer);
        Ok(answer)
    }

    // ─── Agent 接口 ──────────────────────────────────────────────────────

    pub fn inject_system_prompt(&mut self, system_prompt: &str) -> Result<()> {
        self.ctx.push_system(system_prompt);
        Ok(())
    }

    pub fn push_user_message(&mut self, message: &str) {
        self.ctx.push_msg(message);
    }

    pub fn push_tool_result(&mut self, result: &str) -> Result<()> {
        self.ctx.push_msg_system(result);
        Ok(())
    }

    pub fn push_assistant_continuation(&mut self) -> Result<()> {
        self.ctx.push_assistant("");
        Ok(())
    }

    // ─── 内部方法 ────────────────────────────────────────────────────────

    /// 共享生成循环，vision_data 为 (pixel_values, image_grid_thw, image_token_id)
    fn generate_inner(
        &mut self,
        vision_data: Option<(Tensor, Tensor, u32)>,
    ) -> Result<String> {
        let mut answer = String::with_capacity(1024);
        self.model.clr_kv_cache();

        let prompt = self.ctx.render()?;
        let mut ctx_tokens = self.str2tokens(&prompt)?;

        let vision: Option<VisionInput> = vision_data.map(|(pv, gt, tok_id)| {
            let span = qwen3_vl::find_image_token_span(&ctx_tokens, tok_id);
            VisionInput {
                pixel_values: pv,
                image_grid_thw: gt,
                image_token_span: span,
            }
        });

        let start = std::time::Instant::now();
        let ans_start_idx = ctx_tokens.len();

        for index in 0..self.infer_conf.sample_len {
            let v_ref = if index == 0 { vision.as_ref() } else { None };
            let next_token = if index == 0 {
                self.gen_next_token(&ctx_tokens, 0, None, v_ref)?
            } else {
                self.gen_next_token(
                    &ctx_tokens,
                    ans_start_idx + index - 1,
                    Some(ans_start_idx),
                    None,
                )?
            };
            ctx_tokens.push(next_token);

            if let Some(t) = self.tos.next_token(next_token)? {
                answer.push_str(&t);
            }

            if self.is_stop_token(next_token) {
                break;
            }
        }

        if let Some(t) = self.tos.decode_rest()? {
            answer.push_str(&t);
        }

        self.tos.clear();

        let elapsed = start.elapsed().as_secs_f64();
        if elapsed > 0.0 {
            info!(
                "speed: {:.2} token/s, total tokens: {}",
                (ctx_tokens.len() - ans_start_idx) as f64 / elapsed,
                ctx_tokens.len()
            );
        }

        Ok(answer)
    }

    /// 判断是否命中停止 token（eos 或额外 stop tokens）
    fn is_stop_token(&self, t: u32) -> bool {
        t == self.eos_token_id || self.stop_tokens.contains(&t)
    }

    fn str2tokens(&mut self, string: &str) -> Result<Vec<u32>> {
        let tokens = self
            .tos
            .tokenizer()
            .encode(string, true)
            .map_err(Error::msg)?;
        Ok(tokens.get_ids().to_vec())
    }

    fn gen_next_token(
        &mut self,
        ctx_tokens: &[u32],
        idx_pos: usize,
        ans_start_idx: Option<usize>,
        vision: Option<&VisionInput>,
    ) -> Result<u32> {
        let input_arr = match ans_start_idx {
            Some(_) => &[*ctx_tokens.last().unwrap()],
            None => ctx_tokens,
        };

        let input = Tensor::new(input_arr, &self.infer_conf.device)?.unsqueeze(0)?;

        let mut logits = self.model.forward(&input, idx_pos, vision)?;
        // 展平 leading 维度（GGUF 返回 (1, 1, vocab_size)，sample 需要 (vocab_size,)）
        while logits.rank() > 1 && logits.dim(0)? == 1 {
            logits = logits.squeeze(0)?;
        }

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

        self.logits_processor.sample(&logits).map_err(Error::msg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::get_user_prompt;
    use anyhow::Result;
    use futures_util::{StreamExt, pin_mut};
    use std::io;
    use std::io::Write;

    #[tokio::test]
    #[ignore]
    async fn test_prompt() -> Result<()> {
        let mut pipe = ChatPipeline::with_default_config("qwen3.4b_q4").await?;

        let prompts = vec![
            "我是snake，你给我记住了",
            "还记得我是谁吗",
            "你是谁",
            "给我笑一笑",
        ];

        for prompt_str in prompts {
            pipe.push_user_message(prompt_str);
            let answer = pipe.chat_full()?;
            println!("{}", answer);
        }

        Ok(())
    }

    #[tokio::test]
    #[ignore]
    async fn test_pipeline() -> Result<()> {
        tracing_subscriber::fmt::init();

        let mut pipe = ChatPipeline::with_default_config("qwen3.4b_abliterated").await?;

        for _ in 0..3 {
            let prompt_str = get_user_prompt();

            let stream = pipe.chat(&prompt_str);
            pin_mut!(stream);

            while let Some(r) = stream.next().await {
                print!("{}", r?);
                io::stdout().flush()?;
            }
        }

        Ok(())
    }
}
