use crate::model::ModelInference;
use crate::model::VisionInput;
use crate::model::config::{InferenceConfig, ModelLoader};
use crate::model::registry::ModelRegistry;
use crate::utils::chat::ChatContext;
use anyhow::{Error, Result};
use async_stream::try_stream;
use candle::{DType, Tensor};
use candle_examples::token_output_stream::TokenOutputStream;
use candle_transformers::generation::LogitsProcessor;
use futures_core::stream::Stream;
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
        } else {
            let pth = crate::utils::hf::fetch(
                &crate::utils::hf::model_repo(&hub_info.tokenizer_repo)?,
                "config.json",
            )
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

    /// 流式对话（纯文本）
    pub fn chat<'a>(&'a mut self, prompt: &'a str) -> impl Stream<Item = Result<String>> + 'a {
        self.ctx.push_msg(prompt);
        self.decode(None)
    }

    /// 流式对话（多模态，支持图片）
    ///
    /// 预处理失败或模型不支持视觉时，返回一条说明文字而不是报错，
    /// 保持流式接口的返回类型统一。
    pub fn chat_with_images<'a>(
        &'a mut self,
        prompt: &'a str,
        images: &'a [&'a str],
    ) -> Pin<Box<dyn Stream<Item = Result<String>> + 'a>> {
        match self.prepare_vision(prompt, images) {
            Err(e) => Box::pin(try_stream! { yield format!("[图像预处理失败: {e}]"); }),
            Ok(None) => Box::pin(try_stream! { yield "[模型不支持视觉输入]".to_string(); }),
            Ok(Some((pixel_values, image_grid_thw, token_id, full_prompt))) => {
                self.ctx.push_msg(&full_prompt);
                Box::pin(self.decode(Some((pixel_values, image_grid_thw, token_id))))
            }
        }
    }

    /// 完整响应对话（用于 Agent tool calling 循环）
    ///
    /// 调用方负责将 user message 推入 context。
    pub async fn chat_full(&mut self) -> Result<String> {
        self.generate_inner(None).await
    }

    /// 完整响应对话（多模态，支持图片）
    pub async fn chat_full_with_images(&mut self, prompt: &str, images: &[&str]) -> Result<String> {
        let (pixel_values, image_grid_thw, token_id, full_prompt) = self
            .prepare_vision(prompt, images)?
            .ok_or_else(|| anyhow!("model does not support vision"))?;
        self.ctx.push_msg(&full_prompt);
        self.generate_inner(Some((pixel_values, image_grid_thw, token_id)))
            .await
    }

    // ─── Agent 接口 ──────────────────────────────────────────────────────

    /// 清空对话历史（保留 system prompt），同时丢弃 KV cache / GDN 状态
    pub fn clear_history(&mut self) {
        self.ctx.clear_history();
        self.model.clr_kv_cache();
    }

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

    /// 图像预处理，返回 (pixel_values, image_grid_thw, image_token_id, 带占位符的 prompt)
    ///
    /// 模型不支持视觉时返回 Ok(None)。
    fn prepare_vision(
        &self,
        prompt: &str,
        images: &[&str],
    ) -> Result<Option<(Tensor, Tensor, u32, String)>> {
        let Some(vision_config) = self.model.vision_config() else {
            return Ok(None);
        };
        let preprocessed =
            qwen3_vl::preprocess_images(images, &vision_config, &self.infer_conf.device)?;
        let full_prompt = qwen3_vl::build_vision_prompt(prompt, preprocessed.num_placeholders);
        Ok(Some((
            preprocessed.pixel_values,
            preprocessed.image_grid_thw,
            vision_config.image_token_id,
            full_prompt,
        )))
    }

    /// 唯一的解码循环，流式产出文本块。
    ///
    /// `vision_data` 为 (pixel_values, image_grid_thw, image_token_id)，纯文本传 None。
    /// 生成结束后把完整回答写回上下文并清理 token 流缓冲，两条路径（流式/非流式）
    /// 行为因此完全一致。
    fn decode<'a>(
        &'a mut self,
        vision_data: Option<(Tensor, Tensor, u32)>,
    ) -> impl Stream<Item = Result<String>> + 'a {
        try_stream! {
            // 先清缓存再渲染 prompt：清缓存必须在编码之前，否则首 token 会带旧状态
            self.model.clr_kv_cache();

            let rendered = self.ctx.render()?;
            let mut ctx_tokens = self.str2tokens(&rendered)?;

            let vision: Option<VisionInput> = vision_data.map(|(pixel_values, image_grid_thw, token_id)| {
                VisionInput {
                    pixel_values,
                    image_grid_thw,
                    image_token_span: qwen3_vl::find_image_token_span(&ctx_tokens, token_id),
                }
            });

            if std::env::var_os("Q35_PERF").is_some() {
                tracing::info!(
                    "[q35-prefill] prompt_tokens={} msgs={}",
                    ctx_tokens.len(),
                    self.ctx.messages.len()
                );
            }

            let start = std::time::Instant::now();
            let ans_start_idx = ctx_tokens.len();
            let mut answer = String::with_capacity(1024);
            let mut decoded = 0usize;
            let mut decode_start: Option<std::time::Instant> = None;

            for index in 0..self.infer_conf.sample_len {
                let t0 = std::time::Instant::now();

                // 首个 token 走完整 prompt（可能含图像），之后每次只喂上一个 token
                let (input_pos, vision_ref, penalty_from) = if index == 0 {
                    (0, vision.as_ref(), None)
                } else {
                    if decode_start.is_none() {
                        decode_start = Some(std::time::Instant::now());
                    }
                    (ans_start_idx + index - 1, None, Some(ans_start_idx))
                };

                let next_token = self.gen_next_token(&ctx_tokens, input_pos, penalty_from, vision_ref)?;

                // 只记录 CPU 侧提交耗时，不含 GPU 排队；打开后每 token 多一次同步，仅用于诊断
                if std::env::var_os("Q35_PERF").is_some() && index < 200 {
                    tracing::info!(
                        "[q35-perf] idx={index} gen={:?} prompt={ans_start_idx}",
                        t0.elapsed()
                    );
                }

                ctx_tokens.push(next_token);
                decoded += 1;

                if let Some(t) = self.tos.next_token(next_token)? {
                    answer.push_str(&t);
                    yield t;
                }

                if self.is_stop_token(next_token) {
                    break;
                }
            }

            if let Some(ds) = decode_start {
                let secs = ds.elapsed().as_secs_f64();
                if secs > 0.0 {
                    info!(
                        "[q35-perf] decode: {decoded} tok in {secs:.3}s = {:.2} tok/s",
                        decoded as f64 / secs
                    );
                }
            }

            if let Some(t) = self.tos.decode_rest()? {
                answer.push_str(&t);
                yield t;
            }

            self.ctx.push_msg(&answer);
            self.tos.clear();

            let elapsed = start.elapsed().as_secs_f64();
            if elapsed > 0.0 {
                info!(
                    "speed: {:.2} token/s, total tokens: {}",
                    (ctx_tokens.len() - ans_start_idx) as f64 / elapsed,
                    ctx_tokens.len()
                );
            }
        }
    }

    /// 非流式入口：把 decode 跑到结束并收集完整回答
    fn generate_inner(
        &mut self,
        vision_data: Option<(Tensor, Tensor, u32)>,
    ) -> impl std::future::Future<Output = Result<String>> + '_ {
        async move {
            let stream = self.decode(vision_data);
            let mut stream = std::pin::pin!(stream);
            let mut answer = String::with_capacity(1024);
            while let Some(chunk) = futures_util::StreamExt::next(&mut stream).await {
                answer.push_str(&chunk?);
            }
            Ok(answer)
        }
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
                logits = repeat_penalty_inplace(
                    &logits,
                    self.infer_conf.repeat_penalty,
                    &ans_tokens[start_at..],
                )?;
            }
        }

        self.logits_processor.sample(&logits).map_err(Error::msg)
    }
}

/// 与 candle_transformers::utils::apply_repeat_penalty 数值等价，但避免整表回读。
///
/// 上游实现每次都把整个 logits（vocab≈248k×f32≈1MB）CPU↔GPU 搬两趟，
/// 单次约 18ms，占每 token 总耗时的两成以上。这里只把被惩罚的那几个
/// 位置的 logit 取下来（<= repeat_last_n 个标量），算完再写回，往返只有几 KB。
fn repeat_penalty_inplace(logits: &Tensor, penalty: f32, context: &[u32]) -> Result<Tensor> {
    if context.is_empty() || penalty == 1. {
        return Ok(logits.clone());
    }
    let device = logits.device();
    let vocab = logits.dim(0)?;

    // 去重并丢掉越界 id（保持首次出现顺序，结果与 HashSet 版本一致）
    let mut seen = std::collections::HashSet::new();
    let mut ids: Vec<u32> = Vec::with_capacity(context.len());
    for &t in context {
        if (t as usize) < vocab && seen.insert(t) {
            ids.push(t);
        }
    }
    if ids.is_empty() {
        return Ok(logits.clone());
    }

    // 只把被惩罚的位置取下来（<= repeat_last_n 个标量），算完用 index_add 写回
    let idx = Tensor::from_vec(ids.clone(), ids.len(), device)?;
    let picked = logits.index_select(&idx, 0)?.to_dtype(DType::F32)?;
    let picked = picked.to_vec1::<f32>()?;
    let adjusted: Vec<f32> = picked
        .iter()
        .map(|&v| if v >= 0. { v / penalty } else { v * penalty })
        .collect();

    // 差值写回：index_add(dim=0) 把 (n) 加到对应行上
    let delta: Vec<f32> = adjusted
        .iter()
        .zip(picked.iter())
        .map(|(n, o)| n - o)
        .collect();
    let delta = Tensor::from_vec(delta, ids.len(), device)?;
    Ok(logits.index_add(&idx, &delta, 0)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;
    use candle_transformers::utils::apply_repeat_penalty;
    use futures_util::{StreamExt, pin_mut};
    use std::io;
    use std::io::Write;

    /// repeat penalty 改写必须与 candle 上游实现数值一致
    #[test]
    fn test_repeat_penalty_matches_upstream() -> Result<()> {
        let dev = candle::Device::Cpu;
        let logits = candle::Tensor::new(&[2.0f32, -3.0, 0.5, -0.25, 1.0, -6.0, 0.0, 4.5], &dev)?;
        let ctx = [1u32, 1, 3, 5, 99, 2, 2]; // 含重复与越界 id
        let want = apply_repeat_penalty(&logits, 1.3, &ctx)?;
        let got = repeat_penalty_inplace(&logits, 1.3, &ctx)?;
        let diff = (&got - &want)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(diff < 1e-6, "与上游实现不一致: {diff}");
        Ok(())
    }

    #[tokio::test]
    #[ignore]
    async fn test_prompt() -> Result<()> {
        let mut pipe = ChatPipeline::with_default_config("qwen3_5.0_8b_q4").await?;

        let prompts = vec![
            "我是snake，你给我记住了",
            "还记得我是谁吗",
            "你是谁",
            "给我笑一笑",
        ];

        for prompt_str in prompts {
            pipe.push_user_message(prompt_str);
            let answer = pipe.chat_full().await?;
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
            let prompt_str = "介绍一下你自己".to_string();

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
