use crate::model::ModelInference;
use crate::model::VisionInput;
use crate::model::config::{InferenceConfig, ModelLoader};
use crate::model::registry::ModelRegistry;
use crate::utils::chat::ChatContext;
use crate::utils::sampling::Sampler;
use anyhow::{Error, Result};
use async_stream::try_stream;
use candle::{DType, Tensor};
use candle_examples::token_output_stream::TokenOutputStream;
use futures_core::stream::Stream;
use serde_json::Value;
use std::fs;
use std::pin::Pin;
use tracing::info;

use crate::model::qwen3_vl::{self, Qwen3VL};

/// 最近一次生成的真实耗时数据
///
/// 流式接口吐的是**解码后的字符串片段**，不是 token —— 一个片段可能包含多个
/// token，也可能只是一个多字节字符的一部分。用"收到几次 chunk"算 tok/s 会
/// 得到明显偏低的假数字，所以速率必须由生成循环自己统计后交出来。
#[derive(Debug, Clone, Copy, Default)]
pub struct DecodeStats {
    /// 生成的 token 数（不含 prefill）
    pub tokens: usize,
    /// 从开始到首 token 可以输出之间的耗时（prompt 编码 + 首 token）
    pub prefill_secs: f64,
    /// 首 token 之后逐 token 解码的耗时
    pub decode_secs: f64,
}

impl DecodeStats {
    /// 稳态 decode 速率（token/s）
    pub fn tok_per_sec(&self) -> f64 {
        if self.decode_secs > 0.0 {
            self.tokens as f64 / self.decode_secs
        } else {
            0.0
        }
    }
}

pub struct ChatPipeline {
    model: Box<dyn ModelInference>,
    tos: TokenOutputStream,
    sampler: Sampler,
    ctx: ChatContext,
    infer_conf: InferenceConfig,
    eos_token_id: u32,
    /// 额外需要停止的 token id（如 <|im_end|>，Qwen3.5 config 的 eos 是 <|endoftext|>
    /// 但 chat 模板实际以 <|im_end|> 结尾）
    stop_tokens: Vec<u32>,
    /// 最近一次生成的实测数据，供基准与诊断读取
    last_stats: Option<DecodeStats>,
    /// 最近一次 decode 的耗时拆分：(CPU 提交, 重复惩罚, 采样)，单位秒，累计值
    ///
    /// 这两段的比例决定优化方向：提交占大头说明是 launch-bound（该上 CUDA Graph），
    /// 等待占大头说明真在算（该融合算子或提算术强度）。没有这个拆分，
    /// 只能靠 nsys trace，而 nsys 在本机时而抓不到 CUDA 数据。
    last_split: Option<(f64, f64, f64)>,
}

impl ChatPipeline {
    pub async fn new(model_id: &str, config: InferenceConfig) -> Result<Self> {
        let registry = ModelRegistry::new()?;
        let hub_info = registry.get(model_id)?;
        let device = &config.device;
        let arch = model_id.split_once('.').map(|(a, _)| a).unwrap_or(model_id);

        let sampler = Sampler::new(config.seed, Some(config.temperature), config.top_p);

        let mut ctx = ChatContext::from_repo(&hub_info.tokenizer_repo).await?;
        // 诊断开关：Q35_THINKING=1 打开模板的思考分支（部分推理模型关掉思考会退化）
        if std::env::var("Q35_THINKING").is_ok() {
            ctx.enable_thinking = true;
        }

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
            sampler,
            ctx,
            infer_conf: config,
            eos_token_id,
            stop_tokens,
            last_stats: None,
            last_split: None,
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
    ///
    /// 对不支持清缓存的模型（Qwen3-MoE、Qwen3-VL）在已有上下文时会返回错误。
    pub fn clear_history(&mut self) -> Result<()> {
        self.ctx.clear_history();
        self.last_stats = None;
        self.model.reset_kv_cache()
    }

    /// 最近一次生成的实测数据（token 数、prefill 与 decode 耗时）
    ///
    /// 供基准工具读取，避免调用方拿"收到几个字符串片段"当 token 数。
    pub fn last_decode_stats(&self) -> Option<DecodeStats> {
        self.last_stats
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
            self.model.reset_kv_cache()?;

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
            self.last_split = None; // 每轮重新累计
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
                let stats = DecodeStats {
                    tokens: decoded,
                    prefill_secs: ds.duration_since(start).as_secs_f64(),
                    decode_secs: ds.elapsed().as_secs_f64(),
                };
                if stats.decode_secs > 0.0 {
                    info!(
                        "[q35-perf] decode: {} tok in {:.3}s = {:.2} tok/s",
                        stats.tokens,
                        stats.decode_secs,
                        stats.tok_per_sec()
                    );
                }
                self.last_stats = Some(stats);
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
        let ids = tokens.get_ids().to_vec();
        if std::env::var("Q35_DEBUG_IDS").is_ok() {
            eprintln!(
                "[ids] n={} first20={:?} last10={:?}",
                ids.len(),
                &ids[..ids.len().min(20)],
                &ids[ids.len().saturating_sub(10)..]
            );
            let toks = self.tos.tokenizer();
            for id in ids.iter().take(6).chain(ids.iter().rev().take(4)) {
                eprintln!("  id={id} -> {:?}", toks.id_to_token(*id));
            }
        }
        Ok(ids)
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

        let t0 = std::time::Instant::now();
        // 解码步（单 token）且模型支持时走 CUDA Graph：整步已捕获成一张图，重放即可。
        let mut logits = if ans_start_idx.is_some() && self.model.supports_graph() {
            self.model.forward_token(input_arr[0], idx_pos)?
        } else {
            let input = Tensor::new(input_arr, &self.infer_conf.device)?.unsqueeze(0)?;
            self.model.forward(&input, idx_pos, vision)?
        };
        let t_forward = std::time::Instant::now();
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

        let t_penalty = std::time::Instant::now();

        // 采样：贪心走设备端 argmax，只回读 4 字节；top-p 走 quickselect 选核，
        // 不再对 248k 词表做全量 argsort（那一步实测 16.5ms，是 decode 的最大单项）。
        let token = self.sampler.sample(&logits)?;
        if std::env::var("Q35_DEBUG_TOK").is_ok() {
            let top = logits.to_dtype(DType::F32)?.to_vec1::<f32>()?;
            let mut idx: Vec<usize> = (0..top.len()).collect();
            idx.sort_unstable_by(|&a, &b| top[b].total_cmp(&top[a]));
            let head: Vec<String> = idx
                .iter()
                .take(5)
                .map(|&i| format!("#{i}={:.3}", top[i]))
                .collect();
            let sum: f32 = top.iter().map(|v| v.max(0.0)).sum();
            eprintln!(
                "[tok] pick={token} top5[{}] sumexp={sum:.4}",
                head.join(" ")
            );
        }
        let t_end = std::time::Instant::now();

        let (mut submit, mut penalty, mut sample) = self.last_split.unwrap_or_default();
        submit += t_forward.duration_since(t0).as_secs_f64();
        penalty += t_penalty.duration_since(t_forward).as_secs_f64();
        sample += t_end.duration_since(t_penalty).as_secs_f64();
        self.last_split = Some((submit, penalty, sample));

        Ok(token)
    }

    /// 最近一轮 decode 的耗时拆分（秒）：(CPU 提交 forward, 重复惩罚, 采样/回读)
    ///
    /// 提交远大于后两段 = launch-bound，CUDA Graph 才有意义；否则该往算子内部走。
    pub fn last_split(&self) -> Option<(f64, f64, f64)> {
        self.last_split
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

    // 全程留在设备上：只把被惩罚的那几十个 logit 取出来算，算完用 index_add 写回。
    // 早先版本在这中间做了一次 to_vec1 —— 那是一次隐式 synchronize，CUDA Graph
    // 重放时它会把整张图的执行时间都记到「惩罚」这一段头上。
    let n = ids.len();
    let idx = Tensor::from_vec(ids, n, device)?;
    let picked = logits.index_select(&idx, 0)?.to_dtype(DType::F32)?;
    let is_pos = picked.ge(0f32)?.to_dtype(DType::F32)?;
    let is_neg = (&is_pos.neg()? + 1f64)?;
    debug_assert_eq!(is_neg.elem_count(), n);
    let scaled_down = (&picked / penalty as f64)?;
    let scaled_up = (&picked * penalty as f64)?;
    let adjusted = ((&scaled_down * &is_pos)? + (&scaled_up * &is_neg)?)?;
    let delta = (&adjusted - &picked)?.to_dtype(logits.dtype())?;
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
