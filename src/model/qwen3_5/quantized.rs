//! Qwen3.5 implementation with quantization support.
//!
//! Qwen3.5 is a hybrid architecture combining Gated DeltaNet (Linear Attention)
//! and Gated Attention (Full Softmax Attention).
//!
//! Based on the Qwen 3.5 architecture and implemented with quantized weights
//! for reduced memory usage and faster inference.
//!
use candle::quantized::{QTensor, gguf_file};
use candle::{D, DType, Device, Module, Result, Tensor};
use candle_nn::{Activation, Embedding, kv_cache::ConcatKvCache};
use candle_transformers::models::with_tracing::QMatMul;
use candle_transformers::quantized_nn::RmsNorm;
use candle_transformers::utils::repeat_kv;
use std::io::{Read, Seek};
use std::sync::Arc;
use std::time::Instant;

use super::graph::DecodeGraph;
use super::kv_buffer::FixedKvCache;

pub struct Gguf<R: Read + Seek> {
    ct: gguf_file::Content,
    reader: R,
    device: Device,
}

impl<R: Read + Seek> Gguf<R> {
    pub fn new(ct: gguf_file::Content, reader: R, device: Device) -> Self {
        Self { ct, reader, device }
    }

    pub fn qmatmul(&mut self, name: &str) -> Result<(QMatMul, usize)> {
        let ws = self.ct.tensor(&mut self.reader, name, &self.device)?;
        let out_dim = ws.shape().dims()[0];
        Ok((QMatMul::from_weights(ws.into())?, out_dim))
    }

    pub fn rms_norm(&mut self, name: &str, eps: f64) -> Result<RmsNorm> {
        let ws = self.ct.tensor(&mut self.reader, name, &self.device)?;
        RmsNorm::from_qtensor(ws, eps)
    }

    pub fn metadata(&self) -> &std::collections::HashMap<String, gguf_file::Value> {
        &self.ct.metadata
    }

    pub fn tensor(&mut self, name: &str) -> Result<QTensor> {
        self.ct.tensor(&mut self.reader, name, &self.device)
    }
}

#[derive(Debug, Clone)]
struct MlpWeights {
    gate_proj: QMatMul,
    up_proj: QMatMul,
    down_proj: QMatMul,
    act_fn: Activation,
}

impl MlpWeights {
    fn new<R: Read + Seek>(gg: &mut Gguf<R>, prefix: &str) -> Result<Self> {
        let (gate_proj, _) = gg.qmatmul(&format!("{prefix}.ffn_gate.weight"))?;
        let (up_proj, _) = gg.qmatmul(&format!("{prefix}.ffn_up.weight"))?;
        let (down_proj, _) = gg.qmatmul(&format!("{prefix}.ffn_down.weight"))?;
        let act_fn = Activation::Silu;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn,
        })
    }
}

impl Module for MlpWeights {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?.apply(&self.act_fn)?;
        let up = self.up_proj.forward(x)?;
        let gated = (gate * up)?;
        self.down_proj.forward(&gated)
    }
}

#[derive(Debug, Clone)]
pub struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
    /// 实际参与旋转的维度数。Qwen3.5 只转前 64 维（GGUF rope.dimension_count），
    /// 其余维度原样透传 —— 与 llama.cpp 的 ggml_rope_ext(n_rot=64) 对齐。
    rot_dim: usize,
}

impl RotaryEmbedding {
    pub fn new(
        dtype: DType,
        head_dim: usize,
        rot_dim: usize,
        max_position_embeddings: usize,
        rope_theta: f64,
        dev: &Device,
    ) -> Result<Self> {
        let dim = if rot_dim == 0 || rot_dim > head_dim {
            head_dim
        } else {
            rot_dim
        };
        let max_seq_len = max_position_embeddings;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(dtype)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(dtype)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
            rot_dim: dim,
        })
    }

    /// 对最后一维的前 rot_dim 维做旋转，其余维度原样拼回。
    fn rope_partial(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let head_dim = x.dim(D::Minus1)?;
        if self.rot_dim >= head_dim {
            return candle_nn::rotary_emb::rope(&x.contiguous()?, cos, sin);
        }
        let xr = candle_nn::rotary_emb::rope(
            &x.narrow(D::Minus1, 0, self.rot_dim)?.contiguous()?,
            cos,
            sin,
        )?;
        let xt = x.narrow(D::Minus1, self.rot_dim, head_dim - self.rot_dim)?;
        Tensor::cat(&[&xr, &xt], D::Minus1)
    }

    pub fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let (_, _, seq_len, _) = q.dims4()?;
        let cos = self.cos.narrow(0, offset, seq_len)?.to_dtype(q.dtype())?;
        let sin = self.sin.narrow(0, offset, seq_len)?.to_dtype(q.dtype())?;
        let q_embed = self.rope_partial(q, &cos, &sin)?;
        let k_embed = self.rope_partial(k, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }

    /// 图内用：位置由 **device** 张量给出（host offset 会被烘进图，replay 永远取同一段）。
    pub fn apply_dev(&self, q: &Tensor, k: &Tensor, pos_t: &Tensor) -> Result<(Tensor, Tensor)> {
        let (_, _, seq_len, _) = q.dims4()?;
        let offsets = Tensor::arange(0u32, seq_len as u32, pos_t.device())?;
        let idx = pos_t
            .to_dtype(DType::U32)?
            .reshape(())?
            .broadcast_add(&offsets)?;
        let cos = self.cos.index_select(&idx, 0)?.to_dtype(q.dtype())?;
        let sin = self.sin.index_select(&idx, 0)?.to_dtype(q.dtype())?;
        let q_embed = self.rope_partial(q, &cos, &sin)?;
        let k_embed = self.rope_partial(k, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

/// 一次前向的逐步上下文。
///
/// eager 与图内共用同一条前向代码，差别只有三处，全部收在这里：
/// - 位置从哪来：`offset`（host）还是 `pos_t`（device）
/// - KV 读多少：读到 `offset+len`（eager，形状可变）还是读整段 `cap`（图内，形状恒定）
/// - 因果掩码：prefill 用 host 构造的 `causal_mask`，图内 decode 用 device 条件构造的 `mask`
struct StepCtx<'a> {
    offset: usize,
    pos_t: &'a Tensor,
    kv_ids: &'a Tensor,
    mask: Option<&'a Tensor>,
    cap: Option<usize>,
}

/// 图内 decode 的因果掩码：位置 i 可读当且仅当 `i <= pos`。
///
/// 整段缓冲下 attention 会算到 cap 列，靠这个掩码把 pos 之后的位置压成 -inf。
/// 完全由 device 张量构造，捕获一次即可复用。
fn decode_mask(
    pos_t: &Tensor,
    cap: usize,
    dtype: DType,
    consts: Option<&GraphConsts>,
) -> Result<Tensor> {
    let dev = pos_t.device();
    // 走图时全部用捕获前建好的常量：图内不能出现 Tensor::arange / Tensor::full，
    // 它们是「CPU 建再 H2D」，H2D 节点指向的 host 缓冲在捕获结束时已释放。
    let (idx, neg, zero) = match consts {
        Some(c) => (c.mask_idx.clone(), c.mask_neg.clone(), c.mask_zero.clone()),
        None => (
            Tensor::arange(0u32, cap as u32, dev)?,
            Tensor::full(f32::NEG_INFINITY, (1, 1, 1, cap), dev)?.to_dtype(dtype)?,
            Tensor::zeros((1, 1, 1, cap), dtype, dev)?,
        ),
    };
    // candle 的比较算子不做广播（broadcast_le 不存在），把 pos 显式展成 [cap] 再比
    let pos = pos_t.to_dtype(DType::U32)?.reshape(())?.broadcast_as(cap)?;
    let keep = idx.le(&pos)?;
    let keep = keep.reshape((1, 1, 1, cap))?;
    Tensor::where_cond(&keep, &zero, &neg)
}

/// KV 存储的两种实现，用 `Q35_CONCAT_KV=1` 切换（默认固定缓冲）。
///
/// 保留 Concat 只为 A/B 对照：固定缓冲是 CUDA Graph 的前提（形状恒定），
/// 但它在 eager 路径上多了 scatter 与索引构造，值多少必须用同一个二进制、
/// 相邻两跑量出来，而不是跨 session 比历史数字。
#[derive(Debug, Clone)]
enum KvStore {
    Fixed(FixedKvCache),
    Concat(ConcatKvCache),
}

impl KvStore {
    fn fixed_cap(&self) -> Option<usize> {
        match self {
            KvStore::Fixed(c) => Some(c.max_seq()),
            KvStore::Concat(_) => None,
        }
    }

    fn reset(&mut self) {
        match self {
            KvStore::Fixed(c) => c.reset(),
            KvStore::Concat(c) => c.reset(),
        }
    }
}

/// 捕获期失败定位：记录最后进入的阶段名。
///
/// 只有 Q35_STAGE=1 时才真的写字符串（默认零开销），失败时由 stage_err 打印出来。
/// thread_local 是因为同一个进程里可能有多个模型/线程各跑各的前向。
mod stage {
    use std::cell::RefCell;
    thread_local! {
        static CUR: RefCell<String> = const { RefCell::new(String::new()) };
    }
    pub(super) fn enabled() -> bool {
        static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *CACHED.get_or_init(|| std::env::var("Q35_STAGE").is_ok_and(|v| v == "1"))
    }
    pub(super) fn set(args: std::fmt::Arguments<'_>) {
        if enabled() {
            CUR.with(|c| *c.borrow_mut() = args.to_string());
        }
    }
    pub(super) fn get() -> String {
        CUR.with(|c| c.borrow().clone())
    }
}

/// 标记当前阶段（各算子组入口各调一次）
fn mark_stage(args: std::fmt::Arguments<'_>) {
    stage::set(args)
}

/// 给错误补上「最后进入的阶段」——捕获失败时唯一能定位到算子群的信息
fn stage_err(e: candle::Error) -> candle::Error {
    candle::Error::Msg(format!("[stage={}] {e}", stage::get()))
}

/// 标记当前阶段：stage!("attn.rope")  `Q35_STAGE=1` 时才有开销
macro_rules! stage {
    ($($arg:tt)*) => { mark_stage(format_args!($($arg)*)) };
}

/// 诊断开关：跑到指定阶段就返回零张量（二分定位图内非法访问的算子）
fn graph_stop() -> Option<String> {
    static CACHED: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    CACHED
        .get_or_init(|| std::env::var("Q35_GRAPH_STOP").ok())
        .clone()
}

fn stop_hit(name: &str) -> bool {
    graph_stop().as_deref() == Some(name)
}

/// 诊断开关：只跑前 n 层（二分定位图内非法访问）
fn graph_layer_limit() -> Option<usize> {
    static CACHED: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("Q35_GRAPH_LAYERS")
            .ok()
            .and_then(|v| v.parse().ok())
    })
}

/// 诊断开关：捕获后不恢复 GDN 状态（验证「捕获区内的状态张量被释放」是否是非法访问的根因）
fn no_restore() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| std::env::var("Q35_GRAPH_NO_RESTORE").is_ok_and(|v| v == "1"))
}

/// decode 步是否走 CUDA Graph。默认开启（Q35_CUDA_GRAPH=0 关闭）。
///
/// 0.8B Q4 上 decode 21.3 → 87.5 tok/s，是当前最大的单项收益，
/// 因此默认打开；留关闭开关是为了排查「图路径与 eager 不一致」这类问题。
pub fn graph_enabled() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("Q35_CUDA_GRAPH")
            .map(|v| v != "0")
            .unwrap_or(true)
    })
}

fn graph_check_enabled() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| std::env::var("Q35_GRAPH_CHECK").is_ok_and(|v| v == "1"))
}

pub fn bucket_check_enabled() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| std::env::var("Q35_BUCKET_CHECK").is_ok_and(|v| v == "1"))
}

fn use_concat_kv() -> bool {
    std::env::var("Q35_CONCAT_KV").is_ok_and(|v| v == "1")
}

#[derive(Debug, Clone)]
struct AttentionWeights {
    q_proj: QMatMul,
    k_proj: QMatMul,
    v_proj: QMatMul,
    o_proj: QMatMul,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    kv_cache: KvStore,
}

impl AttentionWeights {
    #[allow(clippy::too_many_arguments)]
    fn new<R: Read + Seek>(
        gg: &mut Gguf<R>,
        _num_heads: usize,
        _num_kv_heads: usize,
        head_dim: usize,
        rms_norm_eps: f64,
        rotary_emb: Arc<RotaryEmbedding>,
        act_dtype: DType,
        kv_cap: usize,
        prefix: &str,
    ) -> Result<Self> {
        let (q_proj, q_out) = gg.qmatmul(&format!("{prefix}.attn_q.weight"))?;
        let (k_proj, k_out) = gg.qmatmul(&format!("{prefix}.attn_k.weight"))?;
        let (v_proj, _) = gg.qmatmul(&format!("{prefix}.attn_v.weight"))?;
        let (o_proj, _) = gg.qmatmul(&format!("{prefix}.attn_output.weight"))?;

        let num_heads = q_out / (head_dim * 2); // Q + Gate
        let num_kv_heads = k_out / head_dim;
        let num_kv_groups = num_heads / num_kv_heads;

        let q_norm = gg.rms_norm(&format!("{prefix}.attn_q_norm.weight"), rms_norm_eps)?;
        let k_norm = gg.rms_norm(&format!("{prefix}.attn_k_norm.weight"), rms_norm_eps)?;

        let kv_cache = if use_concat_kv() {
            KvStore::Concat(ConcatKvCache::new(2))
        } else {
            KvStore::Fixed(FixedKvCache::new(
                kv_cap,
                num_kv_heads,
                head_dim,
                act_dtype,
                &gg.device.clone(),
            )?)
        };

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            rotary_emb,
            kv_cache,
        })
    }

    fn forward(&mut self, x: &Tensor, ctx: &StepCtx) -> Result<Tensor> {
        let (b, l, _) = x.dims3()?;

        stage!("attn.qkv_proj");
        let q_out = self.q_proj.forward(x)?;
        let q_out = q_out.reshape((b, l, self.num_heads, self.head_dim * 2))?;

        let q = q_out.narrow(D::Minus1, 0, self.head_dim)?.transpose(1, 2)?;
        let gate = q_out.narrow(D::Minus1, self.head_dim, self.head_dim)?;

        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let k = k
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        stage!("attn.qk_norm");
        let q_flat = q.flatten(0, 2)?;
        let k_flat = k.flatten(0, 2)?;

        let q_flat = self.q_norm.forward(&q_flat)?;
        let k_flat = self.k_norm.forward(&k_flat)?;
        let q = q_flat.reshape((b, self.num_heads, l, self.head_dim))?;
        let k = k_flat.reshape((b, self.num_kv_heads, l, self.head_dim))?;

        stage!("attn.rope");
        let (q, k) = if ctx.cap.is_some() {
            self.rotary_emb.apply_dev(&q, &k, ctx.pos_t)?
        } else {
            self.rotary_emb.apply(&q, &k, ctx.offset)?
        };

        let (k, v) = match &mut self.kv_cache {
            KvStore::Fixed(cache) => {
                stage!("attn.kv_write");
                // 索引 kv_ids 由 ModelWeights 每步构造一次（含 device 位置），此处只做写。
                // decode(l==1) 走 reshape：转置+合并+连续化是三次算子，reshape 是零拷贝视图。
                let (k_row, v_row) = if l == 1 {
                    (
                        k.reshape((self.num_kv_heads, self.head_dim))?
                            .unsqueeze(0)?,
                        v.reshape((self.num_kv_heads, self.head_dim))?
                            .unsqueeze(0)?,
                    )
                } else {
                    (
                        k.transpose(1, 2)?.squeeze(0)?.contiguous()?,
                        v.transpose(1, 2)?.squeeze(0)?.contiguous()?,
                    )
                };
                cache.write_indexed(ctx.kv_ids, &k_row, &v_row)?;
                // 图内读整段（形状恒定），eager 只读到已写位置。
                stage!("attn.kv_read");
                let (k, v) = match ctx.cap {
                    Some(cap) => {
                        let (kf, vf) = cache.full();
                        if cap != cache.max_seq() {
                            candle::bail!("图内读整段要求 cap == 容量，cap={cap}")
                        }
                        (kf.clone(), vf.clone())
                    }
                    None => cache.narrow(ctx.offset + l)?,
                };
                (
                    k.unsqueeze(0)?.transpose(1, 2)?.contiguous()?, // [1, kv_head, len, d]
                    v.unsqueeze(0)?.transpose(1, 2)?.contiguous()?,
                )
            }
            KvStore::Concat(cache) => cache.append(&k, &v)?, // 已是 [1, kv_head, len, d]
        };

        stage!("attn.repeat_kv");
        let k = repeat_kv(k, self.num_kv_groups)?.contiguous()?;
        let v = repeat_kv(v, self.num_kv_groups)?.contiguous()?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        if let Some(m) = ctx.mask {
            let m_dtype = m.dtype();
            let scores_dtype = scores.dtype();
            let mask = if m_dtype != scores_dtype {
                m.to_dtype(scores_dtype)?
            } else {
                m.clone()
            };
            scores = scores.broadcast_add(&mask)?;
        }
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let ctx = probs.matmul(&v)?;

        let ctx = ctx.transpose(1, 2)?;
        // Apply sigmoid gate
        let gate_sigmoid =
            candle_nn::ops::sigmoid(&gate.to_dtype(DType::F32)?)?.to_dtype(ctx.dtype())?;
        let ctx = ctx.broadcast_mul(&gate_sigmoid)?;

        let reshaped_ctx = ctx.reshape((b, l, self.num_heads * self.head_dim))?;
        self.o_proj.forward(&reshaped_ctx)
    }

    fn clear_kv_cache(&mut self) {
        // FixedKvCache 每轮位置从 0 重新写起、读只到已写长度，故无需清零（见 kv_buffer::reset）
        self.kv_cache.reset();
    }
}

#[derive(Debug, Clone)]
struct GatedDeltaNetWeights {
    num_v_heads: usize,
    num_k_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    key_dim: usize,
    value_dim: usize,
    conv_dim: usize,
    conv_kernel_size: usize,

    in_proj_qkv: QMatMul,
    in_proj_z: QMatMul,
    in_proj_b: QMatMul,
    in_proj_a: QMatMul,
    out_proj: QMatMul,

    conv1d_weight: Tensor,
    dt_bias_f32: Tensor,
    neg_a_f32: Tensor,
    norm_weight: Tensor,
    norm_eps: f64,

    conv_state: Option<Tensor>,
    recurrent_state: Option<Tensor>,
    /// 持久显存：走图前一次性分配，图内原地更新（见 StateBuf）
    conv_storage: Option<StateBuf>,
    rec_storage: Option<StateBuf>,
}

impl GatedDeltaNetWeights {
    #[allow(clippy::too_many_arguments)]
    fn new<R: Read + Seek>(
        gg: &mut Gguf<R>,
        _hidden_size: usize,
        num_v_heads: usize,
        num_k_heads: usize,
        head_k_dim: usize,
        head_v_dim: usize,
        conv_kernel_size: usize,
        rms_norm_eps: f64,
        prefix: &str,
    ) -> Result<Self> {
        let key_dim = head_k_dim * num_k_heads;
        let value_dim = head_v_dim * num_v_heads;
        let conv_dim = key_dim * 2 + value_dim;

        let (in_proj_qkv, _) = gg.qmatmul(&format!("{prefix}.attn_qkv.weight"))?;
        let (in_proj_z, _) = gg.qmatmul(&format!("{prefix}.attn_gate.weight"))?;
        let (in_proj_b, _) = gg.qmatmul(&format!("{prefix}.ssm_beta.weight"))?;
        let (in_proj_a, _) = gg.qmatmul(&format!("{prefix}.ssm_alpha.weight"))?;
        let (out_proj, _) = gg.qmatmul(&format!("{prefix}.ssm_out.weight"))?;

        let conv1d_weight = gg
            .tensor(&format!("{prefix}.ssm_conv1d.weight"))?
            .dequantize(&gg.device)?
            .unsqueeze(1)?;

        let a_log = gg
            .tensor(&format!("{prefix}.ssm_a"))?
            .dequantize(&gg.device)?;
        let dt_bias = gg
            .tensor(&format!("{prefix}.ssm_dt.bias"))?
            .dequantize(&gg.device)?;

        let dt_bias_f32 = dt_bias.to_dtype(DType::F32)?;
        let neg_a_f32 = (&a_log.to_dtype(DType::F32)?.exp()? * -1.0)?;

        let norm_weight = gg
            .tensor(&format!("{prefix}.ssm_norm.weight"))?
            .dequantize(&gg.device)?;

        Ok(Self {
            num_v_heads,
            num_k_heads,
            head_k_dim,
            head_v_dim,
            key_dim,
            value_dim,
            conv_dim,
            conv_kernel_size,
            in_proj_qkv,
            in_proj_z,
            in_proj_b,
            in_proj_a,
            out_proj,
            conv1d_weight,
            dt_bias_f32,
            neg_a_f32,
            norm_weight,
            norm_eps: rms_norm_eps,
            conv_state: None,
            recurrent_state: None,
            conv_storage: None,
            rec_storage: None,
        })
    }

    fn l2_norm(xs: &Tensor) -> Result<Tensor> {
        // 用 rsqrt 等价式减少一次中间分配（broadcast_div 后接 sqrt 会多一份 (.., K, 1) 张量）
        let eps = 1e-6;
        let inv = (xs.sqr()?.sum_keepdim(D::Minus1)? + eps)?.sqrt()?.recip()?;
        xs.broadcast_mul(&inv)
    }

    fn rms_norm_gated(&self, x: &Tensor, g: &Tensor) -> Result<Tensor> {
        let x_dtype = x.dtype();
        stage!("gdn.norm.to_f32");
        let x_f32 = x.to_dtype(DType::F32)?;
        let g_f32 = g.to_dtype(DType::F32)?;

        stage!("gdn.norm.silu");
        let gate = g_f32.silu()?;
        stage!("gdn.norm.rms");
        let norm_x = (x_f32.sqr()?.mean_keepdim(D::Minus1)? + self.norm_eps)?.sqrt()?;
        stage!("gdn.norm.div");
        let x_normed = x_f32.broadcast_div(&norm_x)?;
        stage!("gdn.norm.mul");
        (x_normed * gate)?
            .broadcast_mul(&self.norm_weight.to_dtype(DType::F32)?)?
            .to_dtype(x_dtype)
    }

    fn torch_recurrent_gated_delta_rule(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        g: &Tensor,
        beta: &Tensor,
        initial_state: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let query = Self::l2_norm(query)?;
        let key = Self::l2_norm(key)?;

        let query = query.transpose(1, 2)?;
        let key = key.transpose(1, 2)?;
        let value = value.transpose(1, 2)?;
        let beta = beta.transpose(1, 2)?;
        let g = g.transpose(1, 2)?;

        let (batch_size, num_heads, seq_len, k_head_dim) = key.dims4()?;
        let v_head_dim = value.dim(3)?;

        // 注：上游 PR #3461 的分块 WY 扫描（chunked_scan）已引入，但实测在本机的
        // CPU storage 路径上 prefill 反而更慢（t=18 每层 13ms vs 逐 token 0.46ms），
        // 因此默认仍走逐 token 递推；需要时用 Q35_CHUNKED_PREFILL=1 显式启用。
        if chunked_prefill_enabled() && seq_len > 1 {
            let in_dtype = query.dtype();
            let mut state = match initial_state {
                Some(state) => state.to_dtype(DType::F32)?.contiguous()?,
                None => Tensor::zeros(
                    (batch_size, num_heads, k_head_dim, v_head_dim),
                    DType::F32,
                    query.device(),
                )?,
            };
            // 扫描模块用 [b, t, n_heads, d]，此处张量是 [b, n_heads, t, d]
            let q_c = query.to_dtype(DType::F32)?.transpose(1, 2)?.contiguous()?;
            let k_c = key.to_dtype(DType::F32)?.transpose(1, 2)?.contiguous()?;
            let v_c = value.to_dtype(DType::F32)?.transpose(1, 2)?.contiguous()?;
            let beta_c = beta.to_dtype(DType::F32)?.transpose(1, 2)?.contiguous()?;
            let g_c = g.to_dtype(DType::F32)?.transpose(1, 2)?.contiguous()?;
            let log_g = g_c.clamp(f32::MIN_POSITIVE as f64, f64::INFINITY)?.log()?;
            let out = crate::model::qwen3_5::chunked_scan::gated_delta_rule_chunked(
                &q_c, &k_c, &v_c, &log_g, &beta_c, &mut state,
            )?;
            return Ok((out.transpose(1, 2)?.to_dtype(in_dtype)?, state));
        }

        let scale = 1.0 / (query.dim(3)? as f64).sqrt();
        let query = (query * scale)?;

        let mut core_attn_out_vec = Vec::with_capacity(seq_len);

        let mut last_recurrent_state = match initial_state {
            Some(state) => state.to_dtype(DType::F32)?,
            None => Tensor::zeros(
                (batch_size, num_heads, k_head_dim, v_head_dim),
                DType::F32,
                query.device(),
            )?,
        };

        let query_f32 = query.to_dtype(DType::F32)?.contiguous()?;
        let key_f32 = key.to_dtype(DType::F32)?.contiguous()?;
        let value_f32 = value.to_dtype(DType::F32)?.contiguous()?;
        let g_exp = g.to_dtype(DType::F32)?.exp()?.unsqueeze(3)?.contiguous()?;
        let beta_f32 = beta.to_dtype(DType::F32)?.unsqueeze(3)?.contiguous()?;

        // 广播乘在 CUDA 上是慢路径：实测 (1,16,128,128) broadcast_mul (1,16,1,1)
        // 要 ~420µs，而同形状全量 mul 只要 ~20µs。这里把标量门控预先展开成
        // (B,H,1,V)，后续只用普通 mul —— 每层每 token 少 5 次广播。
        // 衰减 g 作用于整块状态 (B,H,K,V) → 展开成 (B,H,K,V)
        let expand_decay = |t: &Tensor| -> Result<Tensor> {
            let (b, h, s, _) = t.dims4()?;
            // (B,H,S,1) -> (B,H,S,1,1) -> 广播 (B,H,S,K,V) -> 合并成 (B,H*S*K, V)
            let t5 = t.reshape((b, h, s, 1, 1))?;
            t5.broadcast_as((b, h, s, k_head_dim, v_head_dim))?
                .reshape((b, h, s * k_head_dim, v_head_dim))?
                .contiguous()
        };
        // beta 作用于 delta (B,H,1,V) → 展开成 (B,H,1,V)
        let expand_beta = |t: &Tensor| -> Result<Tensor> {
            let (b, h, s, _) = t.dims4()?;
            t.broadcast_as((b, h, s, v_head_dim))?.contiguous()
        };

        // ── decode（seq_len == 1）：无循环、无 narrow，步骤数压到最少 ──────────
        if seq_len == 1 {
            let (b, h, kd) = (batch_size, num_heads, k_head_dim);
            let vd = v_head_dim;
            // 衰减：一次预展开 + 一次普通 mul（广播乘是 20 倍慢路径）
            let g_e = g_exp
                .reshape((b, h, 1, 1))?
                .broadcast_as((b, h, kd, vd))?
                .contiguous()?;
            last_recurrent_state = (&last_recurrent_state * &g_e)?;

            // kv_mem = k · S   —— k 已是 (B,H,1,K)
            let kv_mem = key_f32.matmul(&last_recurrent_state)?;

            // δ = (v − kv_mem) · β，β 预展开成 (B,H,1,V)
            let beta_e = beta_f32
                .reshape((b, h, 1, 1))?
                .broadcast_as((b, h, 1, vd))?
                .contiguous()?;
            let delta = ((value_f32 - &kv_mem)? * &beta_e)?;

            // S ← S + kᵀ·δ
            let delta_k = key_f32.transpose(2, 3)?.matmul(&delta)?;
            last_recurrent_state = (&last_recurrent_state + &delta_k)?;

            // out = q · S'
            let out = query_f32.matmul(&last_recurrent_state)?;
            return Ok((out.transpose(1, 2)?, last_recurrent_state));
        }

        for i in 0..seq_len {
            let q_t = query_f32.narrow(2, i, 1)?;
            let k_t = key_f32.narrow(2, i, 1)?;
            let v_t = value_f32.narrow(2, i, 1)?;
            let g_t = g_exp.narrow(2, i, 1)?;
            let beta_t = beta_f32.narrow(2, i, 1)?;

            let g_t = expand_decay(&g_t)?;
            last_recurrent_state = (&last_recurrent_state * &g_t)?;

            let kv_mem = k_t.matmul(&last_recurrent_state)?;
            let beta_t = expand_beta(&beta_t)?;
            let delta = ((&v_t - &kv_mem)? * &beta_t)?;

            let delta_k = k_t.transpose(2, 3)?.matmul(&delta)?;
            last_recurrent_state = (&last_recurrent_state + &delta_k)?;

            let out_t = q_t.matmul(&last_recurrent_state)?;
            core_attn_out_vec.push(out_t);
        }

        let core_attn_out = Tensor::cat(&core_attn_out_vec, 2)?;
        let core_attn_out = core_attn_out.transpose(1, 2)?;
        Ok((core_attn_out, last_recurrent_state))
    }

    /// 走图前分配持久状态显存。形状取 decode 步的定值（batch=1、seq=1）：
    /// prefill 与 decode 写回的状态元素数相同，共用一块缓冲即可。
    fn prepare_state_storage(
        &mut self,
        batch: usize,
        act_dtype: DType,
        dev: &Device,
    ) -> Result<()> {
        let pad = self.conv_kernel_size - 1;
        self.conv_storage = Some(StateBuf::zeros(
            &[batch, self.conv_dim, pad],
            act_dtype,
            dev,
        )?);
        // 递推状态在 delta rule 内部一律走 F32
        self.rec_storage = Some(StateBuf::zeros(
            &[batch, self.num_v_heads, self.head_v_dim, self.head_k_dim],
            DType::F32,
            dev,
        )?);
        Ok(())
    }

    fn forward(&mut self, hidden_states: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len, _) = hidden_states.dims3()?;
        let initial_dtype = hidden_states.dtype();

        stage!("gdn.in_proj");
        let mixed_qkv = self.in_proj_qkv.forward(hidden_states)?.transpose(1, 2)?;
        if stop_hit("gdn.in_proj") {
            return Tensor::zeros_like(hidden_states);
        }

        let z = self.in_proj_z.forward(hidden_states)?;
        let z = z.reshape((batch_size, seq_len, (), self.head_v_dim))?;

        let b = self.in_proj_b.forward(hidden_states)?;
        let a = self.in_proj_a.forward(hidden_states)?;

        let use_precomputed_states = self.conv_state.is_some() && seq_len == 1;

        stage!("gdn.conv");
        let t_conv = Instant::now();
        let mixed_qkv = if use_precomputed_states {
            let conv_state = self.conv_state.clone().unwrap();
            let conv_state_data = Tensor::cat(&[&conv_state, &mixed_qkv], 2)?;
            let next = conv_state_data.narrow(2, 1, self.conv_kernel_size - 1)?;
            // 图内不能"算完换张量"：换了就等于把状态丢在已被回收的显存里
            store_state(self.conv_storage.as_ref(), &mut self.conv_state, &next)?;
            let out = depthwise_conv1d(&conv_state_data, &self.conv1d_weight, 0)?;
            candle_nn::ops::silu(&out)?
        } else {
            let pad = self.conv_kernel_size - 1;
            let padding = Tensor::zeros(
                (batch_size, self.conv_dim, pad),
                mixed_qkv.dtype(),
                mixed_qkv.device(),
            )?;
            let padded_qkv = Tensor::cat(&[&padding, &mixed_qkv], 2)?;
            let next = padded_qkv.narrow(2, seq_len, pad)?;
            store_state(self.conv_storage.as_ref(), &mut self.conv_state, &next)?;
            let out = depthwise_conv1d(&padded_qkv, &self.conv1d_weight, 0)?;
            candle_nn::ops::silu(&out)?
        };

        let dt_conv = t_conv.elapsed();
        if stop_hit("gdn.conv") {
            return Tensor::zeros_like(hidden_states);
        }
        let mixed_qkv = mixed_qkv.transpose(1, 2)?;

        let q = mixed_qkv.narrow(D::Minus1, 0, self.key_dim)?;
        let k = mixed_qkv.narrow(D::Minus1, self.key_dim, self.key_dim)?;
        let v = mixed_qkv.narrow(D::Minus1, self.key_dim * 2, self.value_dim)?;

        let q = q.reshape((batch_size, seq_len, (), self.head_k_dim))?;
        let k = k.reshape((batch_size, seq_len, (), self.head_k_dim))?;
        let v = v.reshape((batch_size, seq_len, (), self.head_v_dim))?;

        stage!("gdn.gate");
        let beta = candle_nn::ops::sigmoid(&b)?;
        let g = {
            let a_f32 = a.to_dtype(DType::F32)?;
            let a_plus_dt = a_f32.broadcast_add(&self.dt_bias_f32)?;
            let softplus = (a_plus_dt.exp()? + 1.0)?.log()?;
            self.neg_a_f32.broadcast_mul(&softplus)?
        };

        let repeat_n = self.num_v_heads / self.num_k_heads;
        // 与 llama.cpp qwen35.cpp:452-453 的 ggml_repeat_4d 对齐：v_head h 用 k_head
        // h % num_k_heads（整块平铺），不是 h / repeat_n（逐头交错）。两者只在
        // num_v_heads != num_k_heads 时不同：0.8B/4B 相等（恒等），27B 是 48 vs 16。
        let q = tile_heads(&q, repeat_n, 2)?;
        let k = tile_heads(&k, repeat_n, 2)?;

        // 先 clone 出句柄再交给递推：写回时要用 &mut self.recurrent_state
        let initial_state = if use_precomputed_states {
            self.recurrent_state.clone()
        } else {
            None
        };

        if stop_hit("gdn.gate") {
            return Tensor::zeros_like(hidden_states);
        }

        stage!("gdn.delta_rule");
        let t_dr = Instant::now();
        let (core_attn_out, new_state) =
            self.torch_recurrent_gated_delta_rule(&q, &k, &v, &g, &beta, initial_state.as_ref())?;
        let dt_dr = t_dr.elapsed();
        store_state(
            self.rec_storage.as_ref(),
            &mut self.recurrent_state,
            &new_state,
        )?;

        if stop_hit("gdn.delta_rule") {
            return Tensor::zeros_like(hidden_states);
        }
        let core_attn_out = core_attn_out.to_dtype(initial_dtype)?;
        let core_attn_out =
            core_attn_out.reshape((batch_size, seq_len, self.num_v_heads, self.head_v_dim))?;
        let core_attn_out = core_attn_out.reshape(((), self.head_v_dim))?;
        let z_flat = z.reshape(((), self.head_v_dim))?;
        stage!("gdn.norm");
        let core_attn_out = self.rms_norm_gated(&core_attn_out, &z_flat)?;
        if stop_hit("gdn.norm") {
            return Tensor::zeros_like(hidden_states);
        }
        stage!("gdn.reshape_out");
        let core_attn_out =
            core_attn_out.reshape((batch_size, seq_len, self.num_v_heads * self.head_v_dim))?;

        stage!("gdn.out_proj");
        let out = self.out_proj.forward(&core_attn_out);
        // 每 token 查一次 env 在热路径上不便宜，用进程级缓存
        if seq_len <= 2 && perf_enabled() {
            tracing::info!("[q35-gdn] conv={dt_conv:.1?} delta_rule={dt_dr:.1?}");
        }
        out
    }

    fn clear_kv_cache(&mut self) {
        self.conv_state = None;
        self.recurrent_state = None;
    }
}

fn chunked_prefill_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("Q35_CHUNKED_PREFILL").is_some())
}

fn perf_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("Q35_PERF").is_some())
}

/// 高效 depthwise conv1d（每通道独立卷积，kernel_size 小、通道多时远快于 candle 的 groups 分解）
/// x: (b, c, l_in), weight: (c, 1, k) → 输出 (b, c, l_out)
fn depthwise_conv1d(x: &Tensor, weight: &Tensor, padding: usize) -> Result<Tensor> {
    let (b, c, _) = x.dims3()?;
    let k = weight.dim(2)?;

    // 用 k 次移位切片 + 乘加累积，替代 candle 的 groups 分解
    let x_padded = if padding > 0 {
        let zeros = Tensor::zeros((b, c, padding), x.dtype(), x.device())?;
        Tensor::cat(&[&zeros, x, &zeros], 2)?
    } else {
        x.clone()
    };
    let l_in = x_padded.dim(2)?;
    if l_in < k {
        candle::bail!("depthwise_conv1d: 输入长度 {l_in} 小于卷积核 {k}");
    }
    let l_out = l_in - k + 1;
    let w = weight.reshape((c, k))?.to_dtype(x.dtype())?.contiguous()?;

    // decode（L==1）走单次「行内点积」：x (b,c,k) ⊙ w (c,k) 再按 k 求和。
    // 逐位置 multiply+add 的写法每次小算子约 30µs 固定开销，k=4 时要 10 次 ≈ 380µs；
    // 合并成一次后实测 ~1µs（同一张量、数值误差 <1e-6）。
    if l_out == 1 {
        let xs = x_padded.reshape((b, c, k))?;
        let w = w.reshape((1, c, k))?;
        return (&xs * &w)?.sum(2)?.unsqueeze(2);
    }

    // prefill：逐位置累加（L>1 时窗口重叠，仍按位移累加）
    let mut acc = x_padded
        .narrow(2, 0, l_out)?
        .broadcast_mul(&w.narrow(1, 0, 1)?.reshape((1, c, 1))?)?;
    for i in 1..k {
        let xi = x_padded.narrow(2, i, l_out)?;
        let wi = w.narrow(1, i, 1)?.reshape((1, c, 1))?;
        acc = (&acc + &xi.broadcast_mul(&wi)?)?;
    }
    Ok(acc)
}

/// 把 dim 上的 n 个头平铺成 repeats*n 个：第 h 个输出头取第 h % n 个输入头。
fn tile_heads(t: &Tensor, repeats: usize, dim: usize) -> Result<Tensor> {
    if repeats == 1 {
        return Ok(t.clone());
    }
    let mut dims = t.dims().to_vec();
    dims[dim] *= repeats;
    let final_dims = dims.clone();
    let t = t.unsqueeze(dim)?;
    let mut expand_dims = t.dims().to_vec();
    expand_dims[dim] = repeats;
    let expanded = t.expand(expand_dims.as_slice())?;
    expanded.reshape(final_dims.as_slice())
}

#[derive(Debug, Clone)]
enum TokenMixer {
    FullAttention(AttentionWeights),
    LinearAttention(GatedDeltaNetWeights),
}

#[derive(Debug, Clone)]
struct LayerWeights {
    token_mixer: TokenMixer,
    mlp: MlpWeights,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl LayerWeights {
    #[allow(clippy::too_many_arguments)]
    fn new<R: Read + Seek>(
        gg: &mut Gguf<R>,
        layer_idx: usize,
        num_attention_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rms_norm_eps: f64,
        rotary: Arc<RotaryEmbedding>,
        act_dtype: DType,
        kv_cap: usize,
        full_attention_interval: usize,
        // Linear attention specifics
        linear_num_key_heads: usize,
        linear_num_value_heads: usize,
        linear_key_head_dim: usize,
        linear_value_head_dim: usize,
        linear_conv_kernel_dim: usize,
        hidden_size: usize,
    ) -> Result<Self> {
        let prefix = format!("blk.{layer_idx}");
        let is_full_attention = (layer_idx + 1) % full_attention_interval == 0;

        let token_mixer = if is_full_attention {
            TokenMixer::FullAttention(AttentionWeights::new(
                gg,
                num_attention_heads,
                num_kv_heads,
                head_dim,
                rms_norm_eps,
                rotary,
                act_dtype,
                kv_cap,
                &prefix,
            )?)
        } else {
            TokenMixer::LinearAttention(GatedDeltaNetWeights::new(
                gg,
                hidden_size,
                linear_num_value_heads,
                linear_num_key_heads,
                linear_key_head_dim,
                linear_value_head_dim,
                linear_conv_kernel_dim,
                rms_norm_eps,
                &prefix,
            )?)
        };

        let mlp = MlpWeights::new(gg, &prefix)?;
        let input_layernorm = gg.rms_norm(&format!("{prefix}.attn_norm.weight"), rms_norm_eps)?;
        let post_attention_layernorm = gg.rms_norm(
            &format!("{prefix}.post_attention_norm.weight"),
            rms_norm_eps,
        )?;

        Ok(Self {
            token_mixer,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(
        &mut self,
        x: &Tensor,
        ctx: &StepCtx,
        layer_kind: &mut &'static str,
        t_mixer: Option<&mut std::time::Duration>,
        _t_mlp: Option<&mut std::time::Duration>,
    ) -> Result<Tensor> {
        let residual = x;
        stage!("layer.input_norm");
        let x = self.input_layernorm.forward(x)?;
        let t0 = std::time::Instant::now();
        let x = match &mut self.token_mixer {
            TokenMixer::FullAttention(attn) => {
                *layer_kind = "full";
                attn.forward(&x, ctx)?
            }
            TokenMixer::LinearAttention(attn) => {
                *layer_kind = "gdn";
                attn.forward(&x)?
            }
        };
        if let Some(d) = t_mixer {
            *d += t0.elapsed();
        }
        stage!("layer.residual1");
        let x = (residual + x)?;

        let residual = &x;
        stage!("layer.post_norm");
        let x = self.post_attention_layernorm.forward(&x)?;
        stage!("layer.mlp");
        let x = x.apply(&self.mlp)?;
        x + residual
    }

    fn clear_kv_cache(&mut self) {
        match &mut self.token_mixer {
            TokenMixer::FullAttention(attn) => attn.clear_kv_cache(),
            TokenMixer::LinearAttention(attn) => attn.clear_kv_cache(),
        }
    }

    /// 快照必须是深拷贝：状态现在写在持久缓冲里，clone 只共享存储，
    /// 后续原地更新会把快照一起改掉。
    fn states(&self) -> Result<(Option<Tensor>, Option<Tensor>)> {
        match &self.token_mixer {
            TokenMixer::LinearAttention(a) => {
                let conv = a.conv_state.as_ref().map(|t| t.copy()).transpose()?;
                let rec = a.recurrent_state.as_ref().map(|t| t.copy()).transpose()?;
                Ok((conv, rec))
            }
            TokenMixer::FullAttention(_) => Ok((None, None)),
        }
    }

    fn set_states(&mut self, s: &(Option<Tensor>, Option<Tensor>)) -> Result<()> {
        if let TokenMixer::LinearAttention(a) = &mut self.token_mixer {
            match &s.0 {
                Some(t) => store_state(a.conv_storage.as_ref(), &mut a.conv_state, t)?,
                None => a.conv_state = None,
            }
            match &s.1 {
                Some(t) => store_state(a.rec_storage.as_ref(), &mut a.recurrent_state, t)?,
                None => a.recurrent_state = None,
            }
        }
        Ok(())
    }

    fn uses_fixed_kv(&self) -> bool {
        matches!(
            &self.token_mixer,
            TokenMixer::FullAttention(a) if a.kv_cache.fixed_cap().is_some()
        )
    }
}

/// 图内常量：全部在捕获**之前**建好，图里只引用、不新建。
///
/// 由来（实测）：candle 的 `Tensor::arange` / `Tensor::full` 是「CPU 建 Vec 再 H2D」，
/// 捕获进图后 H2D 节点指向的 host 缓冲在捕获结束时就被释放，replay 读到垃圾值；
/// 垃圾值当索引用时直接越界（compute-sanitizer 报 badd_u32 读非法地址）。
/// 所以凡是「与 token 无关的常量张量」一律在这里一次性建好。
#[derive(Debug, Clone)]
struct GraphConsts {
    /// [cap] u32，0..cap
    mask_idx: Tensor,
    /// [1, 1, 1, cap]，-inf
    mask_neg: Tensor,
    /// [1, 1, 1, cap]，0
    mask_zero: Tensor,
    /// [1] u32，固定 [0]：图内只支持单 token，seq 恒为 1
    offsets: Tensor,
}

impl GraphConsts {
    fn new(cap: usize, dtype: DType, dev: &Device) -> Result<Self> {
        Ok(Self {
            mask_idx: Tensor::arange(0u32, cap as u32, dev)?,
            mask_neg: Tensor::full(f32::NEG_INFINITY, (1, 1, 1, cap), dev)?.to_dtype(dtype)?,
            mask_zero: Tensor::zeros((1, 1, 1, cap), dtype, dev)?,
            offsets: Tensor::arange(0u32, 1u32, dev)?,
        })
    }
}

/// 持久状态缓冲：模型生命周期内地址不变，图内只能原地写。
///
/// 由来（实测）：实例化图时用了 AUTO_FREE_ON_LAUNCH（不用它第二次 cuGraphLaunch 就
/// 报 CUDA_ERROR_INVALID_VALUE），代价是每次 replay 结束后，图内由 alloc 节点分配的
/// 显存全部回收。而 candle 的状态更新习惯是"算出新张量再赋给字段"，字段里存的是
/// **捕获那一刻**的地址 —— replay 时那张新张量在另一块显存上，字段却还指着旧的，
/// 现象就是状态每步丢失、生成退化成反复吐同几个字。
/// 所以凡是要跨 replay 存活的张量（GDN 的 conv/recurrent 状态、logits）都在捕获前
/// 分配成固定缓冲，图内原地写，字段只指向这块缓冲。
#[derive(Debug, Clone)]
struct StateBuf {
    /// 逻辑形状：读取时 reshape 回来
    shape: Vec<usize>,
    /// [n] 展平后的存储
    data: Tensor,
    /// [rows] u32，与 data 的二维视图配合做原地写
    rows_idx: Tensor,
    /// data 二维视图的列数（= 写入并行度，见 store 的说明）
    cols: usize,
    /// 不能整除时的退路：[n] u32
    flat_idx: Tensor,
}

impl StateBuf {
    fn zeros(shape: &[usize], dtype: DType, dev: &Device) -> Result<Self> {
        let n: usize = shape.iter().product();
        // candle 的 CUDA scatter kernel 只按 numel = left_size * right_size 开线程
        // （candle-kernels/src/indexing.cu 的 scatter()）。沿 dim0 散写整块时
        // numel 就等于 right_size，right_size=1 时**整块由一个线程串行搬**
        // —— 实测 262144 元素要 29.7ms，一 token 37 次原地写就把单步拖到 700ms 以上。
        // 因此把缓冲看成 [rows, cols] 沿 dim0 散写，让 numel 等于 cols：
        // 并行度就是 cols，取 ≤4096 且整除 n 的最大值即可（262144→4096、18432→2048）。
        let cols = (1..=4096).rev().find(|k| n % k == 0).unwrap_or(1);
        let rows = n / cols;
        // candle 要求 indexes 与 source 形状一致（scatter_checks），所以把行号广播成 [rows, cols]
        let rows_idx = Tensor::arange(0u32, rows as u32, dev)?
            .reshape((rows, 1))?
            .broadcast_as((rows, cols))?
            .contiguous()?;
        Ok(Self {
            shape: shape.to_vec(),
            data: Tensor::zeros(n, dtype, dev)?,
            rows_idx,
            cols,
            flat_idx: Tensor::arange(0u32, n as u32, dev)?,
        })
    }

    /// 逻辑形状的视图（同一块存储，地址不变）
    fn view(&self) -> Result<Tensor> {
        self.data.reshape(self.shape.as_slice())
    }

    fn len(&self) -> usize {
        self.data.elem_count()
    }

    fn store(&self, src: &Tensor) -> Result<()> {
        let src = src
            .flatten_all()?
            .to_dtype(self.data.dtype())?
            .contiguous()?;
        if src.elem_count() != self.len() {
            candle::bail!(
                "持久缓冲元素数不符: 缓冲 {} 来源 {}",
                self.len(),
                src.elem_count()
            )
        }
        if self.cols == 1 {
            return self.data.scatter_set(&self.flat_idx, &src, 0);
        }
        let rows = self.len() / self.cols;
        // reshape 是视图，散写落在同一块存储上
        let dst = self.data.reshape((rows, self.cols))?;
        let src = src.reshape((rows, self.cols))?;
        dst.scatter_set(&self.rows_idx, &src, 0)
    }
}

/// 把状态写回槽位：有持久缓冲就原地写（图内必须），否则退回"算完换张量"。
fn store_state(storage: Option<&StateBuf>, slot: &mut Option<Tensor>, src: &Tensor) -> Result<()> {
    match storage {
        Some(sb) if sb.len() == src.elem_count() => {
            sb.store(src)?;
            *slot = Some(sb.view()?);
        }
        _ => *slot = Some(src.clone()),
    }
    Ok(())
}

#[derive(Debug, Clone)]
pub struct ModelWeights {
    embed_tokens: Embedding,
    layers: Vec<LayerWeights>,
    norm: RmsNorm,
    lm_head: QMatMul,
    device: Device,
    dtype: DType,
    /// (num_kv_heads, head_dim)：所有全注意力层一致，用于每步构造一份写入索引
    kv_shape: (usize, usize),
    /// 固定 KV 缓冲的容量（图内读整段时用，等于 FixedKvCache::max_seq）
    kv_cap: usize,
    /// 首次用时懒捕获的 decode 步图（Q35_CUDA_GRAPH=1 才启用）
    graph: Option<Arc<DecodeGraph>>,
    /// 图内常量（捕获前建好），未走图时为 None
    graph_consts: Option<GraphConsts>,
    /// logits 的持久缓冲：图内写、replay 后读（AUTO_FREE 会回收图内分配的 logits）
    logits_buf: Option<StateBuf>,
    vocab_size: usize,
}

impl ModelWeights {
    /// 计算所在设备。CUDA Graph 只在 CUDA 上有意义，supports_graph 据它判断。
    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn from_gguf<R: Read + Seek>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        let mut gg = Gguf::new(ct, reader, device.clone());
        let md_get = |s: &str| match gg.metadata().get(s) {
            Some(v) => Ok(v),
            None => {
                let s35 = s.replace("qwen3.", "qwen35.");
                match gg.metadata().get(&s35) {
                    Some(v) => Ok(v),
                    None => candle::bail!("cannot find {s} or {s35} in metadata"),
                }
            }
        };

        let num_attention_heads = md_get("qwen3.attention.head_count")?.to_u32()? as usize;
        let num_kv_heads = md_get("qwen3.attention.head_count_kv")?.to_u32()? as usize;
        let key_length = md_get("qwen3.attention.key_length")?.to_u32()? as usize;

        let head_dim = key_length;
        // block_count 把 MTP（multi-token prediction）头也算成一层：Qwen3.8-27B 报 65，
        // 但 blk.64 只有 attn_* + nextn.*，真正的 transformer 层是 0..63。
        // 少减这一层会在最后一步去找 blk.64.attn_qkv.weight 然后报 "cannot find tensor info"。
        let block_count = md_get("qwen3.block_count")?.to_u32()? as usize;
        let mtp_layers = md_get("qwen3.nextn_predict_layers")
            .and_then(|v| v.to_u32())
            .unwrap_or(0) as usize;
        let num_layers = block_count.saturating_sub(mtp_layers);
        let hidden_size = md_get("qwen3.embedding_length")?.to_u32()? as usize;
        let max_position_embeddings = md_get("qwen3.context_length")?.to_u32()? as usize;
        let rms_norm_eps = md_get("qwen3.attention.layer_norm_rms_epsilon")?.to_f32()? as f64;
        let rope_freq_base = md_get("qwen3.rope.freq_base")?.to_f32()? as f64;
        // Qwen3.5 用部分旋转：head_dim=256 但只转前 rope_dim 维（27B/0.8B 都是 64，
        // GGUF 键 rope.dimension_count）。rope.dimension_sections=[11,11,10] 只用于
        // 选择位置来源（t/h/w），纯文本时三个位置相同，等价于标准 64 维旋转。
        let rope_dim = md_get("qwen3.rope.dimension_count")
            .and_then(|v| v.to_u32())
            .map(|v| v as usize)
            .unwrap_or(head_dim);

        let full_attention_interval = md_get("qwen3.full_attention_interval")?.to_u32()? as usize;

        // Linear attention specifics
        let linear_num_key_heads = md_get("qwen3.ssm.group_count")?.to_u32()? as usize;
        let ssm_inner_size = md_get("qwen3.ssm.inner_size")?.to_u32()? as usize;
        let linear_key_head_dim = md_get("qwen3.ssm.state_size")?.to_u32()? as usize;
        let linear_value_head_dim = linear_key_head_dim;
        let linear_num_value_heads = ssm_inner_size / linear_value_head_dim;
        let linear_conv_kernel_dim = md_get("qwen3.ssm.conv_kernel")?.to_u32()? as usize;

        let dtype = match gg.metadata().get("general.dtype") {
            Some(v) => match v.to_u32() {
                Ok(0) => DType::F32,
                Ok(1) => DType::F16,
                _ => DType::F16,
            },
            None => DType::F16,
        };

        // KV 缓冲容量：固定分配才有恒定形状。默认取 4096 或模型上下文长度中较小者；
        // 6 层 × 4096 × kv_head × head_dim × 2 × 2 字节，0.8B 上约 48 MB。
        let kv_cap = std::env::var("Q35_KV_CAP")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(4096)
            .min(max_position_embeddings);

        // 词表嵌入按 F32 存要 vocab × hidden × 4 字节：0.8B 上是 1.0 GB，
        // 27B（248320 × 5120）上是 5.1 GB —— 后者直接把 16 GB 卡吃穿。
        // Q35_EMBED_DTYPE=f16 让它只占一半，查表后补一次 to_dtype 转回 F32，
        // 下游所有计算精度不变，只损失一次查表的舍入。
        let embed_dtype = match std::env::var("Q35_EMBED_DTYPE").as_deref() {
            Ok("f16") | Ok("F16") => DType::F16,
            Ok("bf16") | Ok("BF16") => DType::BF16,
            _ => DType::F32,
        };
        let embed_tensor = gg.tensor("token_embd.weight")?;
        let embed_tokens = Embedding::new(
            embed_tensor.dequantize(device)?.to_dtype(embed_dtype)?,
            hidden_size,
        );

        // 激活流统一 F32：KV 缓冲、conv/recurrent 状态的 dtype 都跟着它走。
        // 与嵌入的存储精度解耦，嵌入存半精度时由 forward_inner 补一次转换。
        let act_dtype = DType::F32;

        let rotary = Arc::new(RotaryEmbedding::new(
            dtype,
            head_dim,
            rope_dim,
            max_position_embeddings,
            rope_freq_base,
            device,
        )?);
        eprintln!("[load] rope: head_dim={head_dim} rot_dim={rope_dim}");

        eprintln!("[load] metadata ok, embed {embed_dtype:?}, layers={num_layers}");
        let t_layers = std::time::Instant::now();
        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            if i % 8 == 0 {
                eprintln!(
                    "[load] layer {i}/{num_layers} {:.1}s",
                    t_layers.elapsed().as_secs_f32()
                );
            }
            layers.push(LayerWeights::new(
                &mut gg,
                i,
                num_attention_heads,
                num_kv_heads,
                head_dim,
                rms_norm_eps,
                rotary.clone(),
                act_dtype,
                kv_cap,
                full_attention_interval,
                linear_num_key_heads,
                linear_num_value_heads,
                linear_key_head_dim,
                linear_value_head_dim,
                linear_conv_kernel_dim,
                hidden_size,
            )?);
        }

        let norm = gg.rms_norm("output_norm.weight", rms_norm_eps)?;
        let lm_head_tensor = match gg.tensor("output.weight") {
            Ok(tensor) => tensor,
            Err(_) => gg.tensor("token_embd.weight")?,
        };
        let vocab_size = lm_head_tensor.shape().dims()[0];
        let lm_head = QMatMul::from_weights(lm_head_tensor.into())?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            device: device.clone(),
            kv_shape: (num_kv_heads, head_dim),
            kv_cap,
            graph: None,
            graph_consts: None,
            logits_buf: None,
            vocab_size,
            dtype,
        })
    }

    fn causal_mask(&self, b: usize, tgt: usize, offset: usize) -> Result<Tensor> {
        let minf = f32::NEG_INFINITY;
        let mask: Vec<_> = (0..tgt)
            .flat_map(|i| (0..(tgt + offset)).map(move |j| if j <= i + offset { 0. } else { minf }))
            .collect();
        Tensor::from_slice(&mask, (b, 1, tgt, tgt + offset), &self.device)?.to_dtype(self.dtype)
    }

    /// eager 前向：位置来自 host 整数 offset，因果掩码在 host 上构造。
    pub fn forward(&mut self, input: &Tensor, offset: usize) -> Result<Tensor> {
        let (b, l) = input.dims2()?;
        let pos_t = Tensor::new(offset as u32, &self.device)?;
        let causal_mask = if l == 1 {
            None
        } else {
            Some(self.causal_mask(b, l, offset)?)
        };
        let kv_ids = self.kv_ids_for(&pos_t, l)?;
        let ctx = StepCtx {
            offset,
            pos_t: &pos_t,
            kv_ids: &kv_ids,
            mask: causal_mask.as_ref(),
            cap: None,
        };
        // 自检开关：先跑图内路径（读整段 + device 掩码），比对后再用 eager 产出正式结果。
        // 顺序不能反——GDN 的循环状态每跑一次就前进一格，不幂等，所以第二遍前必须恢复状态。
        // KV 写入是幂等的（同一位置写同一份 k/v），不需要恢复。
        if l == 1
            && bucket_check_enabled()
            && self.kv_cap > 0
            && self.layers.iter().any(|l| l.uses_fixed_kv())
        {
            let snap = self.snapshot_gdn_states()?;
            let want = self.forward_bucketed(input, &pos_t, self.kv_cap)?;
            let c = want
                .argmax(D::Minus1)?
                .to_dtype(DType::U32)?
                .to_vec1::<u32>()?;
            self.restore_gdn_states(&snap)?;
            let out = self.forward_inner(input, &ctx)?;
            let diff = (&out - &want)?.abs()?.max_all()?.to_scalar::<f32>()?;
            let a = out
                .argmax(D::Minus1)?
                .to_dtype(DType::U32)?
                .to_vec1::<u32>()?;
            if a != c {
                candle::bail!(
                    "[bucket-check] offset={offset} argmax 不一致: eager {a:?} vs 图内 {c:?}"
                )
            }
            tracing::warn!("[bucket-check] offset={offset} max_abs_diff={diff:.3e} argmax={a:?}");
            return Ok(out);
        }
        self.forward_inner(input, &ctx)
    }

    /// 图内 decode 一步：位置只从 device 张量 pos_t 读，KV 读整段 cap，
    /// 因果掩码也在 device 上构造。整条路径的形状与数值都不含 host 常量，
    /// 捕获一次后可反复 replay（replay 之间只要更新 pos_t 的内容）。
    ///
    /// 只在 l == 1 时可用：prefill 的 seq 长度是 host 常量、形状随轮次变化，
    /// 属于分桶要解决的问题，不在本函数范围内。
    pub fn forward_bucketed(
        &mut self,
        input: &Tensor,
        pos_t: &Tensor,
        cap: usize,
    ) -> Result<Tensor> {
        let (_, l) = input.dims2()?;
        if l != 1 {
            candle::bail!("图内前向只支持单 token，实际 l={l}")
        }
        stage!("bucket.kv_ids");
        let kv_ids = self.kv_ids_for(pos_t, l)?;
        stage!("bucket.mask");
        let mask = decode_mask(pos_t, cap, self.dtype, self.graph_consts.as_ref())?;
        let ctx = StepCtx {
            offset: 0,
            pos_t,
            kv_ids: &kv_ids,
            mask: Some(&mask),
            cap: Some(cap),
        };
        let out = self.forward_inner(input, &ctx)?;
        // 图内必须把 logits 落到持久缓冲：AUTO_FREE_ON_LAUNCH 会在 replay 结束时回收
        // 图内分配的显存，直接返回那张张量等于返回一块随时会失效的地址。
        match &self.logits_buf {
            Some(b) => {
                b.store(&out)?;
                b.view()
            }
            None => Ok(out),
        }
    }

    /// 捕获前把 GDN 状态缓冲分配好。只作用于线性注意力层，全注意力层没有循环状态。
    fn prepare_persistent_state(&mut self) -> Result<()> {
        let dev = self.device.clone();
        let act_dtype = DType::F32;
        for layer in &mut self.layers {
            if let TokenMixer::LinearAttention(a) = &mut layer.token_mixer {
                a.prepare_state_storage(1, act_dtype, &dev)?;
            }
        }
        Ok(())
    }

    /// CUDA Graph decode 一步：首次调用时懒捕获，之后每步只更新输入再 replay。
    ///
    /// 捕获那一次会真跑前向（会推进 GDN 循环状态、写一行 KV），因此：
    /// - 捕获前快照 GDN 状态，捕获后恢复；
    /// - KV 那行是原地写，随后的真实 replay 会把它覆盖成正确值。
    pub fn forward_graph(&mut self, token: u32, offset: usize) -> Result<Tensor> {
        self.forward_graph_inner(token, offset).map_err(stage_err)
    }

    fn forward_graph_inner(&mut self, token: u32, offset: usize) -> Result<Tensor> {
        if offset >= self.kv_cap {
            candle::bail!(
                "位置 {offset} 超出 KV 缓冲容量 {}：请调大 max_seq 或关掉 Q35_CUDA_GRAPH",
                self.kv_cap
            )
        }
        if self.graph.is_none() {
            let dev = self.device.clone();
            let cap = self.kv_cap;
            // 捕获前必须先用 eager 跑一次同形状的 decode 步。
            // 原因（实测）：candle 的 per-device workspace、量化 matmul 的 scratch 缓冲
            // 都在「第一次遇到某种形状」时分配、并把旧缓冲释放掉，而**捕获期内释放显存是
            // 非法的**（cuMemFreeAsync 返回 CUDA_ERROR_INVALID_VALUE）。冷启动直接捕获会在
            // 某个需要换 workspace 尺寸的算子上炸掉，报错位置还会随形状漂移，极具误导性。
            // 图内常量先建好（arange/full 是 CPU→H2D，捕获进图会读垃圾，见 GraphConsts）
            self.graph_consts = Some(GraphConsts::new(cap, self.dtype, &self.device)?);
            // 状态缓冲必须先于捕获分配好：图内只能原地写，不能再申请显存
            self.prepare_persistent_state()?;
            // candle 的图内 H2D 防护：开着这个 guard 时，小段 host 常量（算子参数、
            // arange 之类）第一次用会被缓存成常驻显存块，捕获期再遇到缓存未命中就
            // 直接报错而不是静默把 host 指针烤进图里。暖机必须在 guard 内跑，
            // 否则捕获期全是未命中。
            let _htod_guard = match &self.device {
                Device::Cuda(d) => Some(d.enable_cuda_graph_htod_cache()),
                _ => None,
            };
            let snap = self.snapshot_gdn_states()?;
            let x = Tensor::new(&[[token]], &self.device)?;
            let pos0 = Tensor::new(&[offset as u32], &self.device)?;
            stage!("warmup");
            // 暖机必须跑**捕获时那条路径**（forward_bucketed），否则 candle 的算子参数
            // 缓存里缺的就是 bucketed 独有的那些 dims/stride 向量，捕获期直接报未命中。
            let warm = self
                .forward_bucketed(&x, &pos0, cap)
                .map_err(|e| candle::Error::Msg(format!("暖机 bucketed 步失败: {e}")))?;
            // logits 持久缓冲照暖机产出的 dtype 建（Q4 反量化是 F32，不是元数据里的 F16）
            self.logits_buf = Some(StateBuf::zeros(
                &[1, self.vocab_size],
                warm.dtype(),
                &self.device,
            )?);
            self.restore_gdn_states(&snap)?;
            // 第二次暖机：上一遍没有 logits 持久缓冲，图里会多出几条 scatter_set / reshape，
            // 它们的参数向量不在 candle 的 h2d param cache 里，捕获期直接报未命中。
            // 状态在两次暖机之间恢复，语义等价于同一步只跑一次。
            self.forward_bucketed(&x, &pos0, cap)
                .map_err(|e| candle::Error::Msg(format!("暖机 bucketed 步(带缓冲)失败: {e}")))?;
            self.restore_gdn_states(&snap)?;
            stage!("graph.capture");
            let g = DecodeGraph::capture(&dev, cap, offset as u32, |i, p, c| {
                self.forward_bucketed(i, p, c).map_err(stage_err)
            })?;
            if !no_restore() {
                self.restore_gdn_states(&snap)?;
            }
            self.graph = Some(Arc::new(g));
            if perf_enabled() {
                tracing::warn!("[graph] decode 步已捕获，cap={cap}");
            }
        }
        // 对拍：eager 先算参考值（会推进 GDN 状态），算完恢复，再 replay。
        let want = if graph_check_enabled() {
            let snap = self.snapshot_gdn_states()?;
            let x = Tensor::new(&[[token]], &self.device)?;
            let out = self.forward(&x, offset)?;
            self.restore_gdn_states(&snap)?;
            Some(out)
        } else {
            None
        };
        let g = self.graph.as_ref().expect("上面刚设过");
        stage!("graph.set");
        g.set(token, offset as u32)?;
        stage!("graph.launch");
        if std::env::var("Q35_GRAPH_SYNC").is_ok_and(|v| v == "1") {
            let _ = self.device.synchronize();
            if let Device::Cuda(d) = &self.device {
                match d.cuda_stream().context().check_err() {
                    Ok(()) => tracing::warn!("[graph-sync] launch 前 ok"),
                    Err(e) => tracing::warn!("[graph-sync] launch 前有错误: {e}"),
                }
            }
        }
        let out = g.replay()?;
        stage!("graph.done");
        if let Some(want) = want {
            let diff = (&want - &out)?.abs()?.max_all()?.to_scalar::<f32>()?;
            let a = want
                .argmax(D::Minus1)?
                .to_dtype(DType::U32)?
                .to_vec1::<u32>()?;
            let b = out
                .argmax(D::Minus1)?
                .to_dtype(DType::U32)?
                .to_vec1::<u32>()?;
            if a != b {
                candle::bail!(
                    "[graph-check] offset={offset} argmax 不一致: eager {a:?} vs 图 {b:?}"
                )
            }
            tracing::warn!("[graph-check] offset={offset} max_abs_diff={diff:.3e} argmax={a:?}");
        }
        Ok(out)
    }
    /// 所有全注意力层共用同一份写入索引：位置取自 device 张量（host 整数会被烘进图），
    /// 且只构造一次，避免每层重复 4 个算子。
    /// Concat 变体不需要索引（append 自己拼），跳过构造省掉每步 4 个算子。
    /// 快照 / 恢复 GDN 层的循环状态（conv + recurrent）。
    ///
    /// 只给自检用：GDN 的递推每跑一次 token 就前进一格，同一 token 跑两遍会推进两格。
    /// Tensor 的 clone 是共享存储，但 GDN 更新走的是"算新张量再赋值"，不会原地改旧存储，
    /// 所以快照在第二遍跑完后仍然有效。
    fn snapshot_gdn_states(&self) -> Result<Vec<(Option<Tensor>, Option<Tensor>)>> {
        self.layers.iter().map(|l| l.states()).collect()
    }

    /// 恢复快照（按引用传，调用方常常要恢复两次：暖机一次、捕获一次）
    fn restore_gdn_states(&mut self, snap: &[(Option<Tensor>, Option<Tensor>)]) -> Result<()> {
        for (layer, s) in self.layers.iter_mut().zip(snap) {
            layer.set_states(s)?;
        }
        Ok(())
    }

    fn kv_ids_for(&self, pos_t: &Tensor, l: usize) -> Result<Tensor> {
        if self.layers.iter().any(|l| l.uses_fixed_kv()) {
            // 走图时用捕获前建好的常量偏移，避免 Tensor::arange 的 H2D 节点被捕获
            match &self.graph_consts {
                Some(c) if c.offsets.elem_count() == l => FixedKvCache::index_for_with(
                    pos_t,
                    &c.offsets,
                    self.kv_shape.0,
                    self.kv_shape.1,
                ),
                _ => FixedKvCache::index_for(pos_t, l, self.kv_shape.0, self.kv_shape.1),
            }
        } else {
            Tensor::new(&[0u32], &self.device)
        }
    }

    fn forward_inner(&mut self, input: &Tensor, ctx: &StepCtx) -> Result<Tensor> {
        let (_, l) = input.dims2()?;
        let offset = ctx.offset;
        stage!("embed");
        let mut h = self.embed_tokens.forward(input)?;
        if h.dtype() != DType::F32 {
            h = h.to_dtype(DType::F32)?;
        }
        if stop_hit("embed") {
            return h.narrow(1, l - 1, 1)?.squeeze(1);
        }
        // 计时探针：不做任何强制同步，只记录 CPU 侧耗时；
        // 需要精确 GPU 时间时由调用方（pipe）在整轮结束处同步一次。
        let mut gdn_mix = std::time::Duration::ZERO;
        let mut full_mix = std::time::Duration::ZERO;
        let mut mlp_t = std::time::Duration::ZERO;
        let mut n_gdn = 0usize;
        let mut n_full = 0usize;
        let layer_limit = graph_layer_limit();
        for (i, layer) in self.layers.iter_mut().enumerate() {
            if layer_limit.is_some_and(|n| i >= n) {
                break;
            }
            let mut kind = "";
            let (mut tm, mut tp) = (std::time::Duration::ZERO, std::time::Duration::ZERO);
            h = layer
                .forward(&h, ctx, &mut kind, Some(&mut tm), Some(&mut tp))
                .map_err(|e| {
                    let s = stage::get();
                    candle::Error::Msg(format!("[stage={s}] layer{i}({kind}) {e}"))
                })?;
            if std::env::var("Q35_DEBUG_LAYER").is_ok() {
                let flat = h.flatten_all()?;
                let amax = flat.abs()?.max(0)?.to_scalar::<f32>()?;
                let amin = flat.abs()?.min(0)?.to_scalar::<f32>()?;
                let mean = flat.mean_all()?.to_scalar::<f32>()?;
                eprintln!(
                    "[layer {i:02} {kind:4}] amax={amax:.4e} amin={amin:.4e} mean={mean:.4e} dtype={:?}",
                    h.dtype()
                );
            }
            if kind == "gdn" {
                n_gdn += 1;
                gdn_mix += tm;
            } else {
                n_full += 1;
                full_mix += tm;
            }
            mlp_t += tp;
        }
        if perf_enabled() {
            tracing::info!(
                "[q35-layers] l={l} offset={offset} gdn({n_gdn})={gdn_mix:.1?} full({n_full})={full_mix:.1?} mlp={mlp_t:.1?}"
            );
        }
        stage!("final_norm");
        let h = self.norm.forward(&h)?;
        if stop_hit("final_norm") {
            return h.narrow(1, l - 1, 1)?.squeeze(1);
        }
        stage!("lm_head");
        let last_hidden = h.narrow(1, l - 1, 1)?;
        self.lm_head.forward(&last_hidden)?.squeeze(1)
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in &mut self.layers {
            layer.clear_kv_cache();
        }
    }
}

#[cfg(test)]
mod dw_tests {
    use super::*;

    #[test]
    fn test_depthwise_conv1d_matches_naive() -> Result<()> {
        let dev = Device::Cpu;
        let (b, c, l, k, pad) = (2usize, 8usize, 10usize, 4usize, 3usize);
        let x = Tensor::randn(0f32, 1f32, (b, c, l), &dev)?;
        let w = Tensor::randn(0f32, 1f32, (c, 1, k), &dev)?;

        let got = depthwise_conv1d(&x, &w, pad)?;

        // 朴素对照: candle groups conv
        let want = x.conv1d(&w, pad, 1, 1, c)?;
        let diff = got.sub(&want)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(diff < 1e-4f32, "depthwise 与 naive 差异过大: {diff}");
        Ok(())
    }

    /// tile_heads 必须等价于 ggml_repeat_4d：输出头 h 取输入头 h % n。
    /// 27B 就是靠这条（48 个 v head 对 16 个 k head），写成 repeat_interleave 会乱码。
    #[test]
    fn test_tile_heads_matches_ggml_repeat() -> Result<()> {
        let dev = Device::Cpu;
        let (b, l, n, d) = (1usize, 2usize, 3usize, 2usize);
        // 每个头的值 = 头号，便于直接断言
        let vals: Vec<f32> = (0..b * l * n * d).map(|i| ((i / d) % n) as f32).collect();
        let x = Tensor::from_vec(vals.clone(), (b, l, n, d), &dev)?;

        let got = tile_heads(&x, 4, 2)?;
        assert_eq!(got.dims4()?, (b, l, n * 4, d));
        let out = got.flatten_all()?.to_vec1::<f32>()?;

        // 期望：头序列 0,1,2,0,1,2,0,1,2,0,1,2（平铺），每个头 d 个相同值
        for l_i in 0..l {
            for h in 0..n * 4 {
                let want = (h % n) as f32;
                for k in 0..d {
                    let idx = (l_i * n * 4 + h) * d + k;
                    assert_eq!(out[idx], want, "l={l_i} h={h} k={k}");
                }
            }
        }
        Ok(())
    }

    /// 部分旋转：rot_dim 之后的维度必须原样保留，rot_dim 之内必须真的被旋转。
    #[test]
    fn test_rope_partial_keeps_tail() -> Result<()> {
        let dev = Device::Cpu;
        let (head_dim, rot_dim, seq) = (8usize, 4usize, 3usize);
        let rope = RotaryEmbedding::new(DType::F32, head_dim, rot_dim, 64, 1e7, &dev)?;

        let x = Tensor::ones((1, 1, seq, head_dim), DType::F32, &dev)?;
        let cos = rope.cos.narrow(0, 0, seq)?;
        let sin = rope.sin.narrow(0, 0, seq)?;
        let out = rope.rope_partial(&x, &cos, &sin)?;
        assert_eq!(out.dims4()?, (1, 1, seq, head_dim));

        let tail = out.narrow(D::Minus1, rot_dim, head_dim - rot_dim)?;
        let tail_sum = tail.sum_all()?.to_scalar::<f32>()?;
        assert!(
            (tail_sum - (seq * (head_dim - rot_dim)) as f32).abs() < 1e-5,
            "未旋转的尾部被改动: sum={tail_sum}"
        );

        let head = out
            .narrow(D::Minus1, 0, rot_dim)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let changed = head.iter().any(|v| (v - 1.0f32).abs() > 1e-6);
        assert!(changed, "前 rot_dim 维没有被旋转: {head:?}");
        Ok(())
    }
}
