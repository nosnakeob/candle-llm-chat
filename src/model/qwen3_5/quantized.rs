//! Qwen3.5 implementation with quantization support.
//!
//! Qwen3.5 is a hybrid architecture combining Gated DeltaNet (Linear Attention)
//! and Gated Attention (Full Softmax Attention).
//!
//! Based on the Qwen 3.5 architecture and implemented with quantized weights
//! for reduced memory usage and faster inference.
//!
use candle_transformers::models::with_tracing::QMatMul;
use candle_transformers::quantized_nn::RmsNorm;
use candle_transformers::utils::repeat_kv;
use candle::quantized::{gguf_file, QTensor};
use candle::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{kv_cache::ConcatKvCache, Activation, Embedding};
use std::io::{Read, Seek};
use std::sync::Arc;

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
}

impl RotaryEmbedding {
    pub fn new(
        dtype: DType,
        head_dim: usize,
        max_position_embeddings: usize,
        rope_theta: f64,
        dev: &Device,
    ) -> Result<Self> {
        let dim = head_dim;
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
        })
    }

    pub fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let (_, _, seq_len, _) = q.dims4()?;
        let cos = self.cos.narrow(0, offset, seq_len)?.to_dtype(q.dtype())?;
        let sin = self.sin.narrow(0, offset, seq_len)?.to_dtype(q.dtype())?;
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
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
    kv_cache: ConcatKvCache,
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

        let kv_cache = ConcatKvCache::new(2);

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

    fn forward(&mut self, x: &Tensor, attn_mask: Option<&Tensor>, offset: usize) -> Result<Tensor> {
        let (b, l, _) = x.dims3()?;

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

        let q_flat = q.flatten(0, 2)?;
        let k_flat = k.flatten(0, 2)?;

        let q_flat = self.q_norm.forward(&q_flat)?;
        let k_flat = self.k_norm.forward(&k_flat)?;
        let q = q_flat.reshape((b, self.num_heads, l, self.head_dim))?;
        let k = k_flat.reshape((b, self.num_kv_heads, l, self.head_dim))?;

        let (q, k) = self.rotary_emb.apply(&q, &k, offset)?;

        let (k, v) = self.kv_cache.append(&k, &v)?;

        let k = repeat_kv(k, self.num_kv_groups)?.contiguous()?;
        let v = repeat_kv(v, self.num_kv_groups)?.contiguous()?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        if let Some(m) = attn_mask {
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
        let gate_sigmoid = candle_nn::ops::sigmoid(&gate.to_dtype(DType::F32)?)?.to_dtype(ctx.dtype())?;
        let ctx = ctx.broadcast_mul(&gate_sigmoid)?;
        
        let reshaped_ctx = ctx.reshape((b, l, self.num_heads * self.head_dim))?;
        self.o_proj.forward(&reshaped_ctx)
    }

    fn clear_kv_cache(&mut self) {
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

        let conv1d_weight = gg.tensor(&format!("{prefix}.ssm_conv1d.weight"))?
            .dequantize(&gg.device)?
            .unsqueeze(1)?;
        
        let a_log = gg.tensor(&format!("{prefix}.ssm_a"))?.dequantize(&gg.device)?;
        let dt_bias = gg.tensor(&format!("{prefix}.ssm_dt.bias"))?.dequantize(&gg.device)?;
        
        let dt_bias_f32 = dt_bias.to_dtype(DType::F32)?;
        let neg_a_f32 = (&a_log.to_dtype(DType::F32)?.exp()? * -1.0)?;
        
        let norm_weight = gg.tensor(&format!("{prefix}.ssm_norm.weight"))?.dequantize(&gg.device)?;

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
        })
    }

    fn l2_norm(xs: &Tensor) -> Result<Tensor> {
        let eps = 1e-6;
        let norm = (xs.sqr()?.sum_keepdim(D::Minus1)? + eps)?.sqrt()?;
        xs.broadcast_div(&norm)
    }

    fn rms_norm_gated(&self, x: &Tensor, g: &Tensor) -> Result<Tensor> {
        let x_dtype = x.dtype();
        let x_f32 = x.to_dtype(DType::F32)?;
        let g_f32 = g.to_dtype(DType::F32)?;
        
        let gate = g_f32.silu()?;
        let norm_x = (x_f32.sqr()?.mean_keepdim(D::Minus1)? + self.norm_eps)?.sqrt()?;
        let x_normed = x_f32.broadcast_div(&norm_x)?;
        (x_normed * gate)?.broadcast_mul(&self.norm_weight.to_dtype(DType::F32)?)?.to_dtype(x_dtype)
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
        
        let scale = 1.0 / (query.dim(3)? as f64).sqrt();
        let query = (query * scale)?;
        
        let mut core_attn_out_vec = Vec::with_capacity(seq_len);
        
        let mut last_recurrent_state = match initial_state {
            Some(state) => state.to_dtype(DType::F32)?,
            None => Tensor::zeros((batch_size, num_heads, k_head_dim, v_head_dim), DType::F32, query.device())?,
        };
        
        let query_f32 = query.to_dtype(DType::F32)?.contiguous()?;
        let key_f32 = key.to_dtype(DType::F32)?.contiguous()?;
        let value_f32 = value.to_dtype(DType::F32)?.contiguous()?;
        let g_exp = g.to_dtype(DType::F32)?.exp()?.unsqueeze(3)?.contiguous()?;
        let beta_f32 = beta.to_dtype(DType::F32)?.unsqueeze(3)?.contiguous()?;
        // kernel 接口：g/beta 用 (B,H)，从 (B,H,1,1) 收缩
        let g_bh = g_exp.squeeze(3)?.squeeze(2)?;
        let beta_bh = beta_f32.squeeze(3)?.squeeze(2)?;

        // ── decode（seq_len == 1）：整条递推链走融合算子 ─────────────────
        // 逐元素算子版本每 token 要 7+ 次 kernel launch；融合后整个递推在一遍扫描里完成。
        let state_heads_ok = last_recurrent_state.dim(1).ok() == Some(num_heads)
            && last_recurrent_state.dim(2).ok() == Some(k_head_dim);
        if seq_len == 1 && state_heads_ok && last_recurrent_state.device().is_cpu() {
            let q_contig = query_f32.contiguous()?;

            let out = gated_delta_step(
                &last_recurrent_state,
                &q_contig,
                &key_f32,
                &value_f32,
                &g_bh,
                &beta_bh,
            )?;
            let core_attn_out = out.transpose(1, 2)?;
            return Ok((core_attn_out, last_recurrent_state));
        }

        for i in 0..seq_len {
            let q_t = query_f32.narrow(2, i, 1)?;
            let k_t = key_f32.narrow(2, i, 1)?;
            let v_t = value_f32.narrow(2, i, 1)?;
            let g_t = g_exp.narrow(2, i, 1)?;
            let beta_t = beta_f32.narrow(2, i, 1)?;

            last_recurrent_state = last_recurrent_state.broadcast_mul(&g_t)?;
            
            let kv_mem = k_t.matmul(&last_recurrent_state)?;
            let delta = (&v_t - &kv_mem)?.broadcast_mul(&beta_t)?;
            
            let delta_k = k_t.transpose(2, 3)?.matmul(&delta)?;
            last_recurrent_state = (&last_recurrent_state + &delta_k)?;
            
            let out_t = q_t.matmul(&last_recurrent_state)?;
            core_attn_out_vec.push(out_t);
        }
        
        let core_attn_out = Tensor::cat(&core_attn_out_vec, 2)?;
        let core_attn_out = core_attn_out.transpose(1, 2)?;
        Ok((core_attn_out, last_recurrent_state))
    }

    fn forward(&mut self, hidden_states: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len, _) = hidden_states.dims3()?;
        let initial_dtype = hidden_states.dtype();
        
        let mixed_qkv = self.in_proj_qkv.forward(hidden_states)?.transpose(1, 2)?; 
        
        let z = self.in_proj_z.forward(hidden_states)?;
        let z = z.reshape((batch_size, seq_len, (), self.head_v_dim))?;
        
        let b = self.in_proj_b.forward(hidden_states)?;
        let a = self.in_proj_a.forward(hidden_states)?;
        
        let use_precomputed_states = self.conv_state.is_some() && seq_len == 1;
        
        let t_conv = std::time::Instant::now();
        let mixed_qkv = if use_precomputed_states {
            let conv_state = self.conv_state.as_mut().unwrap();
            let conv_state_data = Tensor::cat(&[conv_state.as_ref(), &mixed_qkv], 2)?;
            *conv_state = conv_state_data.narrow(2, 1, self.conv_kernel_size - 1)?;
            let out = depthwise_conv1d(&conv_state_data, &self.conv1d_weight, 0)?;
            candle_nn::ops::silu(&out)?
        } else {
            let pad = self.conv_kernel_size - 1;
            let padding = Tensor::zeros((batch_size, self.conv_dim, pad), mixed_qkv.dtype(), mixed_qkv.device())?;
            let padded_qkv = Tensor::cat(&[&padding, &mixed_qkv], 2)?;
            self.conv_state = Some(padded_qkv.narrow(2, seq_len, pad)?);
            let out = depthwise_conv1d(&padded_qkv, &self.conv1d_weight, 0)?;
            candle_nn::ops::silu(&out)?
        };
        
        self.conv1d_weight.device().synchronize().ok();
        let dt_conv = t_conv.elapsed();
        let mixed_qkv = mixed_qkv.transpose(1, 2)?;
        
        let q = mixed_qkv.narrow(D::Minus1, 0, self.key_dim)?;
        let k = mixed_qkv.narrow(D::Minus1, self.key_dim, self.key_dim)?;
        let v = mixed_qkv.narrow(D::Minus1, self.key_dim * 2, self.value_dim)?;
        
        let q = q.reshape((batch_size, seq_len, (), self.head_k_dim))?;
        let k = k.reshape((batch_size, seq_len, (), self.head_k_dim))?;
        let v = v.reshape((batch_size, seq_len, (), self.head_v_dim))?;
        
        let beta = candle_nn::ops::sigmoid(&b)?;
        let g = {
            let a_f32 = a.to_dtype(DType::F32)?;
            let a_plus_dt = a_f32.broadcast_add(&self.dt_bias_f32)?;
            let softplus = (a_plus_dt.exp()? + 1.0)?.log()?;
            self.neg_a_f32.broadcast_mul(&softplus)?
        };
        
        let repeat_n = self.num_v_heads / self.num_k_heads;
        let q = repeat_interleave(&q, repeat_n, 2)?;
        let k = repeat_interleave(&k, repeat_n, 2)?;
        
        let initial_state = if use_precomputed_states {
            self.recurrent_state.as_ref()
        } else {
            None
        };
        
        let t_dr = std::time::Instant::now();
        let (core_attn_out, new_state) = self.torch_recurrent_gated_delta_rule(&q, &k, &v, &g, &beta, initial_state)?;
        q.device().synchronize().ok();
        let dt_dr = t_dr.elapsed();
        self.recurrent_state = Some(new_state);
        
        let core_attn_out = core_attn_out.to_dtype(initial_dtype)?;
        let core_attn_out = core_attn_out.reshape((batch_size, seq_len, self.num_v_heads, self.head_v_dim))?;
        let core_attn_out = core_attn_out.reshape(((), self.head_v_dim))?;
        let z_flat = z.reshape(((), self.head_v_dim))?;
        let core_attn_out = self.rms_norm_gated(&core_attn_out, &z_flat)?;
        let core_attn_out = core_attn_out.reshape((batch_size, seq_len, self.num_v_heads * self.head_v_dim))?;
        
        let out = self.out_proj.forward(&core_attn_out);
        if std::env::var("Q35_PERF").is_ok() && seq_len <= 2 {
            tracing::info!("[q35-gdn] conv={dt_conv:.1?} delta_rule={dt_dr:.1?}");
        }
        out
    }

    fn clear_kv_cache(&mut self) {
        self.conv_state = None;
        self.recurrent_state = None;
    }
}


// ─── 融合算子：Gated DeltaNet 单 token 递推 ────────────────────────────────────
//
// 逐 token 解码时，递推公式若拆成 candle 的逐元素算子会变成 7+ 次独立 kernel launch
// （broadcast_mul / matmul / sub ⋯），每次数十微秒；真正的计算量只有
// H×K×V×4 ≈ 1 MFLOP（<10µs）。这里把「衰减 → kv_mem → delta → 状态更新 → 输出」
// 整条链融进一次 CPU 调用。
//
// 张量约定（decode：seq_len == 1，全部行主序）：
//   state: (B, H, K, V)
//   q, k:  (B, H, K)   v: (B, H, V)   g, beta: (B, H)
//   out:   (B, H, V)
// 混合状态 (K, V)：K=V 时用方阵版，K≠V 时用通用版。

/// 融合算子输入：非 state 的那几路张量 + 指向可变状态的裸指针
struct GatedDeltaStepArgs {
    /// state 的底层数据；生命周期由调用方的 Tensor 持有，CPU 上 resize 即时完成
    state: *mut f32,
    /// q 每行内部真实数据相对行首的偏移（packed 布局里前段是 gate）
    k_offset: usize,
    /// state 第三维的真实长度（= 真实 key 维）
    state_kd: usize,
    /// state 第四维的真实长度（= value 维）
    state_vd: usize,
    q: *const f32,
    k: *const f32,
    v: *const f32,
    g: *const f32,
    beta: *const f32,
    dims: [usize; 8], // b, h, seq, k, v, bq, hq, vq
    scale: f32,
}

// 裸指针只是借用视图，实际访问在 op 内部同步完成
unsafe impl Send for GatedDeltaStepArgs {}
unsafe impl Sync for GatedDeltaStepArgs {}

impl GatedDeltaStepArgs {
    fn run(&self) -> Vec<f32> {
        let [b, h, seq, kd, vd, bq, hq, vq] = self.dims;

        if kd == vd {
            self.step_kv_square(b, h, seq, kd)
        } else {
            self.step_general(b, h, seq, kd, vd, bq, hq, vq)
        }
    }

    /// 方阵版：状态切片连续，衰减 / kv·S / kᵀ·δ / q·S 在同一遍扫描里完成
    fn step_kv_square(&self, b: usize, h: usize, seq: usize, dim: usize) -> Vec<f32> {
        // q 的行 stride 可能大于 k：kernel 里 q/gate 仍是拼接后的 (B,H,1,K*2) 视图
        let dim = self.dims[3]; // 真实 key 维（state 第三维）
        let vd = self.dims[4];
        let qstride = self.dims[5];
        let mut out = vec![0f32; b * h * seq * vd];
        // delta 按 (b, h, vd) 布局，用绝对 head 下标索引
        let mut delta = vec![0f32; b * h * vd];
        let bh = b * h;
        for i in 0..bh {
            let (skd, svd) = (self.state_kd, self.state_vd);
            let state =
                unsafe { std::slice::from_raw_parts_mut(self.state.add(i * skd * svd), skd * svd) };
            debug_assert_eq!((skd, svd), (dim, vd), "state 真实维度与参数不一致");
            for t in 0..seq {
                let qp = (i * seq + t) * qstride; // q 按 (b, h, seq, q_stride) 索引
                let kp = (i * seq + t) * dim;     // k 按 (b, h, seq, k) 索引
                let vp = (i * seq + t) * vd;      // v/out 按 (b, h, seq, v) 索引
                // packed 布局里 q 的真实数据从 k 偏移处开始（前 k 位是 gate）
                let q = unsafe { std::slice::from_raw_parts(self.q.add(qp + self.k_offset), dim) };
                let k = unsafe { std::slice::from_raw_parts(self.k.add(kp), dim) };
                let v = unsafe { std::slice::from_raw_parts(self.v.add(vp), vd) };
                let g = unsafe { *self.g.add(i * seq + t) };
                let beta = unsafe { *self.beta.add(i * seq + t) };
                let d = &mut delta[i * vd..(i + 1) * vd];

                d.iter_mut().for_each(|x| *x = 0.);
                // 1) 先把整块状态按 g 衰减；2) 在**同一个衰减后矩阵**上算 kᵀ·S。
                //    两次内层循环都在整块状态上迭代，避免与外层 k 维的循环顺序耦合。
                for sv in state.iter_mut() {
                    *sv *= g;
                }
                for (r, row) in state.chunks(svd).enumerate().take(skd) {
                    let kr = k[r];
                    if kr == 0. {
                        continue;
                    }
                    for (c, sv) in row.iter().enumerate() {
                        d[c] += kr * sv;
                    }
                }
                // 3) δ = (v − kᵀ·S)·β
                for c in 0..vd {
                    d[c] = (v[c] - d[c]) * beta;
                }
                // 4) S ← S + k ⊗ δ，同时累积输出 q·S
                let o = &mut out[vp..vp + vd];
                o.iter_mut().for_each(|x| *x = 0.);
                for (r, row) in state.chunks_mut(svd).enumerate().take(skd) {
                    let (kr, qr) = (k[r], q[r] * self.scale);
                    for (c, sv) in row.iter_mut().enumerate() {
                        *sv += kr * d[c];
                        o[c] += qr * *sv;
                    }
                }
            }
        }
        out
    }

    /// 通用版（K ≠ V）：q/k 支持按 num_v_heads 广播（bq/hq 可为 1）
    fn step_general(&self, b: usize, h: usize, seq: usize, kd: usize, vd: usize, bq: usize, hq: usize, vq: usize) -> Vec<f32> {
        let _ = (bq, hq, vq);
        let mut out = vec![0f32; b * h * seq * vd];
        let mut delta = vec![0f32; h * vd];
        for bi in 0..b {
            for hi in 0..h {
                let (bqi, hqi, vqi) = (bi % bq, hi % hq, hi % vq);
                let sp = (bi * h + hi) * self.state_kd * vd;
                let state = unsafe { std::slice::from_raw_parts_mut(self.state.add(sp), kd * vd) };
                for t in 0..seq {
                    let qp = ((bqi * hq + hqi) * seq + t) * self.dims[6];
                    let kp = ((bqi * hq + hqi) * seq + t) * kd;
                    let vp = ((bi * h + hi) * seq + t) * vd;
                    let (q, k, v) = unsafe {
                        (
                            std::slice::from_raw_parts(self.q.add(qp), kd),
                            std::slice::from_raw_parts(self.k.add(kp), kd),
                            std::slice::from_raw_parts(self.v.add(vp), vd),
                        )
                    };
                    let g = unsafe { *self.g.add(bqi * hq * seq + hqi * seq + t) };
                    let beta = unsafe { *self.beta.add(bqi * hq * seq + hqi * seq + t) };
                    let d = &mut delta[hi * vd..(hi + 1) * vd];

                    d.iter_mut().for_each(|x| *x = 0.);
                    for (r, sr) in state.chunks_mut(vd).enumerate() {
                        let kr = k[r];
                        for (c, sv) in sr.iter_mut().enumerate() {
                            *sv *= g;
                            d[c] += kr * *sv;
                        }
                    }
                    for c in 0..vd {
                        d[c] = (v[c] - d[c]) * beta;
                    }
                    let o = &mut out[vp..vp + vd];
                    o.iter_mut().for_each(|x| *x = 0.);
                    for (r, sr) in state.chunks_mut(vd).enumerate() {
                        let (kr, qr) = (k[r], q[r] * self.scale);
                        for (c, sv) in sr.iter_mut().enumerate() {
                            *sv += kr * d[c];
                            o[c] += qr * *sv;
                        }
                    }
                }
            }
        }
        out
    }
}

/// 融合执行：(state, q, k, v, g, beta) → out
///
/// state 会被原地更新。candle 的 GPU 路径把 storage 放在锁后（只为 guard 存活期），
/// 所以这里只支持 CPU 张量；调用方在取指针前已保证内存连续。
fn gated_delta_step(
    state: &Tensor,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
) -> Result<Tensor> {
    let (b, h, kd, vd) = state.dims4()?; // state: (B, H, K, V)
    let seq = 1;
    for (tensor, name, rank) in [
        (q, "q", 4), (k, "k", 4), (v, "v", 4), (g, "g", 3), (beta, "beta", 3),
    ] {
        if tensor.rank() != rank {
            candle::bail!("gated_delta_step: {name} 期望 {rank} 维，实际 {:?}", tensor.dims());
        }
    }
    // 所有输入必须是连续的 f32 CPU 张量：state 会被原地改写，非连续会打错内存
    for (tensor, name) in [(state, "state"), (q, "q"), (k, "k"), (v, "v"), (g, "g"), (beta, "beta")] {
        let (guard, layout) = tensor.storage_and_layout();
        if !layout.is_contiguous() {
            candle::bail!("gated_delta_step: {name} 不是连续内存（请先 .contiguous()）");
        }
        if !matches!(&*guard, candle::Storage::Cpu(_)) || tensor.dtype() != DType::F32 {
            candle::bail!(
                "gated_delta_step: {name} 必须是 CPU 上的 f32（实际 {:?}/{:?}）",
                tensor.dtype(),
                tensor.device()
            );
        }
    }

    let (bq, hq, sq, kq) = q.dims4()?;
    let (bv, hv, sv, vq) = v.dims4()?;
    let (gb, gh, gs) = g.dims3()?;
    let (bb, bh, bs) = beta.dims3()?;
    if sq != seq || sv != seq || gs != seq || bs != seq || kq < kd || vq != vd
        || bq != b || hq != h || bv != b || hv != h || gb != b || gh != h || bb != b || bh != h
    {
        candle::bail!(
            "gated_delta_step 形状不匹配: state={b}x{h}x{kd}x{vd} q={bq}x{hq}x{sq}x{kq} v={bv}x{hv}x{sv}x{vq} g={gb}x{gh}x{gs} beta={bb}x{bh}x{bs}"
        );
    }

    // 取指针：storage guard 只在取指针期间存活，之后 Tensor 仍持有 Arc
    let raw = |tensor: &Tensor, name: &str| -> candle::Result<*const f32> {
        let (guard, layout) = tensor.storage_and_layout();
        let start = layout
            .contiguous_offsets()
            .map(|(start, _)| start)
            .ok_or_else(|| candle::Error::Msg(format!("{name} 不是连续内存").into()))?;
        match &*guard {
            candle::Storage::Cpu(cpu) => Ok(unsafe { cpu.as_slice::<f32>()?.as_ptr().add(start) }),
            other => candle::bail!(
                "gated_delta_step 只实现了 CPU 版本（{name} 在 {:?}）",
                other.device()
            ),
        }
    };

    let kd_real = kq / 2;
    let state_ptr = raw(state, "state")? as *mut f32;
    let args = GatedDeltaStepArgs {
        k_offset: kq / 2,
        state_kd: kd_real,
        state_vd: vd,
        state: state_ptr,
        q: raw(q, "q")?,
        k: raw(k, "k")?,
        v: raw(v, "v")?,
        g: raw(g, "g")?,
        beta: raw(beta, "beta")?,
        // dims: [b, h, seq, kd, vd, q_stride, q_stride_v]
        // 真实 key 维 = kq（q 的 packed 宽度的 1/2；非 packed 时等于 kd）
        dims: [b, h, seq, kd_real, vd, kq, kq, hv],
        scale: 1.0 / (kd_real as f32).sqrt(),
    };

    let out = args.run();
    Tensor::from_vec(out, (b, h, seq, vd), state.device())
}

/// 高效 depthwise conv1d（每通道独立卷积，kernel_size 小、通道多时远快于 candle 的 groups 分解）
/// x: (b, c, l_in), weight: (c, 1, k) → 输出 (b, c, l_out)
fn depthwise_conv1d(x: &Tensor, weight: &Tensor, padding: usize) -> Result<Tensor> {
    let (_, c, _) = x.dims3()?;
    let k = weight.dim(2)?;
    let w = weight.reshape((c, k))?.to_dtype(x.dtype())?; // (c, k)

    // 用 k 次移位切片 + 广播乘加累积，替代 candle 的 groups 循环分解
    // 输入先 pad 到 (b, c, padding + l_in)
    let x_padded = if padding > 0 {
        // candle conv1d padding 语义: 左右各 pad `padding` 个
        let zeros = Tensor::zeros((x.dim(0)?, c, padding), x.dtype(), x.device())?;
        Tensor::cat(&[&zeros, x, &zeros], 2)?
    } else {
        x.clone()
    };
    let l_out = x_padded.dim(2)? - k + 1;

    // 累加: out += x_padded[:, :, i:i+l_out] * w[:, i]
    let mut acc = x_padded.narrow(2, 0, l_out)?.broadcast_mul(&w.narrow(1, 0, 1)?.reshape((1, c, 1))?)?;
    for i in 1..k {
        let xi = x_padded.narrow(2, i, l_out)?;
        let wi = w.narrow(1, i, 1)?.reshape((1, c, 1))?;
        acc = acc.broadcast_add(&xi.broadcast_mul(&wi)?)?;
    }
    Ok(acc)
}

fn repeat_interleave(img: &Tensor, repeats: usize, dim: usize) -> Result<Tensor> {
    if repeats == 1 {
        return Ok(img.clone());
    }
    let mut dims = img.dims().to_vec();
    dims[dim] *= repeats;
    let final_dims = dims.clone();
    let img = img.unsqueeze(dim + 1)?;
    let mut expand_dims = img.dims().to_vec();
    expand_dims[dim + 1] = repeats;
    let expanded = img.expand(expand_dims.as_slice())?;
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
        let post_attention_layernorm = gg.rms_norm(&format!("{prefix}.post_attention_norm.weight"), rms_norm_eps)?;

        Ok(Self {
            token_mixer,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(&mut self, x: &Tensor, mask: Option<&Tensor>, offset: usize, layer_kind: &mut &'static str, t_mixer: Option<&mut std::time::Duration>, t_mlp: Option<&mut std::time::Duration>) -> Result<Tensor> {
        let residual = x;
        let x = self.input_layernorm.forward(x)?;
        let t0 = std::time::Instant::now();
        let x = match &mut self.token_mixer {
            TokenMixer::FullAttention(attn) => {
                *layer_kind = "full";
                attn.forward(&x, mask, offset)?
            }
            TokenMixer::LinearAttention(attn) => {
                *layer_kind = "gdn";
                attn.forward(&x)?
            }
        };
        if let Some(d) = t_mixer {
            *d += t0.elapsed();
        }
        let x = (residual + x)?;
        
        let residual = &x;
        let x = self.post_attention_layernorm.forward(&x)?;
        let x = x.apply(&self.mlp)?;
        x + residual
    }

    fn clear_kv_cache(&mut self) {
        match &mut self.token_mixer {
            TokenMixer::FullAttention(attn) => attn.clear_kv_cache(),
            TokenMixer::LinearAttention(attn) => attn.clear_kv_cache(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ModelWeights {
    embed_tokens: Embedding,
    layers: Vec<LayerWeights>,
    norm: RmsNorm,
    lm_head: QMatMul,
    device: Device,
    dtype: DType,
}

impl ModelWeights {
    pub fn from_gguf<R: Read + Seek>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        let mut gg = Gguf::new(ct, reader, device.clone());
        let md_get = |s: &str| {
            match gg.metadata().get(s) {
                Some(v) => Ok(v),
                None => {
                    let s35 = s.replace("qwen3.", "qwen35.");
                    match gg.metadata().get(&s35) {
                        Some(v) => Ok(v),
                        None => candle::bail!("cannot find {s} or {s35} in metadata"),
                    }
                }
            }
        };

        let num_attention_heads = md_get("qwen3.attention.head_count")?.to_u32()? as usize;
        let num_kv_heads = md_get("qwen3.attention.head_count_kv")?.to_u32()? as usize;
        let key_length = md_get("qwen3.attention.key_length")?.to_u32()? as usize;
        
        let head_dim = key_length; 
        let num_layers = md_get("qwen3.block_count")?.to_u32()? as usize;
        let hidden_size = md_get("qwen3.embedding_length")?.to_u32()? as usize;
        let max_position_embeddings = md_get("qwen3.context_length")?.to_u32()? as usize;
        let rms_norm_eps = md_get("qwen3.attention.layer_norm_rms_epsilon")?.to_f32()? as f64;
        let rope_freq_base = md_get("qwen3.rope.freq_base")?.to_f32()? as f64;
        
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

        let embed_tensor = gg.tensor("token_embd.weight")?;
        let embed_tokens = Embedding::new(embed_tensor.dequantize(device)?, hidden_size);

        let rotary = Arc::new(RotaryEmbedding::new(
            dtype,
            head_dim,
            max_position_embeddings,
            rope_freq_base,
            device,
        )?);

        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            layers.push(LayerWeights::new(
                &mut gg,
                i,
                num_attention_heads,
                num_kv_heads,
                head_dim,
                rms_norm_eps,
                rotary.clone(),
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
        let lm_head = QMatMul::from_weights(lm_head_tensor.into())?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            device: device.clone(),
            dtype,
        })
    }

    fn causal_mask(
        &self,
        b: usize,
        tgt: usize,
        offset: usize,
    ) -> Result<Tensor> {
        let minf = f32::NEG_INFINITY;
        let mask: Vec<_> = (0..tgt)
            .flat_map(|i| {
                (0..(tgt + offset)).map(move |j| {
                    if j <= i + offset {
                        0.
                    } else {
                        minf
                    }
                })
            })
            .collect();
        Tensor::from_slice(&mask, (b, 1, tgt, tgt + offset), &self.device)?.to_dtype(self.dtype)
    }

    pub fn forward(&mut self, input: &Tensor, offset: usize) -> Result<Tensor> {
        let (b, l) = input.dims2()?;
        let mut h = self.embed_tokens.forward(input)?;
        let causal_mask = if l == 1 {
            None
        } else {
            Some(self.causal_mask(b, l, offset)?)
        };
        // 分层计时（perf 调查用，开销可忽略）
        let mut gdn_mix = std::time::Duration::ZERO;
        let mut full_mix = std::time::Duration::ZERO;
        let mut mlp_t = std::time::Duration::ZERO;
        let mut n_gdn = 0usize;
        let mut n_full = 0usize;
        for layer in &mut self.layers {
            let mut kind = "";
            let (mut tm, mut tp) = (std::time::Duration::ZERO, std::time::Duration::ZERO);
            h = layer.forward(&h, causal_mask.as_ref(), offset, &mut kind, Some(&mut tm), Some(&mut tp))?;
            if kind == "gdn" { n_gdn += 1; gdn_mix += tm; } else { n_full += 1; full_mix += tm; }
            mlp_t += tp;
        }
        if std::env::var("Q35_PERF").is_ok() {
            tracing::info!(
                "[q35-layers] l={l} offset={offset} gdn({n_gdn})={gdn_mix:.1?} full({n_full})={full_mix:.1?} mlp={mlp_t:.1?}"
            );
        }
        let h = self.norm.forward(&h)?;
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

    /// kernel 的数学正确性（与 JSON 可复现的固定输入逐位对照）
    #[test]
    fn test_gated_delta_step_math() -> Result<()> {
        let dev = Device::Cpu;
        let (b, h, kd, vd) = (1usize, 1usize, 2usize, 2usize);
        let state = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], &dev)?.reshape((b, h, kd, vd))?;
        let gate = Tensor::new(&[0.0f32, 0.0], &dev)?.reshape((b, h, 1, kd))?;
        let q = Tensor::new(&[1.0f32, 0.0], &dev)?.reshape((b, h, 1, kd))?;
        let packed = Tensor::cat(&[&gate, &q], 3)?.reshape((b, h, 1, kd * 2))?;
        let k = Tensor::new(&[1.0f32, 0.0], &dev)?.reshape((b, h, 1, kd))?;
        let v = Tensor::new(&[0.5f32, 1.0], &dev)?.reshape((b, h, 1, vd))?;
        let g = Tensor::new(&[0.5f32], &dev)?.reshape((b, h, 1))?;
        let beta = Tensor::new(&[1.0f32], &dev)?.reshape((b, h, 1))?;

        let out = gated_delta_step(&state, &packed, &k, &v, &g, &beta)?;

        // 手算：S·g = [0.5,1,1.5,2]；kᵀ·S = [0.5,1]；δ=(v−kv)=[0,0]；
        //        S' = S·g；out = q·S'/√2 = [0.5,1]/√2
        let expect_state = [0.5f32, 1.0, 1.5, 2.0];
        let expect_out = [0.5f32 / 2f32.sqrt(), 1.0 / 2f32.sqrt()];
        let got_state = state.flatten_all()?.to_vec1::<f32>()?;
        let got_out = out.flatten_all()?.to_vec1::<f32>()?;
        for (got, want) in got_state.iter().zip(expect_state.iter()) {
            assert!((got - want).abs() < 1e-6, "state {got} != {want}");
        }
        for (got, want) in got_out.iter().zip(expect_out.iter()) {
            assert!((got - want).abs() < 1e-6, "out {got} != {want}");
        }
        Ok(())
    }

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
}
