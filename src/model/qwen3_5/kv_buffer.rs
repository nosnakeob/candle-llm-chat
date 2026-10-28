//! 固定容量 KV 缓冲：CUDA Graph 的前置条件
//!
//! 为什么需要它：`ConcatKvCache` 每步 `Tensor::cat`，形状随 token 数增长，
//! graph 捕获一次就固定了形状，replay 只能复用同一形状。改成"预分配 max_seq、
//! 按位置写入"，形状恒定，位置由 **device 张量** 给出。
//!
//! 关键约束（踩过的坑，写在这里避免重犯）：
//! 写入位置不能是 host 值。`slice_assign(&[(pos..pos+1), …])` 里的 `pos` 是 host
//! 传入的整数，捕获时会被烘进图，replay 永远写同一行。位置必须以 device 张量
//! 参与运算（`pos_t` + arange 的 broadcast_add），这样每次 replay 才知道写哪。
use candle::{D, DType, Device, Result, Tensor};

/// 单层 KV 的固定容量缓冲，形状 `[max_seq, n_kv_head, head_dim]`
#[derive(Debug)]
pub struct FixedKvCache {
    k: Tensor,
    v: Tensor,
    max_seq: usize,
    n_kv_head: usize,
    head_dim: usize,
    dtype: DType,
    device: Device,
}

impl FixedKvCache {
    pub fn new(
        max_seq: usize,
        n_kv_head: usize,
        head_dim: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        Ok(Self {
            k: Tensor::zeros((max_seq, n_kv_head, head_dim), dtype, device)?,
            v: Tensor::zeros((max_seq, n_kv_head, head_dim), dtype, device)?,
            max_seq,
            n_kv_head,
            head_dim,
            dtype,
            device: device.clone(),
        })
    }

    pub fn max_seq(&self) -> usize {
        self.max_seq
    }

    /// 把 `[seq, n_kv_head, head_dim]` 的 k/v 写到第 `pos_t` 行开始的位置。
    ///
    /// `pos_t` 是 rank-0 的 device 张量（u32/i64 皆可），不是 host 整数 —— 见模块头。
    /// `seq` 必须是 host 常量（graph 捕获时形状固定），所以它是 usize 参数。
    pub fn write_at(&mut self, pos_t: &Tensor, k_new: &Tensor, v_new: &Tensor) -> Result<()> {
        let seq = k_new.dim(0)?;
        if k_new.dims() != [seq, self.n_kv_head, self.head_dim] {
            candle::bail!(
                "k_new 形状应为 [{seq}, {}, {}]，实际 {:?}",
                self.n_kv_head,
                self.head_dim,
                k_new.dims()
            )
        }
        let idx = self.row_index(pos_t, seq)?;
        // broadcast 出来的索引是 stride-0 视图，scatter 只接受 contiguous
        let idx = idx
            .broadcast_as((seq, self.n_kv_head, self.head_dim))?
            .contiguous()?;
        self.k.scatter_set(&idx, k_new, 0)?;
        self.v.scatter_set(&idx, v_new, 0)?;
        Ok(())
    }

    /// 图内用：`[seq, n_kv_head, head_dim]` 的行号张量（未 broadcast）
    fn row_index(&self, pos_t: &Tensor, seq: usize) -> Result<Tensor> {
        let offsets = Tensor::arange(0u32, seq as u32, pos_t.device())?;
        let pos = pos_t.to_dtype(DType::U32)?.reshape(())?;
        pos.broadcast_add(&offsets)?.reshape((seq, 1, 1))
    }

    /// eager 用：读出前 `len` 行，等价于 `ConcatKvCache::append` 的返回值
    ///
    /// 注意 `narrow` 返回的是视图，调用方若要 `contiguous` 会多一次拷贝；
    /// 走图时不要用这个函数，直接用整段缓冲 + mask。
    pub fn narrow(&self, len: usize) -> Result<(Tensor, Tensor)> {
        if len > self.max_seq {
            candle::bail!("len {len} 超过容量 {}", self.max_seq)
        }
        Ok((self.k.narrow(0, 0, len)?, self.v.narrow(0, 0, len)?))
    }

    /// 整段缓冲（形状恒定），供图内 attention 使用
    pub fn full(&self) -> (&Tensor, &Tensor) {
        (&self.k, &self.v)
    }

    pub fn reset(&mut self) -> Result<()> {
        self.k = Tensor::zeros(
            (self.max_seq, self.n_kv_head, self.head_dim),
            self.dtype,
            &self.device,
        )?;
        self.v = Tensor::zeros(
            (self.max_seq, self.n_kv_head, self.head_dim),
            self.dtype,
            &self.device,
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// CPU 与（可用时）CUDA 各跑一遍：scatter_set 的两套后端实现不同，
    /// 只在 CPU 上绿没有意义 —— 真正跑图的是 CUDA。
    fn test_devices() -> Vec<Device> {
        let mut devs = vec![Device::Cpu];
        if let Ok(d) = Device::cuda_if_available(0) {
            if d.is_cuda() {
                devs.push(d);
            }
        }
        devs
    }

    /// 与"逐步 cat"的参考实现逐元素对齐：这是替换 ConcatKvCache 的等价性门禁
    #[test]
    fn write_at_matches_concat() -> Result<()> {
        for dev in test_devices() {
            write_at_matches_concat_on(&dev)?;
        }
        Ok(())
    }

    fn write_at_matches_concat_on(dev: &Device) -> Result<()> {
        let (max_seq, h, d) = (8usize, 2usize, 4usize);
        let mut cache = FixedKvCache::new(max_seq, h, d, DType::F32, dev)?;

        let mut ref_k: Option<Tensor> = None;
        let mut ref_v: Option<Tensor> = None;

        for pos in 0..5usize {
            // 每次写 1 行，值随位置变化，写错行会被逐元素比对抓出来
            let base = pos as f32 * 100.0;
            let k = Tensor::arange(0f32, (h * d) as f32, dev)?
                .reshape((1, h, d))?
                .affine(1.0, base as f64)?;
            let v = k.neg()?;
            let pos_t = Tensor::new(pos as u32, dev)?;
            cache.write_at(&pos_t, &k, &v)?;

            ref_k = Some(match ref_k {
                None => k.clone(),
                Some(prev) => Tensor::cat(&[&prev, &k], 0)?,
            });
            ref_v = Some(match ref_v {
                None => v.clone(),
                Some(prev) => Tensor::cat(&[&prev, &v], 0)?,
            });
        }

        let (got_k, got_v) = cache.narrow(5)?;
        let want_k = ref_k.unwrap();
        let want_v = ref_v.unwrap();
        assert_eq!(got_k.dims(), want_k.dims());
        let dk = (got_k - want_k)?.abs()?.sum_all()?.to_scalar::<f32>()?;
        let dv = (got_v - want_v)?.abs()?.sum_all()?.to_scalar::<f32>()?;
        assert!(dk < 1e-6 && dv < 1e-6, "k diff {dk}, v diff {dv}");
        Ok(())
    }

    /// 多行一次写入（prefill 形状）
    #[test]
    fn write_at_multi_row() -> Result<()> {
        for dev in test_devices() {
            write_at_multi_row_on(&dev)?;
        }
        Ok(())
    }

    fn write_at_multi_row_on(dev: &Device) -> Result<()> {
        let (max_seq, h, d) = (8usize, 1usize, 2usize);
        let mut cache = FixedKvCache::new(max_seq, h, d, DType::F32, dev)?;
        let k = Tensor::new(&[[[1f32, 2.]], [[3., 4.]], [[5., 6.]]], dev)?;
        let v = k.affine(2.0, 0.0)?;
        let pos_t = Tensor::new(2u32, dev)?;
        cache.write_at(&pos_t, &k, &v)?;
        let (gk, _) = cache.narrow(8)?;
        // 第 2..5 行应是写入值，其余为 0
        let row0 = gk.narrow(0, 0, 2)?.sum_all()?.to_scalar::<f32>()?;
        let row_written = gk.narrow(0, 2, 3)?.sum_all()?.to_scalar::<f32>()?;
        assert_eq!(row0, 0.0);
        assert_eq!(row_written, 21.0);
        Ok(())
    }
}
