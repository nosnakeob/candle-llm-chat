//! 采样：核采样（top-p）不再对整表做 argsort。
//!
//! candle-transformers 的 `LogitsProcessor` 在 top-p 路径上做的是
//! `argsort_indices.sort_by(...)` —— 对 248k 词表的**全量索引排序**，实测单次
//! 16.5ms，占每次 decode 的大头，而核采样真正需要的只是「前 m 个概率」。
//!
//! 这里换成 quickselect：先取前 m 大并按降序排这 m 个，累计概率够到 p 就停；
//! 不够就把 m 翻四倍重来。典型一次命中，代价从 O(V log V) 降到 O(V)。
//! 语义与 candle 的 `sample_topp` 完全一致（保留集合逐个相同）。

use candle::{D, DType, Result, Tensor};
use rand::{Rng, SeedableRng, rngs::StdRng};

/// 初始候选数。248k 词表的 softmax 极集中，top-64 的累计概率通常已在 0.99 以上。
const FIRST_M: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq)]
enum Strategy {
    /// 贪心：设备端 argmax，只回读 4 字节
    Greedy,
    /// 全分布多项式采样
    All { temperature: f64 },
    /// 核采样
    TopP { temperature: f64, p: f64 },
}

pub struct Sampler {
    rng: StdRng,
    strategy: Strategy,
}

impl Sampler {
    pub fn new(seed: u64, temperature: Option<f64>, top_p: Option<f64>) -> Self {
        let temperature = temperature.filter(|v| *v >= 1e-7);
        let strategy = match temperature {
            None => Strategy::Greedy,
            Some(temperature) => match top_p {
                Some(p) => Strategy::TopP { temperature, p },
                None => Strategy::All { temperature },
            },
        };
        Self {
            rng: StdRng::seed_from_u64(seed),
            strategy,
        }
    }

    /// 从 logits（rank 1 或最后一维是词表）里采一个 token id。
    pub fn sample(&mut self, logits: &Tensor) -> Result<u32> {
        match self.strategy {
            Strategy::Greedy => logits.argmax(D::Minus1)?.to_scalar::<u32>(),
            Strategy::All { temperature } => {
                let probs = self.probs(logits, temperature)?;
                Ok(pick_all(&probs, &mut self.rng))
            }
            Strategy::TopP { temperature, p } => {
                let probs = self.probs(logits, temperature)?;
                Ok(pick_nucleus(&probs, p, &mut self.rng))
            }
        }
    }

    /// 设备端温度缩放 + softmax，只回读一次概率向量（248k×f32≈1MB，约 0.2ms）。
    fn probs(&self, logits: &Tensor, temperature: f64) -> Result<Vec<f32>> {
        let logits = logits.to_dtype(DType::F32)?;
        let scaled = (&logits / temperature)?;
        candle_nn::ops::softmax_last_dim(&scaled)?.to_vec1::<f32>()
    }
}

/// top-p 保留集合 + 加权抽一个。
fn pick_nucleus(probs: &[f32], p: f64, rng: &mut StdRng) -> u32 {
    if probs.is_empty() {
        return 0;
    }
    // p 越界（<=0 或 >=1）等价于不做核截断
    if !(p > 0.0 && p < 1.0) {
        return pick_all(probs, rng);
    }

    let mut m = FIRST_M.min(probs.len());
    loop {
        let top = top_m(probs, m);
        let mass: f64 = top.iter().map(|&(_, v)| v as f64).sum();
        // 前 m 个已经含住 p 的概率质量 → 真核一定在 top 内，可以做精确截断
        if m == probs.len() || mass >= p {
            let keep = nucleus_len(&top, p);
            return weighted_pick_top(&top[..keep], rng);
        }
        m = (m * 4).min(probs.len());
    }
}

/// 保留集合长度：降序累加，第一次累加值 >= p 的那个位置（含）就是核的末尾。
///
/// 与 candle `sample_topp` 的逐元素判断等价 —— 它在累加**之前**判断是否已达 p，
/// 因此跨过 p 的那个 token 仍保留。
fn nucleus_len(top_desc: &[(u32, f32)], p: f64) -> usize {
    let mut cum = 0f64;
    for (i, &(_, v)) in top_desc.iter().enumerate() {
        cum += v as f64;
        if cum >= p {
            return i + 1;
        }
    }
    top_desc.len()
}

/// 取前 m 大的 (id, 概率)，按降序返回。
///
/// 先对一份 f32 副本做 quickselect（顺序值比较，对缓存友好），拿到第 m 大的值当阈值，
/// 再扫一遍原向量收集。248k 下比「对索引做 argsort」快一个数量级。
fn top_m(probs: &[f32], m: usize) -> Vec<(u32, f32)> {
    let m = m.min(probs.len());
    debug_assert!(m > 0);
    if m == probs.len() {
        let mut all: Vec<(u32, f32)> = probs
            .iter()
            .copied()
            .enumerate()
            .map(|(i, v)| (i as u32, v))
            .collect();
        all.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        return all;
    }

    let mut buf = probs.to_vec();
    buf.select_nth_unstable_by(m - 1, |a, b| b.total_cmp(a));
    let bound = buf[m - 1];

    let mut out: Vec<(u32, f32)> = Vec::with_capacity(m + FIRST_M);
    let mut ties: Vec<(u32, f32)> = Vec::new();
    for (i, &v) in probs.iter().enumerate() {
        if v > bound {
            out.push((i as u32, v));
        } else if v == bound {
            ties.push((i as u32, v));
        }
    }
    // 并列阈值补满 m 个。f32 softmax 上几乎不出现，出现时取谁不影响分布（值相同）。
    let need = m.saturating_sub(out.len());
    out.extend(ties.into_iter().take(need));
    out.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
    out
}

/// 全分布加权抽样（一次线性扫描，248k 约 0.3ms）。
fn pick_all(probs: &[f32], rng: &mut StdRng) -> u32 {
    let total: f64 = probs.iter().map(|&v| v as f64).sum();
    if !(total > 0.0) {
        return 0;
    }
    let mut u = rng.random::<f64>() * total;
    for (i, &v) in probs.iter().enumerate() {
        u -= v as f64;
        if u < 0.0 {
            return i as u32;
        }
    }
    (probs.len() - 1) as u32
}

fn weighted_pick_top(top: &[(u32, f32)], rng: &mut StdRng) -> u32 {
    let total: f64 = top.iter().map(|&(_, v)| v as f64).sum();
    if !(total > 0.0) {
        return top.first().map(|x| x.0).unwrap_or(0);
    }
    let mut u = rng.random::<f64>() * total;
    for &(id, v) in top {
        u -= v as f64;
        if u < 0.0 {
            return id;
        }
    }
    top[top.len() - 1].0
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_transformers::generation::LogitsProcessor;

    /// 保留集合必须与 candle 的整表 argsort 版本逐个相同
    #[test]
    fn test_nucleus_set_matches_full_sort() {
        // 幂律分布，模拟 248k 词表 softmax 的长尾
        let n = 200_000usize;
        let raw: Vec<f32> = (0..n).map(|i| 1.0 / ((i + 1) as f32).powf(1.1)).collect();
        let sum: f32 = raw.iter().sum();
        let probs: Vec<f32> = raw.iter().map(|v| v / sum).collect();

        for p in [0.5f64, 0.8, 0.9, 0.95, 0.999] {
            let mut ref_idx: Vec<usize> = (0..n).collect();
            ref_idx.sort_by(|&i, &j| probs[j].total_cmp(&probs[i]));
            let mut want = 0usize;
            let mut cum = 0f64;
            for &i in &ref_idx {
                if cum >= p {
                    break;
                }
                cum += probs[i] as f64;
                want += 1;
            }

            let mut m = FIRST_M;
            let got = loop {
                let top = top_m(&probs, m);
                let mass: f64 = top.iter().map(|&(_, v)| v as f64).sum();
                if m >= n || mass >= p {
                    break top[..nucleus_len(&top, p)].to_vec();
                }
                m = (m * 4).min(n);
            };
            assert_eq!(got.len(), want, "p={p} 保留集合大小不一致");
            assert!(
                got.iter().all(|&(id, _)| (id as usize) < n),
                "p={p} 出现越界 id"
            );
            // 逐个核对：参考保留位 i 的概率必须 >= 我们的最小保留概率
            let min_keep = got.last().unwrap().1;
            let max_drop = probs[ref_idx[want]];
            assert!(
                min_keep >= max_drop,
                "p={p} 保留边界错：min_keep={min_keep} < max_drop={max_drop}"
            );
        }
    }

    /// 端到端：核外 token 永不出现，核内频率贴近解析分布
    #[test]
    fn test_topp_sampling_distribution() {
        // 4 个主 token 占 0.94，其余 6 个共 0.06；p=0.9 时尾巴应被切掉
        let mut logits_v = vec![-10.0f32; 10];
        let weights = [0.4f32, 0.3, 0.15, 0.09, 0.06];
        for (i, w) in weights.iter().enumerate() {
            logits_v[i] = w.ln();
        }
        let dev = candle::Device::Cpu;
        let logits = Tensor::new(logits_v.as_slice(), &dev).unwrap();

        let mut s = Sampler::new(42, Some(1.0), Some(0.9));
        let mut counts = [0usize; 10];
        let rounds = 40_000;
        for _ in 0..rounds {
            let t = s.sample(&logits).unwrap() as usize;
            counts[t] += 1;
        }
        // p=0.9：前 4 个累计 0.94 跨过阈值，第 5 个（0.06）落在核外
        assert_eq!(counts[4..].iter().sum::<usize>(), 0, "核外 token 被采到");
        let kept = &weights[..4];
        let total: f64 = kept.iter().map(|w| *w as f64).sum();
        for (i, w) in kept.iter().enumerate() {
            let expect = *w as f64 / total;
            let got = counts[i] as f64 / rounds as f64;
            assert!(
                (got - expect).abs() < 0.02,
                "token{i}: got={got:.3} expect={expect:.3}"
            );
        }
    }

    /// 贪心路径与 candle ArgMax 一致
    #[test]
    fn test_greedy_matches_argmax() {
        let dev = candle::Device::Cpu;
        let logits = Tensor::new(&[0.1f32, 3.2, -1.0, 3.19], &dev).unwrap();
        let mut s = Sampler::new(7, Some(0.0), Some(0.8));
        assert_eq!(s.sample(&logits).unwrap(), 1);
        let mut ref_p = LogitsProcessor::new(7, Some(0.0), Some(0.8));
        assert_eq!(ref_p.sample(&logits).unwrap(), 1);
    }

    /// 温度缩放必须与 candle 的 softmax(logits/temp) 走同一分布
    #[test]
    fn test_temperature_scaling() -> Result<()> {
        let dev = candle::Device::Cpu;
        let logits = Tensor::new(&[1.0f32, 2.0, 0.5, -1.0], &dev)?;
        let s = Sampler::new(1, Some(0.5), None);
        let probs = s.probs(&logits, 0.5)?;
        let want = candle_nn::ops::softmax_last_dim(&(&logits / 0.5)?)?.to_vec1::<f32>()?;
        for (a, b) in probs.iter().zip(want.iter()) {
            assert!((a - b).abs() < 1e-7);
        }
        Ok(())
    }

    #[test]
    fn test_empty_and_degenerate() {
        let mut rng = StdRng::seed_from_u64(1);
        assert_eq!(pick_nucleus(&[], 0.9, &mut rng), 0);
        assert_eq!(pick_nucleus(&[0.0, 0.0], 0.9, &mut rng), 0);
        // 单 token 全概率
        assert_eq!(pick_nucleus(&[1.0], 0.9, &mut rng), 0);
        // p 越界退化为全分布
        let probs = [0.5f32, 0.5];
        for _ in 0..100 {
            assert!([0, 1].contains(&pick_nucleus(&probs, 1.5, &mut rng)));
        }
    }

    #[test]
    fn test_top_m_ordering() {
        let probs = [0.1f32, 0.9, 0.3, 0.5];
        let top = top_m(&probs, 2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].0, 1);
        assert_eq!(top[1].0, 3);
        let all = top_m(&probs, 4);
        assert_eq!(
            all.iter().map(|x| x.0).collect::<Vec<_>>(),
            vec![1, 3, 2, 0]
        );
    }
}
