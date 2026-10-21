//! 测量环境探针：把 GPU 时钟状态和噪声水平量化出来
//!
//! 背景：本机 GPU 时钟随负载在 772MHz ~ 3060MHz 间浮动，同一段代码
//! 相隔几分钟重测可差一个量级。任何"优化了 N 倍"的结论都必须附带
//! 测量当时的时钟与离散度，否则是噪声。
use candle::{Device, Module, Tensor};
use candle::quantized::{GgmlDType, QMatMul, QTensor};
use std::process::Command;
use std::time::Instant;

fn sm_clock() -> String {
    let out = Command::new("nvidia-smi")
        .args(["--query-gpu=clocks.sm,utilization.gpu,temperature.gpu", "--format=csv,noheader"])
        .output();
    match out {
        Ok(o) => String::from_utf8_lossy(&o.stdout).trim().to_string(),
        Err(_) => "n/a".to_string(),
    }
}

/// 多次采样取中位，返回 (中位数, 最慢/最快比)
fn sample<F: FnMut()>(name: &str, n: usize, rounds: usize, dev: &Device, mut f: F) -> anyhow::Result<()> {
    // 预热：让 GPU 从空闲时钟升上来
    for _ in 0..20 { f(); }
    dev.synchronize()?;
    let mut per: Vec<f64> = Vec::with_capacity(rounds);
    for _ in 0..rounds {
        let t = Instant::now();
        for _ in 0..n { f(); }
        dev.synchronize()?;
        per.push(t.elapsed().as_secs_f64() * 1e6 / n as f64);
    }
    per.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = per[per.len() / 2];
    let ratio = per[per.len() - 1] / per[0];
    println!("{name:44} 中位 {med:8.2} µs   最快 {:8.2}   最慢 {:8.2}   离散比 {ratio:.1}x   [{}]", per[0], per[per.len() - 1], sm_clock());
    Ok(())
}

#[test]
fn measurement_noise_level() -> anyhow::Result<()> {
    let dev = Device::cuda_if_available(0)?;
    println!("GPU 初始状态: [{}]", sm_clock());
    let s = Tensor::randn(0f32, 1f32, (1, 16, 128, 128), &dev)?;
    let g_small = Tensor::randn(0f32, 1f32, (1, 16, 1, 1), &dev)?;
    let g_full = g_small.broadcast_as((1, 16, 128, 128))?.contiguous()?;

    sample("broadcast_mul (1,16,1,1)", 50, 7, &dev, || { let _ = s.broadcast_mul(&g_small).unwrap(); })?;
    sample("mul 预展开 (1,16,128,128)", 50, 7, &dev, || { let _ = (&s * &g_full).unwrap(); })?;
    sample("mul 同形状", 50, 7, &dev, || { let _ = (&s * &s).unwrap(); })?;
    sample("小算子+同步", 20, 7, &dev, || { let _ = (&s + &s).unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap(); })?;

    // 注意：不能用同进程内另一个线程压 GPU —— 那样测到的是资源竞争。
    // 高频状态需要外部进程（run_both.sh）拉起来，这里只做短测量。
    let tag = std::env::var("BENCH_TAG").unwrap_or_else(|_| "无外部负载".into());
    println!("\n--- {tag} [{}] ---", sm_clock());
    sample("broadcast_mul", 50, 5, &dev, || { let _ = s.broadcast_mul(&g_small).unwrap(); })?;
    sample("mul 预展开", 50, 5, &dev, || { let _ = (&s * &g_full).unwrap(); })?;
    sample("小算子+同步", 20, 5, &dev, || { let _ = (&s + &s).unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap(); })?;

    // ── 验证我做的两项优化在可靠测量下的真实收益 ──────────────────────
    println!("\n--- 优化收益复核 [{}] ---", sm_clock());

    // 1) conv1d：旧实现（k 次移位累加） vs 新实现（单次行内点积）
    let (b, c, k) = (1usize, 6144usize, 4usize);
    let x = Tensor::randn(0f32, 1f32, (b, c, k), &dev)?;
    let w = Tensor::randn(0f32, 1f32, (c, 1, k), &dev)?;
    let w2 = w.reshape((c, k))?;
    sample("conv1d 旧（4 次移位乘加）", 20, 5, &dev, || {
        let l_out = 1usize;
        let mut acc = x.narrow(2, 0, l_out).unwrap().broadcast_mul(&w2.narrow(1, 0, 1).unwrap().reshape((1, c, 1)).unwrap()).unwrap();
        for i in 1..k {
            let xi = x.narrow(2, i, l_out).unwrap();
            let wi = w2.narrow(1, i, 1).unwrap().reshape((1, c, 1)).unwrap();
            acc = (&acc + &xi.broadcast_mul(&wi).unwrap()).unwrap();
        }
    })?;
    sample("conv1d 新（单次行内点积）", 20, 5, &dev, || {
        let xs = x.reshape((b, c, k)).unwrap();
        let _ = (&xs * &w2.reshape((1, c, k)).unwrap()).unwrap().sum(2).unwrap();
    })?;

    // 2) repeat penalty：旧（整表回读） vs 新（index_select + index_add）
    let logits = Tensor::randn(0f32, 1f32, (248320,), &dev)?;
    let ids: Vec<u32> = (0..64).map(|i| i * 7).collect();
    let idx = Tensor::from_vec(ids.clone(), ids.len(), &dev)?;
    sample("repeat_penalty 旧（整表 to_vec1）", 20, 5, &dev, || {
        let v = logits.to_vec1::<f32>().unwrap();
        let _ = v.len();
    })?;
    sample("repeat_penalty 新（select+add）", 20, 5, &dev, || {
        let picked = logits.index_select(&idx, 0).unwrap().to_vec1::<f32>().unwrap();
        let d = Tensor::from_vec(picked.clone(), picked.len(), &dev).unwrap();
        let _ = logits.index_add(&idx, &d, 0).unwrap();
    })?;

    // 3) 参考：Q4K matvec（decode 实际路径）
    let xq = Tensor::randn(0f32, 1f32, (1, 1, 2048), &dev)?;
    let wq = Tensor::randn(0f32, 1f32, (3584, 2048), &dev)?;
    let qm: QTensor = QTensor::quantize(&wq, GgmlDType::Q4K)?;
    let mm = QMatMul::from_qtensor(qm)?;
    sample("Q4K matvec (1x1x2048)", 20, 5, &dev, || { let _ = Module::forward(&mm, &xq).unwrap(); })?;
    Ok(())
}
