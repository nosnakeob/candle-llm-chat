//! 最小复现: candle CUDA conv1d 性能问题
//! cargo test --release --test conv1d_bench -- --ignored --nocapture
use candle::{Device, Tensor};
use std::time::Instant;

#[test]
#[ignore]
fn conv1d_cuda_bench() -> anyhow::Result<()> {
    let dev = Device::new_cuda(0)?;
    let (c, k, seq) = (3072usize, 4usize, 4usize); // decode 形状

    // 模拟 GDN 的 conv1d_weight: dequantize 后 unsqueeze(1), F32
    let w = Tensor::randn(0f32, 1f32, (c, 1, k), &dev)?;
    let x = Tensor::randn(0f32, 1f32, (1, c, seq), &dev)?;

    // 预热
    for _ in 0..3 {
        let _ = x.conv1d(&w, 0, 1, 1, c)?;
    }
    dev.synchronize()?;

    // 连续输入
    let t = Instant::now();
    for _ in 0..10 {
        let _ = x.conv1d(&w, 0, 1, 1, c)?;
    }
    dev.synchronize()?;
    println!("连续输入   : {:?}/次", t.elapsed() / 10);

    // 非连续输入（模拟 transpose 后的 mixed_qkv）
    let xt = Tensor::randn(0f32, 1f32, (1, seq, c), &dev)?.transpose(1, 2)?;
    let t = Instant::now();
    for _ in 0..10 {
        let _ = xt.conv1d(&w, 0, 1, 1, c)?;
    }
    dev.synchronize()?;
    println!("非连续输入 : {:?}/次", t.elapsed() / 10);

    // 对照: 同尺寸 matmul
    let a = Tensor::randn(0f32, 1f32, (1, seq * k), &dev)?;
    let b = Tensor::randn(0f32, 1f32, (seq * k, c), &dev)?;
    let t = Instant::now();
    for _ in 0..10 {
        let _ = a.matmul(&b)?;
    }
    dev.synchronize()?;
    println!("对照 matmul: {:?}/次", t.elapsed() / 10);

    Ok(())
}
