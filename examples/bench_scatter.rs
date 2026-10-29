use candle::{DType, Device, Result, Tensor};
use std::time::Instant;

fn main() -> Result<()> {
    let dev = Device::cuda_if_available(0)?;
    for n in [18432usize, 262144] {
        let dst = Tensor::zeros(n, DType::F32, &dev)?;
        let idx = Tensor::arange(0u32, n as u32, &dev)?;
        let src = Tensor::ones(n, DType::F32, &dev)?;
        // 预热
        for _ in 0..3 {
            dst.scatter_set(&idx, &src, 0)?;
        }
        dev.synchronize()?;
        let t = Instant::now();
        let reps = 20;
        for _ in 0..reps {
            dst.scatter_set(&idx, &src, 0)?;
        }
        dev.synchronize()?;
        let per = t.elapsed().as_secs_f64() * 1e3 / reps as f64;
        println!("scatter_set n={n}: {per:.3} ms/次");

        // 对照：同形状的 copy（会分配）与 add
        let t = Instant::now();
        for _ in 0..reps {
            let _ = (&dst + &src)?;
        }
        dev.synchronize()?;
        println!(
            "  add n={n}: {:.3} ms/次",
            t.elapsed().as_secs_f64() * 1e3 / reps as f64
        );
    }
    Ok(())
}
