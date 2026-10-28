//! CUDA Graph 可行性探针：验证 candle 0.11 + cudarc 0.17.8 在当前机器能否 capture/replay。
//!
//! 两个用例：
//! - launch-bound：算子小、GPU 执行时间可忽略，墙钟几乎全是 CPU 提交开销 —— 对应真实 decode（306 kernel/token，GPU 实执行 1.2ms）；
//! - compute-bound：GEMM 大，GPU 执行占主导 —— 用来看 graph 在高负载下会不会反而损失（图内不允许并行下发）。
//!
//! 用法：cargo run --release --example graph_probe

use candle::cuda_backend::cudarc::driver::{CudaStream, sys};
use candle::{DType, Device, Result, Tensor};
use std::sync::Arc;
use std::time::Instant;

const LAYERS: usize = 18;
const WARMUP: usize = 5;
const ITERS: usize = 50;

struct Weights {
    up: Vec<Tensor>,
    down: Vec<Tensor>,
    gate: Vec<Tensor>,
}

fn build_weights(dev: &Device, dtype: DType, hidden: usize, inter: usize) -> Result<Weights> {
    let mut up = Vec::with_capacity(LAYERS);
    let mut down = Vec::with_capacity(LAYERS);
    let mut gate = Vec::with_capacity(LAYERS);
    for _ in 0..LAYERS {
        up.push(Tensor::randn(0f64, 1f64, (hidden, inter), dev)?.to_dtype(dtype)?);
        down.push(Tensor::randn(0f64, 1f64, (inter, hidden), dev)?.to_dtype(dtype)?);
        gate.push(Tensor::randn(0f64, 1f64, (1, inter), dev)?.to_dtype(dtype)?);
    }
    Ok(Weights { up, down, gate })
}

fn forward(x: &Tensor, w: &Weights) -> Result<Tensor> {
    let mut h = x.clone();
    for i in 0..LAYERS {
        let a = h.matmul(&w.up[i])?; // cublas GEMM
        let a = candle_nn::ops::silu(&a)?; // 逐元素
        let a = a.broadcast_mul(&w.gate[i])?; // 广播乘（GDN 里的典型形态）
        let a = a.matmul(&w.down[i])?; // cublas GEMM
        h = (&h + &a)?; // 残差加
    }
    Ok(h)
}

/// 默认内存池的 release_threshold 抬到 u64::MAX：避免捕获期间异步释放把块还给系统。
unsafe fn pin_default_mempool(ordinal: i32) -> std::result::Result<(), String> {
    unsafe {
        let mut pool: sys::CUmemoryPool = std::ptr::null_mut();
        let r = sys::cuDeviceGetDefaultMemPool(&mut pool, ordinal);
        if r != sys::CUresult::CUDA_SUCCESS {
            return Err(format!("cuDeviceGetDefaultMemPool: {r:?}"));
        }
        let mut v: u64 = u64::MAX;
        let r = sys::cuMemPoolSetAttribute(
            pool,
            sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
            &mut v as *mut u64 as *mut std::ffi::c_void,
        );
        if r != sys::CUresult::CUDA_SUCCESS {
            return Err(format!("cuMemPoolSetAttribute: {r:?}"));
        }
    }
    Ok(())
}

fn run_case(
    name: &str,
    dev: &Device,
    cuda_dev: &candle::CudaDevice,
    hidden: usize,
    inter: usize,
) -> Result<()> {
    let stream: Arc<CudaStream> = cuda_dev.cuda_stream();
    let dtype = DType::BF16;
    let w = build_weights(dev, dtype, hidden, inter)?;
    let x = Tensor::randn(0f64, 1f64, (1, hidden), dev)?.to_dtype(dtype)?;

    for _ in 0..WARMUP {
        let y = forward(&x, &w)?;
        let _ = y.to_dtype(DType::F32)?.to_vec2::<f32>()?;
    }

    let t0 = Instant::now();
    for _ in 0..ITERS {
        let y = forward(&x, &w)?;
        dev.synchronize()?;
        let _ = y;
    }
    let eager_ms = t0.elapsed().as_secs_f64() * 1000.0 / ITERS as f64;

    let guard = cuda_dev.enable_cuda_graph_htod_cache();
    stream
        .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
        .map_err(|e| candle::Error::Msg(format!("begin_capture: {e}")))?;
    let cap_out = forward(&x, &w);
    let cap_err = cap_out.as_ref().err().map(|e| e.to_string());
    let status = stream.capture_status();
    let graph = match stream
        .end_capture(sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)
    {
        Ok(Some(g)) => Some(g),
        Ok(None) => None,
        Err(e) => {
            println!("  end_capture 失败: {e}");
            None
        }
    };
    drop(guard);
    dev.synchronize()?;

    let graph_ms = match graph {
        Some(graph) => {
            for _ in 0..WARMUP {
                graph
                    .launch()
                    .map_err(|e| candle::Error::Msg(format!("launch: {e}")))?;
                dev.synchronize()?;
            }
            let t0 = Instant::now();
            for _ in 0..ITERS {
                graph
                    .launch()
                    .map_err(|e| candle::Error::Msg(format!("launch: {e}")))?;
                dev.synchronize()?;
            }
            Some(t0.elapsed().as_secs_f64() * 1000.0 / ITERS as f64)
        }
        None => None,
    };

    println!("\n--- {name}  hidden={hidden} inter={inter}  {LAYERS} 层 x 6 算子 ---");
    println!("  捕获 err={cap_err:?} status={status:?}");
    println!("  逐次提交 : {eager_ms:.3} ms/iter");
    match graph_ms {
        Some(ms) => println!("  graph    : {ms:.3} ms/iter   加速 {:.2}x", eager_ms / ms),
        None => println!("  graph    : 捕获失败"),
    }
    Ok(())
}

fn main() -> Result<()> {
    let dev = Device::cuda_if_available(0)?;
    let cuda_dev = match &dev {
        Device::Cuda(d) => d.clone(),
        _ => panic!("探针需要 CUDA 设备"),
    };
    println!("device = {dev:?}");
    // 设 GRAPH_PROBE_NOPIN=1 可跳过抬阈值，用来实证这一步到底必要不必要
    if std::env::var("GRAPH_PROBE_NOPIN").is_ok() {
        println!("pin mempool = SKIPPED (GRAPH_PROBE_NOPIN)");
    } else {
        println!("pin mempool = {:?}", unsafe { pin_default_mempool(0) });
    }
    run_case("launch-bound", &dev, &cuda_dev, 256, 512)?;
    run_case("compute-bound", &dev, &cuda_dev, 1280, 3584)?;
    Ok(())
}
