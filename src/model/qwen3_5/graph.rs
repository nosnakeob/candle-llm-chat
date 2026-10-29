//! CUDA Graph 捕获 decode 一步
//!
//! 为什么：candle 每 token 提交约 306 个 kernel，GPU 实执行 1.2ms 而墙钟 ~30ms ——
//! 93% 的时间花在 CPU 侧提交（cuLaunchKernel + cuMemAllocAsync + 布局推导）。
//! 把整步捕获成一张图后，replay 只剩一次 cuGraphLaunch。
//!
//! 捕获的硬性前提（前面几步已经铺好）：
//! - 张量形状恒定：KV 走 FixedKvCache 读整段缓冲（src/model/qwen3_5/kv_buffer.rs）
//! - 不含 host 常量：位置、因果掩码全部由 device 张量参与运算（StepCtx / decode_mask）
//! - 输入是**静态缓冲**：token 与位置写进捕获前就已分配、捕获后不再重新分配的张量
//!
//! 输入写法：candle 0.11 没有公开的原地写接口（Tensor::slice_assign 返回新张量、
//! CudaStorage::as_cuda_slice_mut 取不到），只有 Tensor::scatter_set 是在地写，
//! 所以每步把一个 [1] 的小张量 scatter 进静态缓冲。这条写路径在**图外**执行，
//! 不进入被捕获的算子序列。
//!
//! 状态恢复：捕获时闭包会真跑一次前向 —— 它会推进 GDN 的循环状态。捕获方必须
//! 在捕获前快照、捕获后恢复（见 ModelWeights::snapshot_gdn_states）。KV 写入是
//! 原地写同一行，会被随后的真实 replay 覆盖，不需要恢复。
use candle::cuda_backend::cudarc::driver::{CudaGraph, CudaStream, sys};
use candle::{DType, Device, Result, Tensor};
use std::fmt;
use std::sync::Arc;

impl fmt::Debug for DecodeGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // CudaGraph 没有 Debug；模型结构体需要 Debug，这里只报存在性与形状。
        f.debug_struct("DecodeGraph")
            .field("input", &self.input.dims())
            .field("out", &self.out.dims())
            .finish_non_exhaustive()
    }
}

/// 排空 cudarc 记在 CudaContext::error_state 里的陈旧错误。
///
/// 由来：捕获期内被 drop 的中间张量会走 CudaSlice::drop → cuMemFreeAsync，而在捕获态下
/// 释放显存是非法的（CUDA_ERROR_INVALID_VALUE）。Drop 不能返回错误，cudarc 就把这条错误
/// 暂存进上下文；此后任何一次 CUDA 调用（alloc/memcpy/launch 都会先 bind_to_thread
/// → check_err）都会带着它失败一次，现象是「捕获之后连 Tensor::new 都建不出来」。
/// 排空只是一次原子交换，不产生 CUDA 调用。
fn drain_stale_ctx_error(dev: &Device) {
    if let Device::Cuda(d) = dev {
        if let Err(e) = d.cuda_stream().context().check_err() {
            tracing::warn!("[graph] 排空陈旧 CUDA 错误（捕获期 drop 的缓冲）: {e}");
        }
    }
}

/// 诊断：Q35_GRAPH_SYNC=1 时在关键点强制同步并把 CUDA 错误打出来（定位非法访问发生的时刻）
fn sync_probe_enabled() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| std::env::var("Q35_GRAPH_SYNC").is_ok_and(|v| v == "1"))
}

fn sync_probe(dev: &Device, what: &str) {
    if !sync_probe_enabled() {
        return;
    }
    match dev.synchronize() {
        Ok(()) => {
            if let Device::Cuda(d) = dev {
                if let Err(e) = d.cuda_stream().context().check_err() {
                    tracing::warn!("[graph-sync] {what} 同步后仍有错误: {e}");
                } else {
                    tracing::warn!("[graph-sync] {what} ok");
                }
            }
        }
        Err(e) => tracing::warn!("[graph-sync] {what} 同步失败: {e}"),
    }
}

/// 一整套 decode 步的捕获结果：静态输入缓冲 + 图 + 捕获时的输出张量
pub struct DecodeGraph {
    dev: Device,
    /// [1, 1] u32 的静态 token 缓冲，捕获前后地址不变
    input: Tensor,
    /// [1] u32 的静态位置缓冲
    pos: Tensor,
    /// [1] u32 全 0 索引，用于写 pos（scatter 要求 indexes 与 source 形状一致）
    idx0: Tensor,
    /// [1, 1] u32 全 0 索引，用于写 input
    idx00: Tensor,
    graph: CudaGraph,
    /// 捕获时那次前向的输出。它的存储由图的显存节点管理，replay 复写同一地址。
    out: Tensor,
}

impl DecodeGraph {
    /// 捕获一次 decode 步。`step` 内部必须只用静态形状、只用传进来的 input/pos 作位置来源。
    pub fn capture(
        dev: &Device,
        cap: usize,
        init_pos: u32,
        step: impl FnOnce(&Tensor, &Tensor, usize) -> Result<Tensor>,
    ) -> Result<Self> {
        let cuda_dev = match dev {
            Device::Cuda(d) => d.clone(),
            other => candle::bail!("CUDA Graph 需要 CUDA 设备，实际 {other:?}"),
        };
        let ctx = |what: &str, e: candle::Error| candle::Error::Msg(format!("capture/{what}: {e}"));
        let input = Tensor::zeros((1, 1), DType::U32, dev).map_err(|e| ctx("建 input", e))?;
        let pos = Tensor::zeros(1, DType::U32, dev).map_err(|e| ctx("建 pos", e))?;
        let idx0 = Tensor::zeros(1, DType::U32, dev).map_err(|e| ctx("建 idx0", e))?;
        let idx00 = Tensor::zeros((1, 1), DType::U32, dev).map_err(|e| ctx("建 idx00", e))?;
        // 捕获那一次也真跑前向：位置先设成真实行号，捕获写完的行随后会被 replay 覆盖。
        let init = Tensor::new(&[init_pos], dev).map_err(|e| ctx("建 init", e))?;
        pos.scatter_set(&idx0, &init, 0).map_err(|e| ctx("写初始位置", e))?;

        let stream: Arc<CudaStream> = cuda_dev.cuda_stream();
        // 捕获前先同步：残留的在飞 kernel 会让捕获失败
        dev.synchronize().map_err(|e| ctx("捕获前同步", e))?;
        stream
            .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
            .map_err(|e| candle::Error::Msg(format!("begin_capture: {e}")))?;
        // 实例化标志必须是 AUTO_FREE_ON_LAUNCH：图内每个 alloc 节点占的显存要在 launch
        // 结束时还回池子，否则第二次 cuGraphLaunch 直接返回 CUDA_ERROR_INVALID_VALUE
        // （实测：probe5 单算子图，flags=0 时 launch2 必失败，带此标志可无限重放）。
        // 代价：图内分配的缓冲（含 step 的返回值）在 launch 后即失效 —— 调用方必须
        // 在 replay 后立刻读，且中间不能再分配显存（见 ModelWeights::forward_graph 注释）。
        let flags = sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH;
        let out = match step(&input, &pos, cap) {
            Ok(out) => out,
            Err(e) => {
                // 失败必须收尾，否则流永久停在捕获态
                let _ = stream.end_capture(flags);
                return Err(e);
            }
        };
        let graph = stream
            .end_capture(flags)
            .map_err(|e| candle::Error::Msg(format!("end_capture: {e}")))?
            .ok_or_else(|| candle::Error::Msg("end_capture 返回 None：没有捕获到任何算子".into()))?;
        dev.synchronize().map_err(|e| ctx("实例化后同步", e))?;
        drain_stale_ctx_error(dev);
        Ok(Self {
            dev: dev.clone(),
            input,
            pos,
            idx0,
            idx00,
            graph,
            out,
        })
    }

    /// 把本步的 token 与位置写进静态缓冲（图外执行）
    pub fn set(&self, token: u32, pos: u32) -> Result<()> {
        drain_stale_ctx_error(&self.dev);
        let ctx = |what: &str, e: candle::Error| candle::Error::Msg(format!("graph set/{what}: {e}"));
        // 形状必须严格对齐 candle 的 scatter_checks：src 与 dest 同 rank、非 dim 维相等，
        // 且 indexes 与 src 形状完全一致。所以 input 用 [1,1] 源配 idx00，pos 用 [1] 源配 idx0。
        let t = Tensor::new(&[[token]], &self.dev).map_err(|e| ctx("建 token", e))?;
        self.input
            .scatter_set(&self.idx00, &t, 0)
            .map_err(|e| ctx("写 token", e))?;
        let p = Tensor::new(&[pos], &self.dev).map_err(|e| ctx("建 pos", e))?;
        self.pos
            .scatter_set(&self.idx0, &p, 0)
            .map_err(|e| ctx("写 pos", e))?;
        sync_probe(&self.dev, "graph.set 写入后");
        Ok(())
    }

    /// replay 一次，返回捕获时那个输出张量的句柄。
    ///
    /// 图的显存节点每次 launch 都复用同一地址，所以返回的句柄无需重新绑定；
    /// 调用方要在下一次 launch 前把值读走（同步一次）。
    pub fn replay(&self) -> Result<Tensor> {
        self.graph
            .launch()
            .map_err(|e| candle::Error::Msg(format!("graph launch: {e}")))?;
        sync_probe(&self.dev, "graph.launch 之后");
        Ok(self.out.clone())
    }

    /// 捕获时的输出张量（logits）
    pub fn out(&self) -> &Tensor {
        &self.out
    }
}