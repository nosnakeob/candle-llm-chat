pub mod chunked_scan;
pub mod dense;
pub mod graph;
pub mod kv_buffer;
pub mod quantized;

pub use dense::{Config, Config as Qwen3_5Config, ModelForCausalLM};
pub use quantized::ModelWeights as QuantizedModelWeights;

use crate::model::ModelInference;
use crate::model::VisionInput;
use candle::Tensor;

impl ModelInference for ModelForCausalLM {
    fn forward(
        &mut self,
        x: &Tensor,
        index_pos: usize,
        _vision: Option<&VisionInput>,
    ) -> anyhow::Result<Tensor> {
        self.forward(x, index_pos).map_err(anyhow::Error::msg)
    }

    fn reset_kv_cache(&mut self) -> anyhow::Result<()> {
        self.clear_kv_cache();
        Ok(())
    }
}

impl ModelInference for QuantizedModelWeights {
    fn forward(
        &mut self,
        x: &Tensor,
        index_pos: usize,
        _vision: Option<&VisionInput>,
    ) -> anyhow::Result<Tensor> {
        self.forward(x, index_pos).map_err(anyhow::Error::msg)
    }

    fn reset_kv_cache(&mut self) -> anyhow::Result<()> {
        self.clear_kv_cache();
        Ok(())
    }

    fn supports_graph(&self) -> bool {
        // 图捕获只对 CUDA 有意义；CPU 上退回 eager
        self::quantized::graph_enabled() && self.device().is_cuda()
    }

    fn forward_token(&mut self, token: u32, index_pos: usize) -> anyhow::Result<Tensor> {
        self.forward_graph(token, index_pos)
            .map_err(|e| anyhow::anyhow!("forward_graph: {e}"))
    }
}
