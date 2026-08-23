pub mod dense;
pub mod quantized;

pub use dense::{Config, Config as Qwen3_5Config, ModelForCausalLM};
pub use quantized::ModelWeights as QuantizedModelWeights;

use crate::model::VisionInput;
use crate::model::ModelInference;
use candle::{Device, Tensor};

impl ModelInference for ModelForCausalLM {
    fn forward(
        &mut self,
        x: &Tensor,
        index_pos: usize,
        _vision: Option<&VisionInput>,
    ) -> anyhow::Result<Tensor> {
        self.forward(x, index_pos).map_err(anyhow::Error::msg)
    }

    fn clr_kv_cache(&mut self) {
        self.clear_kv_cache();
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

    fn clr_kv_cache(&mut self) {
        self.clear_kv_cache();
    }
}
