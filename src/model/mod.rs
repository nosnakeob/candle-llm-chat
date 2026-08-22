use anyhow::Result;
use candle::quantized::gguf_file::Content;
use candle::{Device, Tensor};
use candle_transformers::models::{quantized_qwen3, quantized_qwen3_moe, qwen3};
use std::io::{Read, Seek};pub mod config;
pub mod hub;
pub mod qwen3_5;
pub mod qwen3_vl;
pub mod registry;

/// 视觉模型的配置信息（非视觉模型返回 None）
#[derive(Debug, Clone)]
pub struct VisionConfig {
    pub patch_size: usize,
    pub spatial_merge_size: usize,
    pub temporal_patch_size: usize,
    pub image_token_id: u32,
}

/// 视觉输入数据，传递给 forward 的 vision 参数
pub struct VisionInput {
    pub pixel_values: Tensor,
    pub image_grid_thw: Tensor,
    pub image_token_span: Vec<Vec<(usize, usize)>>,
}

macro_rules! impl_model_traits {
    ($($model:ty),+ $(,)?) => {
        $(
            impl crate::model::ModelInference for $model {
                fn forward(
                    &mut self,
                    x: &candle::Tensor,
                    index_pos: usize,
                    _vision: Option<&crate::model::VisionInput>,
                ) -> anyhow::Result<candle::Tensor> {
                    self.forward(x, index_pos).map_err(anyhow::Error::msg)
                }

                fn clr_kv_cache(&mut self) {
                    self.clear_kv_cache();
                }
            }
        )+
    };
}

pub trait ModelInference {
    fn forward(&mut self, x: &Tensor, index_pos: usize, vision: Option<&VisionInput>) -> Result<Tensor>;

    fn clr_kv_cache(&mut self);

    /// 如果是视觉模型，返回视觉配置信息
    fn vision_config(&self) -> Option<&VisionConfig> {
        None
    }
}

impl_model_traits!(
    quantized_qwen3::ModelWeights,
    qwen3::ModelForCausalLM
);

/// GGUFQWenMoE 的封装，提供 clr_kv_cache
pub struct Qwen3MoEModel {
    pub(crate) inner: quantized_qwen3_moe::GGUFQWenMoE,
}

impl ModelInference for Qwen3MoEModel {
    fn forward(
        &mut self,
        x: &Tensor,
        index_pos: usize,
        _vision: Option<&VisionInput>,
    ) -> anyhow::Result<Tensor> {
        self.inner.forward(x, index_pos).map_err(anyhow::Error::msg)
    }

    fn clr_kv_cache(&mut self) {
        // 上游 GGUFQWenMoE 的 kv_cache 字段私有且未暴露 reset（截至 0.11/main），
        // 需向 candle 提 PR 暴露 clear_kv_cache 后才能实现。
        // 临时方案：多轮对话间重建 pipeline。
    }
}
