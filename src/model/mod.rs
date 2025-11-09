use anyhow::Result;
use candle::quantized::gguf_file::Content;
use candle::{Device, Tensor};
use candle_transformers::models::{quantized_llama, quantized_qwen2, quantized_qwen3};
use std::io::{Read, Seek};

pub mod config;
pub mod registry;
pub mod hub;

/// 实现模型 trait 的宏
macro_rules! impl_model_traits {
    // 支持多个模块标识符，自动添加 ::ModelWeights
    ($($module:ident),+ $(,)?) => {
        $(
            impl crate::model::Forward for $module::ModelWeights {
                fn forward(
                    &mut self,
                    x: &candle::Tensor,
                    index_pos: usize,
                ) -> anyhow::Result<candle::Tensor> {
                    self.forward(x, index_pos).map_err(anyhow::Error::msg)
                }
            }
        )+
    };
}

pub trait Forward {
    fn forward(&mut self, x: &Tensor, index_pos: usize) -> Result<Tensor>;
}

// 一次性为多个模型实现 Forward trait
impl_model_traits!(quantized_llama, quantized_qwen2, quantized_qwen3);
