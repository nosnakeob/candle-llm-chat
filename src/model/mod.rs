use anyhow::Result;
use candle::quantized::gguf_file::Content;
use candle::{Device, Tensor};
use candle_transformers::models::{quantized_llama, quantized_qwen3, qwen3};
use std::io::{Read, Seek};

pub mod config;
pub mod hub;
pub mod registry;

macro_rules! impl_model_traits {
    ($($model:ty),+ $(,)?) => {
        $(
            impl crate::model::Forward for $model {
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

impl_model_traits!(
    quantized_llama::ModelWeights,
    quantized_qwen3::ModelWeights,
    qwen3::ModelForCausalLM
);
