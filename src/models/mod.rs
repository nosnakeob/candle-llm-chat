use anyhow::Result;
use candle::quantized::gguf_file::Content;
use candle::{Device, Tensor};
use std::io::{Read, Seek};

pub mod q_llama;
pub mod q_qwen2;
pub mod q_qwen3;

/// 实现模型 trait 的宏
#[macro_export]
macro_rules! impl_model_traits {
    ($weight:ty) => {
        impl crate::models::Forward for $weight {
            fn forward(
                &mut self,
                x: &candle::Tensor,
                index_pos: usize,
            ) -> anyhow::Result<candle::Tensor> {
                self.forward(x, index_pos).map_err(anyhow::Error::msg)
            }
        }

        impl crate::models::FromGGUF for $weight {
            fn from_gguf<R: std::io::Seek + std::io::Read>(
                ct: candle::quantized::gguf_file::Content,
                reader: &mut R,
                device: &candle::Device,
            ) -> anyhow::Result<Self> {
                Self::from_gguf(ct, reader, device).map_err(anyhow::Error::msg)
            }
        }
    };
}

#[derive(Debug, Clone)]
pub struct HubModelInfo {
    pub model_repo: &'static str,
    pub model_file: &'static str,
    pub tokenizer_repo: &'static str,
}

pub trait HubInfo {
    // 添加关联类型，指定该配置对应的模型权重类型
    type ModelWeight: FromGGUF + Forward;

    fn info(&self) -> HubModelInfo;
}

pub trait FromGGUF {
    fn from_gguf<R: Seek + Read>(ct: Content, reader: &mut R, device: &Device) -> Result<Self>
    where
        Self: Sized;
}

pub trait Forward {
    fn forward(&mut self, x: &Tensor, index_pos: usize) -> Result<Tensor>;
}
