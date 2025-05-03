use anyhow::Result;
use candle::quantized::gguf_file::Content;
use candle::{Device, Tensor};
use std::io::{Read, Seek};

pub mod q_llama;
pub mod q_qwen2;

#[derive(Debug, Clone)]
pub struct HubModelInfo {
    pub model_repo: &'static str,
    pub model_file: &'static str,
    pub tokenizer_repo: &'static str,
    // todo 从仓库中获取
    pub eos_token: &'static str,
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
