extern crate intel_mkl_src;

use super::{Forward, FromGGUF, HubInfo, HubModelInfo};
use anyhow::{Error, Result};
use candle::quantized::gguf_file::Content;
use candle::{Device, Tensor};
use candle_transformers::models::quantized_llama::ModelWeights;
use std::io::{Read, Seek};

impl Forward for ModelWeights {
    fn forward(&mut self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        self.forward(x, index_pos).map_err(Error::msg)
    }
}

impl FromGGUF for ModelWeights {
    fn from_gguf<R: Seek + Read>(ct: Content, reader: &mut R, device: &Device) -> Result<Self> {
        ModelWeights::from_gguf(ct, reader, device).map_err(Error::msg)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub enum Which {
    #[default]
    SmolLM2_1BInstruct,
    // 聊天模板输出错误
    DeepseekR1Llama8b,
}

impl HubInfo for Which {
    fn info(&self) -> HubModelInfo {
        match self {
            Which::SmolLM2_1BInstruct => HubModelInfo {
                model_repo: "HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF",
                model_file: "smollm2-1.7b-instruct-q4_k_m",
                tokenizer_repo: "HuggingFaceTB/SmolLM2-1.7B-Instruct",
                eos_token: "<|im_end|>",
            },
            Which::DeepseekR1Llama8b => HubModelInfo {
                model_repo: "lmstudio-community/DeepSeek-R1-Distill-Llama-8B-GGUF",
                model_file: "DeepSeek-R1-Distill-Llama-8B-Q4_K_M",
                tokenizer_repo: "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                eos_token: "<｜end▁of▁sentence｜>",
            },
        }
    }
}
