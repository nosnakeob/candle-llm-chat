extern crate intel_mkl_src;

use crate::models::{Forward, FromGGUF, HubInfo, HubModelInfo};
use anyhow::{Error, Result};
use candle::quantized::gguf_file::Content;
use candle::{Device, Tensor};
use candle_transformers::models::quantized_qwen2::ModelWeights;
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

/// 支持模型型号
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub enum Which {
    // gpu推理报错
    W2_0_5b,
    W2_1_5b,
    W2_7b,
    W2_72b,
    W25_0_5b,
    W25_1_5b,
    #[default]
    W25_7b,
    W25_14b,
    W25_32b,
    DeepseekR1Qwen7B,
}

impl HubInfo for Which {
    type ModelWeight = ModelWeights;

    fn info(&self) -> HubModelInfo {
        let (qwen_eos, ds_eos) = ("<|im_end|>", "<｜end▁of▁sentence｜>");
        match self {
            Which::W2_0_5b => HubModelInfo {
                model_repo: "Qwen/Qwen2-0.5B-Instruct-GGUF",
                model_file: "qwen2-0_5b-instruct-q4_0",
                tokenizer_repo: "Qwen/Qwen2-0.5B-Instruct",
                eos_token: qwen_eos,
            },
            Which::W2_1_5b => HubModelInfo {
                model_repo: "Qwen/Qwen2-1.5B-Instruct-GGUF",
                model_file: "qwen2-1_5b-instruct-q4_0",
                tokenizer_repo: "Qwen/Qwen2-1.5B-Instruct",
                eos_token: qwen_eos,
            },
            Which::W2_7b => HubModelInfo {
                model_repo: "Qwen/Qwen2-7B-Instruct-GGUF",
                model_file: "qwen2-7b-instruct-q4_0",
                tokenizer_repo: "Qwen/Qwen2-7B-Instruct",
                eos_token: qwen_eos,
            },
            Which::W2_72b => HubModelInfo {
                model_repo: "Qwen/Qwen2-72B-Instruct-GGUF",
                model_file: "qwen2-72b-instruct-q4_0",
                tokenizer_repo: "Qwen/Qwen2-72B-Instruct",
                eos_token: qwen_eos,
            },
            Which::W25_0_5b => HubModelInfo {
                model_repo: "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
                model_file: "qwen2.5-0.5b-instruct-q4_0",
                tokenizer_repo: "Qwen/Qwen2.5-0.5B-Instruct",
                eos_token: qwen_eos,
            },
            Which::W25_1_5b => HubModelInfo {
                model_repo: "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
                model_file: "qwen2.5-1_5b-instruct-q4_0",
                tokenizer_repo: "Qwen/Qwen2.5-1.5B-Instruct",
                eos_token: qwen_eos,
            },
            Which::W25_7b => HubModelInfo {
                model_repo: "Qwen/Qwen2.5-7B-Instruct-GGUF",
                model_file: "qwen2.5-7b-instruct-q4_0",
                tokenizer_repo: "Qwen/Qwen2.5-7B-Instruct",
                eos_token: qwen_eos,
            },
            Which::W25_14b => HubModelInfo {
                model_repo: "Qwen/Qwen2.5-14B-Instruct-GGUF",
                model_file: "qwen2.5-14b-instruct-q4_0",
                tokenizer_repo: "Qwen/Qwen2.5-14B-Instruct",
                eos_token: qwen_eos,
            },
            Which::W25_32b => HubModelInfo {
                model_repo: "Qwen/Qwen2.5-32B-Instruct-GGUF",
                model_file: "qwen2.5-32b-instruct-q4_0",
                tokenizer_repo: "Qwen/Qwen2.5-32B-Instruct",
                eos_token: qwen_eos,
            },
            Which::DeepseekR1Qwen7B => HubModelInfo {
                model_repo: "unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF",
                model_file: "DeepSeek-R1-Distill-Qwen-7B-Q4_K_M",
                tokenizer_repo: "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                eos_token: ds_eos,
            },
        }
    }
}
