use super::{HubInfo, HubModelInfo};
use crate::impl_model_traits;
use candle_transformers::models::quantized_llama::ModelWeights;

impl_model_traits!(ModelWeights);

#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub enum Which {
    #[default]
    SmolLM2_1BInstruct,
    // 聊天模板输出错误
    DeepseekR1Llama8b,
}

impl HubInfo for Which {
    type ModelWeight = ModelWeights;

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
