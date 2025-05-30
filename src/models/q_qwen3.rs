use crate::impl_model_traits;
use crate::models::{HubInfo, HubModelInfo};
use candle_transformers::models::quantized_qwen3::ModelWeights;

impl_model_traits!(ModelWeights);

#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub enum Which {
    W3_0_6b,
    W3_1_7b,
    W3_4b,
    #[default]
    W3_8b,
    W3_14b,
    W3_32b,
}

impl HubInfo for Which {
    type ModelWeight = ModelWeights;

    fn info(&self) -> HubModelInfo {
        match self {
            Which::W3_0_6b => HubModelInfo {
                model_repo: "unsloth/Qwen3-0.6B-GGUF",
                model_file: "Qwen3-0.6B-Q4_K_M",
                tokenizer_repo: "Qwen/Qwen3-0.6B",
            },
            Which::W3_1_7b => HubModelInfo {
                model_repo: "unsloth/Qwen3-1.7B-GGUF",
                model_file: "Qwen3-1.7B-Q4_K_M",
                tokenizer_repo: "Qwen/Qwen3-1.7B",
            },
            Which::W3_4b => HubModelInfo {
                model_repo: "Qwen/Qwen3-4B-GGUF",
                model_file: "Qwen3-4B-Q4_K_M",
                tokenizer_repo: "Qwen/Qwen3-4B",
            },
            Which::W3_8b => HubModelInfo {
                model_repo: "Qwen/Qwen3-8B-GGUF",
                model_file: "Qwen3-8B-Q4_K_M",
                tokenizer_repo: "Qwen/Qwen3-8B",
            },
            Which::W3_14b => HubModelInfo {
                model_repo: "Qwen/Qwen3-14B-GGUF",
                model_file: "Qwen3-14B-Q4_K_M",
                tokenizer_repo: "Qwen/Qwen3-14B",
            },
            Which::W3_32b => HubModelInfo {
                model_repo: "Qwen/Qwen3-32B-GGUF",
                model_file: "Qwen3-32B-Q4_K_M",
                tokenizer_repo: "Qwen/Qwen3-32B",
            },
        }
    }
}
