use anyhow::Result;
use candle_core::Device;
use candle_core::quantized::gguf_file;
use candle_transformers::models::quantized_llama::ModelWeights;
use hf_hub::api::tokio::Api;
use tokenizers::Tokenizer;
use tokio::fs::File;

use crate::utils::format_size;

#[derive(Debug, Clone)]
pub struct Args {
    pub sample_len: usize,
    pub temperature: f64,
    pub seed: u64,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
    pub device: Device,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            sample_len: 1000,
            temperature: 0.8,
            seed: 299792458,
            repeat_penalty: 1.1,
            repeat_last_n: 64,
            device: Device::new_cuda(0).unwrap_or(Device::Cpu),
        }
    }
}

impl Args {
    pub async fn tokenizer(&self) -> Result<Tokenizer> {
        let tokenizer_path = Api::new()?.model("openchat/openchat_3.5".to_string())
            .get("tokenizer.json").await?;

        Tokenizer::from_file(tokenizer_path).map_err(anyhow::Error::msg)
    }

    pub async fn model(&self) -> Result<ModelWeights> {
        let (repo, filename) = ("TheBloke/openchat_3.5-GGUF", "openchat_3.5.Q2_K.gguf");

        let model_path = Api::new()?.model(repo.to_string()).get(filename).await?;
        let mut file = File::open(&model_path).await?.into_std().await;
        let start = std::time::Instant::now();

        // This is the model instance
        let model = gguf_file::Content::read(&mut file)?;
        let mut total_size_in_bytes = 0;
        for (_, tensor) in model.tensor_infos.iter() {
            let elem_count = tensor.shape.elem_count();
            total_size_in_bytes +=
                elem_count * tensor.ggml_dtype.type_size() / tensor.ggml_dtype.block_size();
        }
        info!(
            "loaded {:?} tensors ({}) in {:.2}s",
            model.tensor_infos.len(),
            &format_size(total_size_in_bytes),
            start.elapsed().as_secs_f32(),
        );

        ModelWeights::from_gguf(model, &mut file, &self.device).map_err(anyhow::Error::msg)
    }
}

