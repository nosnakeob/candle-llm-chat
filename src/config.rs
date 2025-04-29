use crate::models::{FromGGUF, HubInfo};
use crate::utils::load::{load_gguf, load_tokenizer};
use anyhow::{Error, Result};
use candle::Device;
use tokenizers::Tokenizer;

#[derive(Debug, Clone)]
pub struct BaseConfig<W> {
    /// The length of the sample to generate (in tokens).
    pub sample_len: usize,

    /// The temperature used to generate samples, use 0 for greedy sampling.
    pub temperature: f64,

    /// Nucleus sampling probability cutoff.
    pub top_p: Option<f64>,

    /// Only sample among the top K samples.
    pub top_k: Option<usize>,

    /// The seed to use when generating random samples.
    pub seed: u64,

    /// The device to use for inference.
    pub device: Device,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    pub repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    pub repeat_last_n: usize,

    /// The model to use.
    pub which: W,
}

impl<W: Default> Default for BaseConfig<W> {
    fn default() -> Self {
        Self {
            sample_len: 1000,
            temperature: 0.8,
            top_p: None,
            top_k: None,
            seed: 299792458,
            device: candle_examples::device(false).unwrap(),
            repeat_penalty: 1.1,
            repeat_last_n: 64,
            which: W::default(),
        }
    }
}

impl<W: Default> From<W> for BaseConfig<W> {
    fn from(value: W) -> Self {
        Self {
            which: value,
            ..Default::default()
        }
    }
}

impl<Wi: HubInfo> BaseConfig<Wi> {
    pub async fn setup_model<W: FromGGUF>(&self) -> Result<W> {
        let info = self.which.info();
        let (mut file, model) = load_gguf(info.model_repo, info.model_file).await?;
        let model = W::from_gguf(model, &mut file, &self.device)?;

        Ok(model)
    }

    pub async fn setup_tokenizer(&self) -> Result<Tokenizer> {
        load_tokenizer(self.which.info().tokenizer_repo).await
    }
}
