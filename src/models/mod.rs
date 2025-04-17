use anyhow::Result;
use candle::Device;
use tokenizers::Tokenizer;

pub mod quantized_qwen2;

#[derive(Debug, Clone)]
pub struct BaseConfig<W: Default> {
    /// The length of the sample to generate (in tokens).
    pub(crate) sample_len: usize,

    /// The temperature used to generate samples, use 0 for greedy sampling.
    pub(crate) temperature: f64,

    /// Nucleus sampling probability cutoff.
    pub(crate) top_p: Option<f64>,

    /// Only sample among the top K samples.
    pub(crate) top_k: Option<usize>,

    /// The seed to use when generating random samples.
    pub(crate) seed: u64,

    pub(crate) device: Device,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    pub(crate) repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    pub(crate) repeat_last_n: usize,

    /// The model size to use.
    pub(crate) which: W,
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

pub trait Setup {
    type Weight;

    async fn setup_model(&self) -> Result<Self::Weight>;

    async fn setup_tokenizer(&self) -> Result<Tokenizer>;
}