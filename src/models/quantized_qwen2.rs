extern crate intel_mkl_src;

use crate::utils::{format_size, setup_logits_processor};
use std::path::PathBuf;
use std::process::Command;

use async_stream::try_stream;

use futures_core::stream::Stream;

use tokenizers::Tokenizer;

use hf_hub::api::tokio::Api;

use anyhow::{bail, Error, Result};
use candle::quantized::gguf_file;
use candle::{Device, Tensor};
use candle_transformers::generation::LogitsProcessor;

use candle_transformers::models::quantized_qwen2::ModelWeights;
use candle_transformers::utils::apply_repeat_penalty;
use hf_hub::{Cache, Repo};

#[allow(unused)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Which {
    // gpu推理报错
    W2_0_5b,
    W2_1_5b,
    W2_7b,
    W2_72b,
    W25_0_5b,
    W25_1_5b,
    W25_7b,
    W25_14b,
    W25_32b,
}

impl Which {
    // 返回tokenizer, model repo, filename
    pub fn info(&self) -> (&'static str, &'static str, &'static str, &'static str) {
        let tokenizer_fname = "tokenizer.json";
        match self {
            Which::W2_0_5b => (
                "Qwen/Qwen2-0.5B-Instruct",
                tokenizer_fname,
                "Qwen/Qwen2-0.5B-Instruct-GGUF",
                "qwen2-0_5b-instruct-q4_0.gguf",
            ),
            Which::W2_1_5b => (
                "Qwen/Qwen2-1.5B-Instruct",
                tokenizer_fname,
                "Qwen/Qwen2-1.5B-Instruct-GGUF",
                "qwen2-1_5b-instruct-q4_0.gguf",
            ),
            Which::W2_7b => (
                "Qwen/Qwen2-7B-Instruct",
                tokenizer_fname,
                "Qwen/Qwen2-7B-Instruct-GGUF",
                "qwen2-7b-instruct-q4_0.gguf",
            ),
            Which::W2_72b => (
                "Qwen/Qwen2-72B-Instruct",
                tokenizer_fname,
                "Qwen/Qwen2-72B-Instruct-GGUF",
                "qwen2-72b-instruct-q4_0.gguf",
            ),
            Which::W25_0_5b => (
                "Qwen/Qwen2.5-0.5B-Instruct",
                tokenizer_fname,
                "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
                "qwen2.5-0.5b-instruct-q4_0.gguf",
            ),
            Which::W25_1_5b => (
                "Qwen/Qwen2.5-1.5B-Instruct",
                tokenizer_fname,
                "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
                "qwen2.5-1.5b-instruct-q4_0.gguf",
            ),
            Which::W25_7b => (
                "Qwen/Qwen2.5-7B-Instruct",
                tokenizer_fname,
                "Qwen/Qwen2.5-7B-Instruct-GGUF",
                "qwen2.5-7b-instruct-q4_0.gguf",
            ),
            Which::W25_14b => (
                "Qwen/Qwen2.5-14B-Instruct",
                tokenizer_fname,
                "Qwen/Qwen2.5-14B-Instruct-GGUF",
                "qwen2.5-14b-instruct-q4_0.gguf",
            ),
            Which::W25_32b => (
                "Qwen/Qwen2.5-32B-Instruct",
                tokenizer_fname,
                "Qwen/Qwen2.5-32B-Instruct-GGUF",
                "qwen2.5-32b-instruct-q4_0.gguf",
            ),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Args {
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
    which: Which,
}

impl Args {
    pub async fn tokenizer(&self) -> Result<PathBuf> {
        let (repo, filename, _, _) = self.which.info();
        Api::new()?
            .model(repo.to_string())
            .get(filename)
            .await
            .map_err(Error::msg)
    }

    pub async fn model(&self) -> Result<PathBuf> {
        let (_, _, repo, filename) = self.which.info();

        let model_path = if let Some(path) = Cache::default()
            .repo(Repo::model(repo.to_string()))
            .get(filename)
        {
            path
        } else {
            let repo = Api::new()?.model(repo.to_string());

            let split_filenames: Vec<String> = repo
                .info()
                .await?
                .siblings
                .into_iter()
                .map(|sibling| sibling.rfilename)
                .filter(|s| s.starts_with(filename.strip_suffix(".gguf").unwrap()))
                .collect();

            let mut split_paths = vec![];
            for filename in &split_filenames {
                split_paths.push(repo.get(filename).await?);
            }

            // 获取输出目录
            let output_dir = split_paths[0].parent().unwrap();
            // println!("{:?}", output_dir);

            let merged_path = output_dir.join(filename);

            // 构建命令
            let exe_path = which::which("llama-gguf-split")?;
            let mut command = Command::new(exe_path);
            command
                .arg("--merge")
                .arg(split_paths[0].to_str().unwrap())
                .arg(merged_path.to_str().unwrap());

            // println!("{:?}", command);

            // 执行命令
            let output = command.output()?;

            if !output.status.success() {
                let error = String::from_utf8_lossy(&output.stderr);
                bail!("llama-gguf-split failed: {}", error)
            }

            let merged_path = output_dir.join(filename);
            merged_path
        };

        Ok(model_path)
    }
}

impl Default for Args {
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
            which: Which::W25_7b,
        }
    }
}

pub(crate) async fn setup_model(args: &Args) -> Result<ModelWeights> {
    let model_path = args.model().await?;
    let mut file = std::fs::File::open(&model_path)?;
    let start = std::time::Instant::now();

    // 构建模型
    let model = {
        let model = gguf_file::Content::read(&mut file).map_err(|e| e.with_path(&model_path))?;
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
        ModelWeights::from_gguf(model, &mut file, &args.device)?
    };

    Ok(model)
}

pub(crate) async fn setup_tokenizer(args: &Args) -> Result<Tokenizer> {
    let pth = args.tokenizer().await?;
    Tokenizer::from_file(pth).map_err(Error::msg)
}

pub(crate) fn fmt_prompt(prompt: &str) -> String {
    format!(
        "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
        prompt
    )
}

pub(crate) fn token2str(token: u32, tokenizer: &Tokenizer) -> Result<String> {
    tokenizer.decode(&[token], true).map_err(Error::msg)
}

// 一次对话
pub struct TextGeneration {
    model: ModelWeights,
    tokenizer: Tokenizer,
    logits_processor: LogitsProcessor,
    args: Args,
    ctx_tokens: Vec<u32>,
    eos_token: u32,
}

impl TextGeneration {
    pub async fn new(args: Args) -> Result<Self> {
        let tokenizer = setup_tokenizer(&args).await?;
        let eos_token = *tokenizer.get_vocab(true).get("<|im_end|>").unwrap();

        Ok(Self {
            model: setup_model(&args).await?,
            tokenizer,
            logits_processor: setup_logits_processor(
                args.temperature,
                args.seed,
                args.top_k,
                args.top_p,
            ),
            args,
            ctx_tokens: Vec::with_capacity(1024),
            eos_token,
        })
    }

    pub fn chat<'a>(&'a mut self, prompt: &'a str) -> impl Stream<Item = Result<String>> + 'a {
        try_stream! {
            let prompt_tokens = self.process_prompt(prompt)?;
            self.ctx_tokens.extend_from_slice(&prompt_tokens);

            let start = std::time::Instant::now();

            // 生成第一个token
            let mut next_token = self.gen_next_token(0, None)?;
            let ans_start_idx = self.ctx_tokens.len();
            self.ctx_tokens.push(next_token);

            if let Ok(t) = token2str(next_token, &self.tokenizer) {
                yield t;
            }

            // 循环生成回答
            for index in 0..self.args.sample_len.saturating_sub(1) {
                next_token = self.gen_next_token(ans_start_idx + index, Some(ans_start_idx))?;
                self.ctx_tokens.push(next_token);

                if let Ok(t) = token2str(next_token, &self.tokenizer) {
                    yield t;
                }

                if next_token == self.eos_token {
                    break;
                }
            }

            // 在生成速度统计前添加换行
            yield "\n".to_string();

            let dt = start.elapsed();
            info!( "speed: {:.2} token/s",
                (self.ctx_tokens.len() - ans_start_idx) as f64 / dt.as_secs_f64(),
            );
        }
    }

    fn process_prompt(&self, prompt: &str) -> Result<Vec<u32>> {
        // 格式化提示词
        let prompt = fmt_prompt(&prompt);

        // 将提示词转换为token
        let tokens = self.tokenizer.encode(prompt, true).map_err(Error::msg)?;
        let tokens = tokens.get_ids().to_vec();

        Ok(tokens)
    }

    fn gen_next_token(&mut self, idx_pos: usize, ans_start_idx: Option<usize>) -> Result<u32> {
        let logits = if let Some(ans_start_idx) = ans_start_idx {
            let input = Tensor::new(&[*self.ctx_tokens.last().unwrap()], &self.args.device)?
                .unsqueeze(0)?;
            let mut logits = self.model.forward(&input, idx_pos)?;
            logits = logits.squeeze(0)?;

            if self.args.repeat_penalty != 1. {
                let ans_tokens = &self.ctx_tokens[ans_start_idx..];
                let start_at = ans_tokens.len().saturating_sub(self.args.repeat_last_n);
                logits = apply_repeat_penalty(
                    &logits,
                    self.args.repeat_penalty,
                    &ans_tokens[start_at..],
                )?
            };

            logits
        } else {
            let input = Tensor::new(&*self.ctx_tokens, &self.args.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, idx_pos)?;
            logits.squeeze(0)?
        };

        self.logits_processor.sample(&logits).map_err(Error::msg)
    }
}
