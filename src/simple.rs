#![allow(unused)]

#[macro_use]
extern crate tracing;

use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::time;
use tokenizers::Tokenizer;

use candle_core::quantized::gguf_file;
use candle_core::{cuda, quantized, utils};
use candle_core::{Device, Tensor};
use candle_transformers::generation::LogitsProcessor;

use anyhow::Result;
use candle_core::utils::cuda_is_available;
use candle_transformers::models::quantized_llama::ModelWeights;
use candle_transformers::utils::apply_repeat_penalty;
use hf_hub::api::tokio::Api;
use hf_hub::{Repo, RepoType};
use token_output_stream::TokenOutputStream;

mod token_output_stream;
mod bot;

#[derive(Debug, Clone)]
struct Args {
    // tokenizer: String,
    // model: String,
    sample_len: usize,
    temperature: f64,
    seed: u64,
    repeat_penalty: f32,
    repeat_last_n: usize,
    gqa: usize,
    device: Device,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            sample_len: 1000,
            temperature: 0.8,
            seed: 299792458,
            repeat_penalty: 1.1,
            repeat_last_n: 64,
            gqa: 8,
            device: Device::new_cuda(0).unwrap_or(Device::Cpu),
        }
    }
}

impl Args {
    async fn tokenizer(&self) -> Result<Tokenizer> {
        let api = Api::new()?;

        let tokenizer_path = api.model("openchat/openchat_3.5".to_string())
            .get("tokenizer.json").await?;

        Tokenizer::from_file(tokenizer_path).map_err(anyhow::Error::msg)
    }

    async fn model(&self) -> Result<PathBuf> {
        let (repo, filename) = ("TheBloke/openchat_3.5-GGUF", "openchat_3.5.Q2_K.gguf");

        let api = Api::new()?;

        Ok(api.model(repo.to_string())
            .get(filename).await?)
    }
}

fn format_size(size_in_bytes: usize) -> String {
    if size_in_bytes < 1_000 {
        format!("{}B", size_in_bytes)
    } else if size_in_bytes < 1_000_000 {
        format!("{:.2}KB", size_in_bytes as f64 / 1e3)
    } else if size_in_bytes < 1_000_000_000 {
        format!("{:.2}MB", size_in_bytes as f64 / 1e6)
    } else {
        format!("{:.2}GB", size_in_bytes as f64 / 1e9)
    }
}

async fn load_model(args: &Args) -> Result<ModelWeights> {
    // load model
    let model_path = args.model().await?;
    let mut file = File::open(&model_path)?;
    let start = std::time::Instant::now();

    // This is the model instance
    let model = gguf_file::Content::read(&mut file)?;
    let mut total_size_in_bytes = 0;
    for (_, tensor) in model.tensor_infos.iter() {
        let elem_count = tensor.shape.elem_count();
        total_size_in_bytes +=
            elem_count * tensor.ggml_dtype.type_size() / tensor.ggml_dtype.block_size();
    }
    println!(
        "loaded {:?} tensors ({}) in {:.2}s",
        model.tensor_infos.len(),
        &format_size(total_size_in_bytes),
        start.elapsed().as_secs_f32(),
    );
    let model = ModelWeights::from_gguf(model, &mut file, &args.device)?;
    println!("model built");

    Ok(model)
}


#[tokio::main]
async fn main() -> Result<()> {
    quantized::cuda::set_force_dmmv(false);

    cuda::set_gemm_reduced_precision_f16(true);
    cuda::set_gemm_reduced_precision_bf16(true);

    println!(
        "avx: {}, neon: {}, simd128: {}, f16c: {}",
        utils::with_avx(),
        utils::with_neon(),
        utils::with_simd128(),
        utils::with_f16c()
    );

    let args = Args::default();

    let device = &args.device;

    let mut model = load_model(&args).await?;

    // load tokenizer
    let tokenizer = args.tokenizer().await?;
    let mut tos = TokenOutputStream::new(tokenizer);

    let eos_token = tos.tokenizer().token_to_id("<|end_of_turn|>").unwrap();

    let mut logits_processor = LogitsProcessor::new(args.seed, Some(args.temperature), None);


    // left for future improvement: interactive
    loop {
        print!("> ");
        // 把缓冲区字符串刷到控制台上
        std::io::stdout().flush()?;
        let mut prompt = String::new();
        std::io::stdin().read_line(&mut prompt)?;

        let prompt_str = format!("User: {} <|end_of_turn|> Assistant: ", prompt.trim());
        print!("bot: ");

        let tokens = tos
            .tokenizer()
            .encode(prompt_str, true)
            .map_err(anyhow::Error::msg)?;

        let prompt_tokens = tokens.get_ids();
        let mut all_tokens = vec![];

        let start_prompt_processing = time::Instant::now();
        let mut next_token = {
            let input = Tensor::new(prompt_tokens, device)?.unsqueeze(0)?;
            let logits = model.forward(&input, 0)?;
            let logits = logits.squeeze(0)?;
            logits_processor.sample(&logits)?
        };
        let prompt_dt = start_prompt_processing.elapsed();
        all_tokens.push(next_token);
        if let Some(t) = tos.next_token(next_token)? {
            print!("{t}");
            std::io::stdout().flush()?;
        }

        let start_post_prompt = time::Instant::now();
        let mut sampled = 1;
        while sampled < args.sample_len {
            let input = Tensor::new(&[next_token], device)?.unsqueeze(0)?;
            let logits = model.forward(&input, prompt_tokens.len() + sampled)?;
            let logits = logits.squeeze(0)?;
            let logits = if args.repeat_penalty == 1. {
                logits
            } else {
                let start_at = all_tokens.len().saturating_sub(args.repeat_last_n);
                apply_repeat_penalty(
                    &logits,
                    args.repeat_penalty,
                    &all_tokens[start_at..],
                )?
            };
            next_token = logits_processor.sample(&logits)?;
            all_tokens.push(next_token);
            if let Some(t) = tos.next_token(next_token)? {
                print!("{t}");
                std::io::stdout().flush()?;
            }
            sampled += 1;
            if next_token == eos_token {
                break;
            };
        }
        if let Some(rest) = tos.decode_rest().map_err(candle_core::Error::msg)? {
            print!("{rest}");
        }
        std::io::stdout().flush()?;
        let dt = start_post_prompt.elapsed();
        println!(
            "\n\n{:4} prompt tokens processed: {:.2} token/s",
            prompt_tokens.len(),
            prompt_tokens.len() as f64 / prompt_dt.as_secs_f64(),
        );
        println!(
            "{sampled:4} tokens generated: {:.2} token/s",
            sampled as f64 / dt.as_secs_f64(),
        );
    }

    Ok(())
}
