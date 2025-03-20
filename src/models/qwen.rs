extern crate intel_mkl_src;

use crate::utils::format_size;
use std::path::PathBuf;
use tokenizers::Tokenizer;

use hf_hub::api::tokio::Api;

use anyhow::{Error, Result};
use candle::quantized::gguf_file;
use candle_transformers::generation::{LogitsProcessor, Sampling};

use candle_examples::token_output_stream::TokenOutputStream;
use candle_transformers::models::quantized_qwen2::ModelWeights as Qwen2;

const DEFAULT_PROMPT: &str = "编写一个函数来计算小于等于N的质数数量。";

#[allow(unused)]
#[derive(Clone, Debug, Copy, PartialEq, Eq)]
enum Which {
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

#[derive(Debug)]
struct Args {
    /// The length of the sample to generate (in tokens).
    sample_len: usize,

    /// The temperature used to generate samples, use 0 for greedy sampling.
    temperature: f64,

    /// Nucleus sampling probability cutoff.
    top_p: Option<f64>,

    /// Only sample among the top K samples.
    top_k: Option<usize>,

    /// The seed to use when generating random samples.
    seed: u64,

    /// Run on CPU rather than GPU even if a GPU is available.
    cpu: bool,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    repeat_last_n: usize,

    /// The model size to use.
    which: Which,
}

impl Args {
    async fn tokenizer(&self) -> Result<Tokenizer> {
        let repo = match self.which {
            Which::W2_0_5b => "Qwen/Qwen2-0.5B-Instruct",
            Which::W2_1_5b => "Qwen/Qwen2-1.5B-Instruct",
            Which::W2_7b => "Qwen/Qwen2-7B-Instruct",
            Which::W2_72b => "Qwen/Qwen2-72B-Instruct",
            Which::W25_0_5b => "Qwen/Qwen2.5-0.5B-Instruct",
            Which::W25_1_5b => "Qwen/Qwen2.5-1.5B-Instruct",
            Which::W25_7b => "Qwen/Qwen2.5-7B-Instruct",
            Which::W25_14b => "Qwen/Qwen2.5-14B-Instruct",
            Which::W25_32b => "Qwen/Qwen2.5-32B-Instruct",
        };
        let tokenizer_path = Api::new()?
            .model(repo.to_string())
            .get("tokenizer.json")
            .await?;
        Tokenizer::from_file(tokenizer_path).map_err(Error::msg)
    }

    async fn model(&self) -> Result<PathBuf> {
        let (repo, filename) = match self.which {
            Which::W2_0_5b => (
                "Qwen/Qwen2-0.5B-Instruct-GGUF",
                "qwen2-0_5b-instruct-q4_0.gguf",
            ),
            Which::W2_1_5b => (
                "Qwen/Qwen2-1.5B-Instruct-GGUF",
                "qwen2-1_5b-instruct-q4_0.gguf",
            ),
            Which::W2_7b => ("Qwen/Qwen2-7B-Instruct-GGUF", "qwen2-7b-instruct-q4_0.gguf"),
            Which::W2_72b => (
                "Qwen/Qwen2-72B-Instruct-GGUF",
                "qwen2-72b-instruct-q4_0.gguf",
            ),
            Which::W25_0_5b => (
                "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
                "qwen2.5-0.5b-instruct-q4_0.gguf",
            ),
            Which::W25_1_5b => (
                "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
                "qwen2.5-1.5b-instruct-q4_0.gguf",
            ),
            Which::W25_7b => (
                "Qwen/Qwen2.5-7B-Instruct-GGUF",
                "qwen2.5-7b-instruct-q4_0*.gguf",
            ),
            Which::W25_14b => (
                "Qwen/Qwen2.5-14B-Instruct-GGUF",
                "qwen2.5-14b-instruct-q4_0.gguf",
            ),
            Which::W25_32b => (
                "Qwen/Qwen2.5-32B-Instruct-GGUF",
                "qwen2.5-32b-instruct-q4_0.gguf",
            ),
        };
        let model_path = Api::new()?.model(repo.to_string()).get(filename).await?;
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
            cpu: false,
            repeat_penalty: 1.1,
            repeat_last_n: 64,
            which: Which::W2_7b,
        }
    }
}

async fn setup_model(args: &Args) -> Result<(Qwen2, candle::Device)> {
    let device = candle_examples::device(args.cpu)?;
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
        println!(
            "loaded {:?} tensors ({}) in {:.2}s",
            model.tensor_infos.len(),
            &format_size(total_size_in_bytes),
            start.elapsed().as_secs_f32(),
        );
        Qwen2::from_gguf(model, &mut file, &device)?
    };

    Ok((model, device))
}

async fn setup_tokenizer(args: &Args) -> Result<TokenOutputStream> {
    let tokenizer = args.tokenizer().await?;
    Ok(TokenOutputStream::new(tokenizer))
}

fn setup_logits_processor(args: &Args) -> LogitsProcessor {
    let temperature = args.temperature;
    let sampling = if temperature <= 0. {
        Sampling::ArgMax
    } else {
        match (args.top_k, args.top_p) {
            (None, None) => Sampling::All { temperature },
            (Some(k), None) => Sampling::TopK { k, temperature },
            (None, Some(p)) => Sampling::TopP { p, temperature },
            (Some(k), Some(p)) => Sampling::TopKThenTopP { k, p, temperature },
        }
    };
    LogitsProcessor::from_sampling(args.seed, sampling)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::Tensor;
    use hf_hub::{Cache, Repo};
    use std::env;
    use std::io::{self, BufRead, Write};
    use std::process::Command;
    use encoding_rs::GB18030;
    
    #[test]
    fn test_cmd() -> Result<()> {
        // 在 Windows 中使用 dir 命令代替 ls
        let output = Command::new("cmd")
            .args(&["/C", "dir"])
            // 设置命令输出的编码为 UTF-8
            .env("PYTHONIOENCODING", "utf-8")
            // 使用系统默认代码页
            .output()?;
        
        if output.status.success() {
            // 使用 GB18030 解码 Windows 命令行输出
            let stdout = GB18030.decode(&output.stdout).0;
            println!("Command output: {}", stdout);
        } else {
            let stderr = GB18030.decode(&output.stderr).0;
            eprintln!("Command failed with error: {}", stderr);
        }
        
        Ok(())
    }

    #[tokio::test]
    async fn test_get_qwen() -> Result<()> {
        env::set_var("HTTPS_PROXY", "http://127.0.0.1:10808");

        let which = Which::W25_7b;

        let (repo_str, filename) = match which {
            Which::W2_0_5b => (
                "Qwen/Qwen2-0.5B-Instruct-GGUF",
                "qwen2-0_5b-instruct-q4_0.gguf",
            ),
            Which::W2_1_5b => (
                "Qwen/Qwen2-1.5B-Instruct-GGUF",
                "qwen2-1_5b-instruct-q4_0.gguf",
            ),
            Which::W2_7b => ("Qwen/Qwen2-7B-Instruct-GGUF", "qwen2-7b-instruct-q4_0.gguf"),
            Which::W2_72b => (
                "Qwen/Qwen2-72B-Instruct-GGUF",
                "qwen2-72b-instruct-q4_0.gguf",
            ),
            Which::W25_0_5b => (
                "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
                "qwen2.5-0.5b-instruct-q4_0.gguf",
            ),
            Which::W25_1_5b => (
                "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
                "qwen2.5-1.5b-instruct-q4_0.gguf",
            ),
            Which::W25_7b => (
                "Qwen/Qwen2.5-7B-Instruct-GGUF",
                "qwen2.5-7b-instruct-q4_0.gguf",
            ),
            Which::W25_14b => (
                "Qwen/Qwen2.5-14B-Instruct-GGUF",
                "qwen2.5-14b-instruct-q4_0.gguf",
            ),
            Which::W25_32b => (
                "Qwen/Qwen2.5-32B-Instruct-GGUF",
                "qwen2.5-32b-instruct-q4_0.gguf",
            ),
        };

        let model_path = if let Some(path) = Cache::default()
            .repo(Repo::model(repo_str.to_string()))
            .get(filename)
        {
            path
        } else {
            let repo = Api::new()?.model(repo_str.to_string());
            let info = repo.info().await?;
            // println!("{:?}", info);

            let stem = filename.strip_suffix(".gguf").unwrap();

            let split_filenames: Vec<String> = info
                .siblings
                .into_iter()
                .map(|sibling| sibling.rfilename)
                .filter(|s| s.starts_with(stem))
                .collect();

            let mut split_paths = vec![];
            for filename in &split_filenames {
                split_paths.push(repo.get(filename).await?);
            }

            println!("Downloaded split files: {:#?}", split_paths);

            // 获取输出目录
            let output_dir = split_paths[0].parent().unwrap();
            println!("{:?}", output_dir);

            let merged_path = output_dir.join(filename);

            // 构建命令
            let mut command = Command::new("copy");
            command
                .arg("/b")
                .arg(format!("{stem}*.gguf"))
                .arg(merged_path.to_str().unwrap());
            
            println!("{:?}", command);

            // 执行命令
            let output = command.output()?;
            
            if !output.status.success() {
                let error = String::from_utf8_lossy(&output.stderr);
                return Err(Error::msg(format!("gguf-utils merge failed: {}", error)));
            }
            
            let merged_path = output_dir.join(filename);
            merged_path
            // Default::default()
        };

        env::remove_var("HTTPS_PROXY");
        Ok(())
    }

    fn get_user_prompt() -> String {
        println!("请输入您的问题 (直接回车使用默认问题):");
        let stdin = io::stdin();
        let mut line = String::new();
        stdin
            .lock()
            .read_line(&mut line)
            .expect("Failed to read line");

        // 去除末尾的换行符
        line = line.trim().to_string();

        if line.is_empty() {
            DEFAULT_PROMPT.to_string()
        } else {
            line
        }
    }

    fn process_prompt(
        prompt: String,
        model: &mut Qwen2,
        tos: &mut TokenOutputStream,
        device: &candle::Device,
        logits_processor: &mut LogitsProcessor,
    ) -> Result<(Vec<u32>, u32)> {
        let prompt = format!(
            "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            prompt
        );
        // print!("formatted instruct prompt: {}", &prompt);

        let tokens = tos.tokenizer().encode(prompt, true).map_err(Error::msg)?;
        let tokens = tokens.get_ids().to_vec();

        let next_token = {
            let input = Tensor::new(&*tokens, device)?.unsqueeze(0)?;
            let logits = model.forward(&input, 0)?;
            let logits = logits.squeeze(0)?;
            logits_processor.sample(&logits)?
        };
        Ok((tokens, next_token))
    }

    fn generate_tokens(
        model: &mut Qwen2,
        tos: &mut TokenOutputStream,
        args: &Args,
        device: &candle::Device,
        initial_tokens: Vec<u32>,
        mut next_token: u32,
        logits_processor: &mut LogitsProcessor,
    ) -> Result<usize> {
        let mut all_tokens = vec![next_token];
        if let Some(t) = tos.next_token(next_token)? {
            print!("{t}");
            io::stdout().flush()?;
        }

        let eos_token = *tos.tokenizer().get_vocab(true).get("<|im_end|>").unwrap();
        let to_sample = args.sample_len.saturating_sub(1);
        let mut generated_count = 0;

        for index in 0..to_sample {
            let input = Tensor::new(&[next_token], device)?.unsqueeze(0)?;
            let logits = model.forward(&input, initial_tokens.len() + index)?;
            let logits = logits.squeeze(0)?;

            let logits = if args.repeat_penalty == 1. {
                logits
            } else {
                let start_at = all_tokens.len().saturating_sub(args.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    args.repeat_penalty,
                    &all_tokens[start_at..],
                )?
            };

            next_token = logits_processor.sample(&logits)?;
            all_tokens.push(next_token);

            if let Some(t) = tos.next_token(next_token)? {
                print!("{t}");
                io::stdout().flush()?;
            }
            generated_count += 1;

            if next_token == eos_token {
                break;
            };
        }

        if let Some(rest) = tos.decode_rest().map_err(candle::Error::msg)? {
            print!("{rest}");
        }
        io::stdout().flush()?;

        Ok(generated_count)
    }

    #[tokio::test]
    async fn test_gen() -> Result<()> {
        env::set_var("HTTPS_PROXY", "http://127.0.0.1:10808");
        let args = Args::default();

        println!("{args:?}");

        // 分别初始化各个组件
        let (mut model, device) = setup_model(&args).await?;
        let mut tos = setup_tokenizer(&args).await?;
        let mut logits_processor = setup_logits_processor(&args);

        // 从控制台获取用户输入
        let user_prompt = get_user_prompt();
        let prompt = user_prompt;

        let start_prompt_processing = std::time::Instant::now();
        let (tokens, next_token) = process_prompt(
            prompt,
            &mut model,
            &mut tos,
            &device,
            &mut logits_processor,
        )?;
        let prompt_dt = start_prompt_processing.elapsed();

        let start_post_prompt = std::time::Instant::now();
        let generated_count = generate_tokens(
            &mut model,
            &mut tos,
            &args,
            &device,
            tokens.clone(),
            next_token,
            &mut logits_processor,
        )?;
        let dt = start_post_prompt.elapsed();

        println!(
            "\n\n{:4} prompt tokens processed: {:.2} token/s",
            tokens.len(),
            tokens.len() as f64 / prompt_dt.as_secs_f64(),
        );
        println!(
            "{:4} tokens generated: {:.2} token/s",
            generated_count,
            generated_count as f64 / dt.as_secs_f64(),
        );

        env::remove_var("HTTPS_PROXY");
        Ok(())
    }
}
