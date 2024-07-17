use tokio::fs::File;
use std::io::Write;
use std::sync::Arc;
use std::task::Poll;
use std::time;
use candle_transformers::models::quantized_llama::ModelWeights;
use crate::{Args, format_size};
use super::token_output_stream::TokenOutputStream;
use anyhow::Result;
use candle_core::quantized::gguf_file;
use candle_core::Tensor;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::utils::apply_repeat_penalty;
use futures::Stream;
use tokio::runtime::Handle;
use tokio::sync::oneshot::Receiver;

struct ChatBot {
    model: ModelWeights,
    tos: TokenOutputStream,
    logits_processor: LogitsProcessor,

    args: Args,
    eos_token: u32,
}

impl ChatBot {
    async fn from_args(args: Args) -> Result<Self> {
        let model_path = args.model().await?;

        let mut file = File::open(&model_path).await?.into_std().await;
        let start = time::Instant::now();

        // This is the model instance
        let model = gguf_file::Content::read(&mut file)?;
        let mut total_size_in_bytes = 0;
        for (_, tensor) in model.tensor_infos.iter() {
            let elem_count = tensor.shape.elem_count();
            total_size_in_bytes +=
                elem_count * tensor.ggml_dtype.type_size() / tensor.ggml_dtype.block_size();
        }
        info!("loaded {:?} tensors ({}) in {:.2}s",
            model.tensor_infos.len(),
            &format_size(total_size_in_bytes),
            start.elapsed().as_secs_f32(),
        );
        let model = ModelWeights::from_gguf(model, &mut file, &args.device)?;

        let tokenizer = args.tokenizer().await?;
        let tos = TokenOutputStream::new(tokenizer);

        let eos_token = tos.tokenizer().token_to_id("<|end_of_turn|>").unwrap();

        let logits_processor = LogitsProcessor::new(args.seed, Some(args.temperature), None);

        Ok(Self {
            model,
            tos,
            logits_processor,
            args,
            eos_token,
        })
    }

    fn chat(&mut self, input: String) -> Result<String> {
        let mut res = String::new();

        let prompt = format!("User: {} <|end_of_turn|> Assistant: ", input.trim());

        let tokens = self.tos
            .tokenizer()
            .encode(prompt, true)
            .map_err(anyhow::Error::msg)?;

        let prompt_tokens = tokens.get_ids();
        let mut all_tokens = vec![];

        let start_prompt_processing = time::Instant::now();
        let mut next_token = {
            let input = Tensor::new(prompt_tokens, &self.args.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, 0)?;
            let logits = logits.squeeze(0)?;
            self.logits_processor.sample(&logits)?
        };
        let prompt_dt = start_prompt_processing.elapsed();

        if let Some(t) = self.tos.next_token(next_token)? {
            // 第一个单词
            res.push_str(&t);
            // return Poll::Ready(Some(Ok(t)));
        }

        let start_post_prompt = time::Instant::now();
        let to_sample = self.args.sample_len.saturating_sub(1);
        let mut sampled = 1;
        while sampled < to_sample {
            let input = Tensor::new(&[next_token], &self.args.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, prompt_tokens.len() + sampled)?;
            let logits = logits.squeeze(0)?;
            let logits = if self.args.repeat_penalty == 1. {
                logits
            } else {
                let start_at = all_tokens.len().saturating_sub(self.args.repeat_last_n);
                apply_repeat_penalty(
                    &logits,
                    self.args.repeat_penalty,
                    &all_tokens[start_at..],
                )?
            };
            next_token = self.logits_processor.sample(&logits)?;
            all_tokens.push(next_token);
            if let Some(t) = self.tos.next_token(next_token)? {
                res.push_str(&t);
                // return Poll::Ready(Some(Ok(t)));
            }
            sampled += 1;
            if next_token == self.eos_token {
                break;
            };
        }
        if let Some(rest) = self.tos.decode_rest().map_err(candle_core::Error::msg)? {
            res.push_str(&rest);
            // return Poll::Ready(Some(Ok(rest)));
        }
        let dt = start_post_prompt.elapsed();

        info!("{} prompt tokens processed: {:.2} token/s",
            prompt_tokens.len(),
            prompt_tokens.len() as f64 / prompt_dt.as_secs_f64(),
        );
        info!("{sampled} tokens generated: {:.2} token/s",
            sampled as f64 / dt.as_secs_f64(),
        );
        // Poll::Ready(None)

        Ok(res)
    }
}

#[cfg(test)]
mod test {
    use futures::StreamExt;
    use super::*;

    #[tokio::test]
    async fn t_chatbot_create() -> Result<()> {
        let args = Args::default();
        let mut chatbot = ChatBot::from_args(args).await?;

        Ok(())
    }

    #[tokio::test]
    async fn t_chatbot_chat() -> Result<()> {
        tracing_subscriber::fmt::init();

        let args = Args::default();
        let mut chatbot = ChatBot::from_args(args).await?;

        let mut input = "hi".to_string();
        let mut output = chatbot.chat(input)?;

        // print!("output: ");
        // while let Some(Ok(word)) = output.next().await {
        //     print!("{}", word);
        // }

        println!("output: {}", output);
        //
        // input = "你好";
        // output = chatbot.chat(input)?;
        // println!("output: {}", output);

        Ok(())
    }
}