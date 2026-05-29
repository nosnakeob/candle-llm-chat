//! Qwen3-VL 多模态推理管线
//!
//! 基于 candle-transformers 的 `Qwen3VLModel`，提供图文对话能力。
//!
//! # 设计说明
//!
//! Qwen3VLModel 的 forward 签名与普通文本模型不同（9 参数，`&self`），
//! 因此不通过 `ModelInference` trait 集成，而是作为独立管线。
//! KV cache 通过内部 `Arc<Mutex<KvCache>>` 管理，无需外部清理。

use anyhow::{Context, Error, Result};
use async_stream::try_stream;
use candle::{DType, Device, IndexOp, Tensor};
use candle_examples::token_output_stream::TokenOutputStream;
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::qwen3_vl::{Config as Qwen3VLConfig, Qwen3VLModel};
use futures_core::stream::Stream;
use hf_hub::api::tokio::ApiBuilder;
use serde_json::Value;
use std::path::Path;
use std::pin::Pin;
use tracing::info;

use crate::model::config::InferenceConfig;
use crate::model::registry::ModelRegistry;
use crate::utils::chat::ChatContext;
use crate::utils::load::{ApiRepoExt, load_tokenizer};

// ─── 图像预处理常量 ───────────────────────────────────────────────────────────────

const IMAGE_MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
const IMAGE_STD: [f32; 3] = [0.26862954, 0.26130258, 0.27577711];

// ─── 预处理结果 ─────────────────────────────────────────────────────────────────

struct Qwen3VLPreprocessed {
    pixel_values: Option<Tensor>,
    image_grid_thw: Option<Tensor>,
    num_placeholders: usize,
}

impl Qwen3VLPreprocessed {
    fn none() -> Self {
        Self { pixel_values: None, image_grid_thw: None, num_placeholders: 0 }
    }
}

// ─── Qwen3VL 结构体 ──────────────────────────────────────────────────────────────

pub struct Qwen3VL {
    model: Qwen3VLModel,
    tokenizer: tokenizers::Tokenizer,
    tos: TokenOutputStream,
    logits_processor: LogitsProcessor,
    ctx: ChatContext,
    infer_conf: InferenceConfig,
    eos_token_id: u32,

    // Vision 配置
    patch_size: usize,
    spatial_merge_size: usize,
    temporal_patch_size: usize,
    image_token_id: u32,
}

impl Drop for Qwen3VL {
    fn drop(&mut self) {
        // 同步 CUDA 设备，防止 cuDNN handle 在 drop 时因未完成的内核而崩溃
        let _ = self.infer_conf.device.synchronize();
    }
}

impl Qwen3VL {
    pub async fn new(model_id: &str, config: InferenceConfig) -> Result<Self> {
        let registry = ModelRegistry::new()?;
        let hub_info = registry.get(model_id)?;
        let device = &config.device;

        let api = ApiBuilder::from_env().build()?;
        let repo = api.model(hub_info.model_repo.clone());

        // 下载并解析 config.json
        let config_path = repo.get("config.json").await?;
        let config_content = std::fs::read_to_string(&config_path)?;
        let mut config_value: Value = serde_json::from_str(&config_content)?;

        // 修复 tie_word_embeddings 缺失问题
        if let Some(text_config) = config_value
            .as_object_mut()
            .and_then(|o| o.get_mut("text_config"))
            .and_then(|v| v.as_object_mut())
        {
            if !text_config.contains_key("tie_word_embeddings") {
                text_config.insert("tie_word_embeddings".to_string(), Value::Bool(false));
            }
            // 限制 max_position_embeddings 以减少 KV cache 显存占用
            let should_limit = text_config
                .get("max_position_embeddings")
                .and_then(|v| v.as_u64())
                .map_or(false, |n| n > 8192);
            if should_limit {
                text_config.insert(
                    "max_position_embeddings".to_string(),
                    Value::Number(serde_json::Number::from(8192u64)),
                );
            }
        }

        let vl_config: Qwen3VLConfig = serde_json::from_value(config_value.clone())?;

        // 加载模型权重
        let model_files = match repo.get(&hub_info.model_file).await {
            Ok(single_file) => vec![single_file],
            Err(_) => repo.get_safetensors().await?,
        };
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&model_files, DType::BF16, device)? };
        let model = Qwen3VLModel::new(&vl_config, vb)?;

        // 加载 tokenizer
        let tokenizer = load_tokenizer(&hub_info.tokenizer_repo)?;

        // eos_token_id
        let eos_token_id = config_value
            .get("eos_token_id")
            .or_else(|| config_value.get("text_config").and_then(|tc| tc.get("eos_token_id")))
            .and_then(|x| x.as_u64())
            .unwrap_or(151643) as u32;

        let patch_size = vl_config.vision_config.patch_size;
        let spatial_merge_size = vl_config.vision_config.spatial_merge_size;
        let temporal_patch_size = vl_config.vision_config.temporal_patch_size;
        let image_token_id = vl_config.image_token_id;

        let logits_processor =
            LogitsProcessor::new(config.seed, Some(config.temperature), config.top_p);
        let ctx = ChatContext::from_repo(&hub_info.tokenizer_repo).await?;
        let tos = TokenOutputStream::new(tokenizer.clone());

        Ok(Self {
            model, tokenizer, tos, logits_processor, ctx, infer_conf: config, eos_token_id,
            patch_size, spatial_merge_size, temporal_patch_size, image_token_id,
        })
    }

    pub async fn with_default_config(model_id: &str) -> Result<Self> {
        Self::new(model_id, InferenceConfig::default()).await
    }

    pub async fn default() -> Result<Self> {
        Self::with_default_config("qwen3_vl").await
    }

    // ─── 公有 API ────────────────────────────────────────────────────────────

    /// 完整响应对话
    pub fn chat_full(&mut self, prompt: &str, images: Option<&[&str]>) -> Result<String> {
        let has_images = images.map_or(false, |imgs| !imgs.is_empty());

        let pp = if has_images {
            self.preprocess_images(images.unwrap())?
        } else {
            Qwen3VLPreprocessed::none()
        };

        let full_prompt = if pp.num_placeholders > 0 {
            build_vision_prompt(prompt, pp.num_placeholders)
        } else {
            prompt.to_string()
        };

        self.ctx.push_msg(&full_prompt);
        let rendered = self.ctx.render()?;

        let answer = self.generate_inner(&rendered, pp)?;

        self.ctx.push_msg(&answer);
        Ok(answer)
    }

    /// 流式对话
    pub fn chat<'a>(
        &'a mut self,
        prompt: &'a str,
        images: Option<&[&'a str]>,
    ) -> Pin<Box<dyn Stream<Item = Result<String>> + 'a>> {
        let has_images = images.map_or(false, |imgs| !imgs.is_empty());

        let pp = match if has_images {
            self.preprocess_images(images.unwrap())
        } else {
            Ok(Qwen3VLPreprocessed::none())
        } {
            Ok(pp) => pp,
            Err(e) => return Box::pin(try_stream! { yield format!("[图像预处理失败: {}]", e); }),
        };

        let full_prompt = if pp.num_placeholders > 0 {
            build_vision_prompt(prompt, pp.num_placeholders)
        } else {
            prompt.to_string()
        };

        self.ctx.push_msg(&full_prompt);

        let rendered = match self.ctx.render() {
            Ok(r) => r,
            Err(e) => return Box::pin(try_stream! { yield format!("[{}]", e); }),
        };

        let prompt = rendered;
        let device = self.infer_conf.device.clone();
        let sample_len = self.infer_conf.sample_len;
        let eos_token_id = self.eos_token_id;

        let prompt_tokens = match self.tokens_from_render(&prompt) {
            Ok(t) => t,
            Err(e) => return Box::pin(try_stream! { yield format!("[{}]", e); }),
        };
        let prompt_len = prompt_tokens.len();

        let continuous_img_pad = if pp.num_placeholders > 0 {
            find_image_token_span(&prompt_tokens, self.image_token_id)
        } else {
            vec![]
        };

        Box::pin(try_stream! {
            let prompt = prompt;
            let mut token_ids = prompt_tokens;
            let mut answer = String::with_capacity(1024);
            let start = std::time::Instant::now();

            // 预填充
            let input = Tensor::new(token_ids.as_slice(), &device)?.unsqueeze(0)?;
            let seq_len = input.dim(1)?;

            let logits = self.model.forward(
                &input, pp.pixel_values, None, pp.image_grid_thw, None,
                vec![seq_len], continuous_img_pad, vec![], &[0],
            )?;

            let mut next_token = {
                let l = logits.squeeze(1)?.i(0)?.to_dtype(DType::F32)?;
                self.logits_processor.sample(&l)?
            };

            for index in 0..sample_len {
                if next_token == eos_token_id { break; }

                if let Some(t) = self.tos.next_token(next_token)? {
                    answer.push_str(&t);
                    yield t;
                }
                token_ids.push(next_token);

                let input = Tensor::new(&[next_token], &device)?.unsqueeze(0)?;
                let pos = prompt_len + index;

                let logits = self.model.forward(
                    &input, None, None, None, None,
                    vec![1], vec![], vec![], &[pos],
                )?;

                next_token = {
                    let l = logits.i(0)?.to_dtype(DType::F32)?;
                    self.logits_processor.sample(&l)?
                };
            }

            if let Some(t) = self.tos.decode_rest()? {
                if !t.is_empty() { answer.push_str(&t); yield t; }
            }

            self.ctx.push_msg(&answer);
            self.tos.clear();

            let elapsed = start.elapsed().as_secs_f64();
            if elapsed > 0.0 {
                info!("speed: {:.2} token/s, total tokens: {}",
                    (token_ids.len() - prompt_len) as f64 / elapsed, token_ids.len());
            }
        })
    }

    // ─── 内部生成 ────────────────────────────────────────────────────────────

    /// 内部生成函数 - 分离以便借用法
    fn generate_inner(&mut self, prompt: &str, pp: Qwen3VLPreprocessed) -> Result<String> {
        let mut token_ids = self.str2tokens(prompt)?;
        let continuous_img_pad = if pp.num_placeholders > 0 {
            find_image_token_span(&token_ids, self.image_token_id)
        } else {
            vec![]
        };

        let prompt_len = token_ids.len();
        let mut answer = String::with_capacity(1024);
        let start = std::time::Instant::now();

        // 预填充 - 借用 self.model (&self) 完成 forward 后释放
        let input = Tensor::new(token_ids.as_slice(), &self.infer_conf.device)?.unsqueeze(0)?;
        let seq_len = input.dim(1)?;

        let logits = self.model.forward(
            &input, pp.pixel_values, None, pp.image_grid_thw, None,
            vec![seq_len], continuous_img_pad, vec![], &[0],
        )?;

        // 采样 - 借用 self.logits_processor (&mut self) 后释放
        // 注意: forward_embeds 已返回最后 token 的 logits (1, vocab_size)
        let mut next_token = {
            let l = logits.squeeze(1)?.i(0)?.to_dtype(DType::F32)?;
            self.logits_processor.sample(&l)?
        };

        // 解码循环 - 交替借用 self.model, self.logits_processor, self.tos
        for index in 0..self.infer_conf.sample_len {
            if next_token == self.eos_token_id { break; }

            // 输出当前 token - 借用 self.tos
            let token_str = self.tos.next_token(next_token)?;
            if let Some(t) = &token_str { answer.push_str(t); }
            drop(token_str);

            token_ids.push(next_token);

            // 生成下一个 token - 借用 self.model
            let input = Tensor::new(&[next_token], &self.infer_conf.device)?.unsqueeze(0)?;
            let pos = prompt_len + index;

            let logits = self.model.forward(
                &input, None, None, None, None,
                vec![1], vec![], vec![], &[pos],
            )?;

            // 采样 - 借用 self.logits_processor
            next_token = {
                let l = logits.i(0)?.to_dtype(DType::F32)?;
                self.logits_processor.sample(&l)?
            };
        }

        // 解码剩余
        if let Some(t) = self.tos.decode_rest()? {
            if !t.is_empty() { answer.push_str(&t); }
        }

        self.tos.clear();

        let elapsed = start.elapsed().as_secs_f64();
        if elapsed > 0.0 {
            info!("speed: {:.2} token/s, total tokens: {}",
                (token_ids.len() - prompt_len) as f64 / elapsed, prompt_len);
        }

        Ok(answer)
    }

    // ─── 图像预处理 ──────────────────────────────────────────────────────────

    fn preprocess_images(&self, paths: &[&str]) -> Result<Qwen3VLPreprocessed> {
        if paths.is_empty() { return Ok(Qwen3VLPreprocessed::none()); }

        let path = paths[0];
        let img = image::io::Reader::open(path)
            .context(format!("无法打开图片: {}", path))?
            .decode()
            .context(format!("无法解码图片: {}", path))?;

        let block_size = (self.patch_size * self.spatial_merge_size) as u32;
        let (target_w, target_h) = smart_resize(img.width(), img.height(), block_size);

        let img = img.resize_exact(target_w, target_h, image::imageops::FilterType::CatmullRom);
        let rgb = img.to_rgb8();

        let h_patches = target_h as usize / self.patch_size;
        let w_patches = target_w as usize / self.patch_size;
        let total_patches = h_patches * w_patches;
        let patch_dim = self.patch_size * self.patch_size * self.temporal_patch_size * 3;

        let mut pixel_data = Vec::with_capacity(total_patches * patch_dim);

        for py in 0..h_patches {
            for px in 0..w_patches {
                let y_off = py * self.patch_size;
                let x_off = px * self.patch_size;

                for c in 0..3 {
                    let mean = IMAGE_MEAN[c];
                    let std = IMAGE_STD[c];
                    for _t in 0..self.temporal_patch_size {
                        for y in 0..self.patch_size {
                            for x in 0..self.patch_size {
                                let pv = rgb.get_pixel(
                                    (x_off + x) as u32, (y_off + y) as u32,
                                )[c] as f32;
                                pixel_data.push((pv / 255.0 - mean) / std);
                            }
                        }
                    }
                }
            }
        }

        let pixel_values = Tensor::from_vec(pixel_data, (total_patches, patch_dim), &Device::Cpu)?
            .to_device(&self.infer_conf.device)?;

        let grid = Tensor::new(&[1u32, h_patches as u32, w_patches as u32], &Device::Cpu)?
            .unsqueeze(0)?
            .to_device(&self.infer_conf.device)?;

        let num_placeholders = total_patches / (self.spatial_merge_size * self.spatial_merge_size);

        Ok(Qwen3VLPreprocessed {
            pixel_values: Some(pixel_values),
            image_grid_thw: Some(grid),
            num_placeholders,
        })
    }

    fn str2tokens(&self, string: &str) -> Result<Vec<u32>> {
        Ok(self.tokenizer.encode(string, true).map_err(Error::msg)?.get_ids().to_vec())
    }

    fn tokens_from_render(&self, rendered: &str) -> Result<Vec<u32>> {
        self.str2tokens(rendered)
    }
}

// ─── 自由函数 ─────────────────────────────────────────────────────────────────

fn build_vision_prompt(prompt: &str, num_placeholders: usize) -> String {
    let image_tokens: String = (0..num_placeholders).map(|_| "<|image_pad|>").collect();
    format!("<|vision_start|>{}<|vision_end|>{}", image_tokens, prompt)
}

fn smart_resize(width: u32, height: u32, factor: u32) -> (u32, u32) {
    let h = ((height + factor - 1) / factor) * factor;
    let w = ((width + factor - 1) / factor) * factor;
    (w.max(factor), h.max(factor))
}

fn find_image_token_span(tokens: &[u32], image_token_id: u32) -> Vec<Vec<(usize, usize)>> {
    let positions: Vec<usize> = tokens.iter().enumerate()
        .filter(|&(_, id)| *id == image_token_id)
        .map(|(i, _)| i).collect();

    if positions.is_empty() { return vec![]; }
    let start = positions[0];
    let end = positions[positions.len() - 1] + 1;
    vec![vec![(start, end)]]
}

// ─── 测试 ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_smart_resize() {
        let factor = 28;
        assert_eq!(smart_resize(448, 448, factor), (448, 448));
        assert_eq!(smart_resize(450, 450, factor), (476, 476));
        assert_eq!(smart_resize(10, 10, factor), (28, 28));
    }

    #[test]
    fn test_find_image_token_span() {
        let tokens = vec![1u32, 2, 3, 151655, 151655, 151655, 4, 5];
        let spans = find_image_token_span(&tokens, 151655);
        assert_eq!(spans, vec![vec![(3, 6)]]);
    }

    #[test]
    fn test_empty_span() {
        let tokens = vec![1u32, 2, 3, 4];
        let spans = find_image_token_span(&tokens, 151655);
        assert!(spans.is_empty());
    }

    #[tokio::test]
    #[ignore]
    async fn test_chat_text() -> Result<()> {
        let mut cfg = InferenceConfig::default();
        cfg.temperature = 0.8;
        cfg.repeat_penalty = 1.0;
        let mut model = Qwen3VL::new("qwen3_vl", cfg).await?;

        let answer = model.chat_full("What is 2+2?", None)?;
        println!("Answer: '{}'", answer);
        assert!(!answer.is_empty());
        assert!(answer.len() > 3, "answer too short: '{}'", answer);
        Ok(())
    }

    #[tokio::test]
    #[ignore]
    async fn test_chat_with_image() -> Result<()> {
        let img_path = "test.jpg";
        if !Path::new(img_path).exists() {
            println!("跳过测试：找不到 {}", img_path);
            return Ok(());
        }
        let mut model = Qwen3VL::default().await?;
        let answer = model.chat_full("请描述这张图片", Some(&[img_path]))?;
        println!("Answer: {}", answer);
        assert!(!answer.is_empty());
        Ok(())
    }

    #[tokio::test]
    #[ignore]
    async fn test_chat_with_real_image() -> Result<()> {
        // 程序化生成一张房屋场景图（网络不可靠时无需外部下载）
        let tmp_dir = std::env::temp_dir();
        let img_path = tmp_dir.join("qwen3_vl_test_house.png");

        let mut img = image::ImageBuffer::new(448, 448);
        // 天空渐变
        for y in 0..224 {
            let b = 128 + (y as f32 / 223.0 * 127.0) as u8;
            for x in 0..448 {
                img.put_pixel(x, y, image::Rgb([135 - y as u8 / 3, 206 - y as u8 / 3, b]));
            }
        }
        // 草地
        for y in 224..448 {
            for x in 0..448 {
                img.put_pixel(x, y, image::Rgb([34, 139 + (y - 224) as u8 / 8, 34]));
            }
        }
        // 太阳（左上角黄色圆形）
        for y in 0..120u32 {
            for x in 0..120u32 {
                let dx = x as i32 - 60;
                let dy = y as i32 - 60;
                if dx * dx + dy * dy <= 2500 {
                    img.put_pixel(x, y, image::Rgb([255, 255, 0]));
                }
            }
        }
        // 房子主体（棕色矩形）
        for y in 180..340 {
            for x in 120..320 {
                img.put_pixel(x, y, image::Rgb([139, 90, 43]));
            }
        }
        // 屋顶（红色三角形）
        for y in 0..100 {
            let half_width = 100 - y as i32;
            for x in (120 + half_width)..(320 - half_width) {
                if x >= 0 && x < 448 && (180 - y as i32) >= 0 {
                    img.put_pixel(x as u32, 180 - y as u32, image::Rgb([178, 34, 34]));
                }
            }
        }
        // 门（深棕色矩形）
        for y in 250..340 {
            for x in 195..245 {
                img.put_pixel(x, y, image::Rgb([101, 67, 33]));
            }
        }
        // 窗户（浅蓝色矩形）
        for y in 210..250 {
            for x in 140..180 {
                img.put_pixel(x, y, image::Rgb([173, 216, 230]));
            }
        }
        // 树（树干 + 绿色树冠）
        for y in 200..380 {
            for x in 370..385 {
                img.put_pixel(x, y, image::Rgb([101, 67, 33]));
            }
        }
        for dy in 0..80i32 {
            for dx in 0..100i32 {
                let cx = 345 + dx;
                let cy = 180 + dy;
                let rx = dx as i32 - 50;
                let ry = dy as i32 - 40;
                if rx * rx / 2500 + ry * ry / 1600 <= 1 {
                    if cx < 448 && cy < 448 {
                        img.put_pixel(cx as u32, cy as u32, image::Rgb([34, 139, 34]));
                    }
                }
            }
        }
        img.save(&img_path)?;
        println!("测试图片已生成: {:?}", img_path);

        let mut cfg = InferenceConfig::default();
        cfg.temperature = 0.3;
        let mut model = Qwen3VL::new("qwen3_vl", cfg).await?;
        let answer = model.chat_full(
            "请用一句话描述这张图片里有什么物体",
            Some(&[img_path.to_str().unwrap()]),
        )?;
        println!("Answer: {}", answer);
        assert!(!answer.is_empty());

        let _ = std::fs::remove_file(&img_path);
        Ok(())
    }
}
