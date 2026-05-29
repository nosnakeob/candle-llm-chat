//! Qwen3-VL 模型包装 + 图像预处理
//!
//! 本模块提供 Qwen3VLModel 的 `ModelInference` trait 实现，
//! 以及图像预处理自由函数供 `ChatPipeline` 调用。
//!
//! # 设计
//!
//! Qwen3VLModel 的 forward 签名与普通文本模型不同（9 参数，`&self`），
//! 但通过 `VisionInput` 封装差异后，可以实现统一的 `ModelInference` trait。
//! KV cache 通过内部 `Arc<Mutex<KvCache>>` 管理，无需外部清理。

use anyhow::{Context, Result};
use candle::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen3_vl::{Config as Qwen3VLConfig, Qwen3VLModel};
use hf_hub::api::tokio::ApiBuilder;
use image::GenericImageView;
use serde_json::Value;
use tokenizers::Tokenizer;

use crate::model::VisionInput;
use crate::model::hub::HubInfo;
use crate::model::{VisionConfig, ModelInference};
use crate::utils::load::{ApiRepoExt, load_tokenizer};

// ─── 图像预处理常量 ───────────────────────────────────────────────────────────────

const IMAGE_MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
const IMAGE_STD: [f32; 3] = [0.26862954, 0.26130258, 0.27577711];

// ─── Qwen3VL 结构体 ──────────────────────────────────────────────────────────────

pub struct Qwen3VL {
    model: Qwen3VLModel,
    pub(crate) vision_config: VisionConfig,
    device: Device,
}

impl Drop for Qwen3VL {
    fn drop(&mut self) {
        // 同步 CUDA 设备，防止 cuDNN handle 在 drop 时因未完成的内核而崩溃
        let _ = self.device.synchronize();
    }
}

impl Qwen3VL {
    /// 加载 Qwen3VL 模型，返回 (模型包装, Tokenizer, eos_token_id)
    pub async fn load(hub_info: &HubInfo, device: &Device) -> Result<(Self, Tokenizer, u32)> {
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

        let eos_token_id = config_value
            .get("eos_token_id")
            .or_else(|| config_value.get("text_config").and_then(|tc| tc.get("eos_token_id")))
            .and_then(|x| x.as_u64())
            .unwrap_or(151643) as u32;

        let vision_config = VisionConfig {
            patch_size: vl_config.vision_config.patch_size,
            spatial_merge_size: vl_config.vision_config.spatial_merge_size,
            temporal_patch_size: vl_config.vision_config.temporal_patch_size,
            image_token_id: vl_config.image_token_id,
        };

        Ok((Self { model, vision_config, device: device.clone() }, tokenizer, eos_token_id))
    }
}

impl ModelInference for Qwen3VL {
    fn forward(
        &mut self,
        x: &Tensor,
        index_pos: usize,
        vision: Option<&VisionInput>,
    ) -> Result<candle::Tensor> {
        if let Some(v) = vision {
            // 预填充阶段：传入图像数据
            let seq_len = x.dim(1)?;
            let logits = self.model.forward(
                x,
                Some(v.pixel_values.clone()),
                None,
                Some(v.image_grid_thw.clone()),
                None,
                vec![seq_len],
                v.image_token_span.clone(),
                vec![],
                &[0],
            )?;
            // logits 形状: (1, seq_len, vocab_size)，取最后一位
            let last = logits.narrow(1, seq_len - 1, 1)?.to_dtype(DType::F32)?;
            Ok(last)
        } else {
            // 解码阶段：纯文本
            let logits = self.model.forward(
                x, None, None, None, None, vec![1], vec![], vec![], &[index_pos],
            )?;
            Ok(logits.to_dtype(DType::F32)?)
        }
    }

    fn clr_kv_cache(&mut self) {
        // Qwen3VLModel 的 KV cache 通过内部 Arc<Mutex<KvCache>> 管理，无需外部清理
    }

    fn vision_config(&self) -> Option<&VisionConfig> {
        Some(&self.vision_config)
    }
}

// ─── 图像预处理自由函数 ──────────────────────────────────────────────────────────

/// 图像预处理结果
pub struct PreprocessResult {
    pub pixel_values: Tensor,
    pub image_grid_thw: Tensor,
    pub num_placeholders: usize,
}

/// 预处理图像：加载 → resize → 归一化 → 构造视觉张量
pub fn preprocess_images(paths: &[&str], config: &VisionConfig, device: &Device) -> Result<PreprocessResult> {
    if paths.is_empty() {
        anyhow::bail!("no images provided");
    }

    let path = paths[0];
    let img = image::io::Reader::open(path)
        .context(format!("无法打开图片: {}", path))?
        .decode()
        .context(format!("无法解码图片: {}", path))?;

    let block_size = (config.patch_size * config.spatial_merge_size) as u32;
    let (target_w, target_h) = smart_resize(img.width(), img.height(), block_size);

    let img = img.resize_exact(target_w, target_h, image::imageops::FilterType::CatmullRom);
    let rgb = img.to_rgb8();

    let h_patches = target_h as usize / config.patch_size;
    let w_patches = target_w as usize / config.patch_size;
    let total_patches = h_patches * w_patches;
    let patch_dim = config.patch_size * config.patch_size * config.temporal_patch_size * 3;

    let mut pixel_data = Vec::with_capacity(total_patches * patch_dim);

    for py in 0..h_patches {
        for px in 0..w_patches {
            let y_off = py * config.patch_size;
            let x_off = px * config.patch_size;

            for c in 0..3 {
                let mean = IMAGE_MEAN[c];
                let std = IMAGE_STD[c];
                for _t in 0..config.temporal_patch_size {
                    for y in 0..config.patch_size {
                        for x in 0..config.patch_size {
                            let pv = rgb.get_pixel(
                                (x_off + x) as u32,
                                (y_off + y) as u32,
                            )[c] as f32;
                            pixel_data.push((pv / 255.0 - mean) / std);
                        }
                    }
                }
            }
        }
    }

    let pixel_values = Tensor::from_vec(pixel_data, (total_patches, patch_dim), &Device::Cpu)?
        .to_device(device)?;

    let grid = Tensor::new(&[1u32, h_patches as u32, w_patches as u32], &Device::Cpu)?
        .unsqueeze(0)?
        .to_device(device)?;

    let num_placeholders = total_patches / (config.spatial_merge_size * config.spatial_merge_size);

    Ok(PreprocessResult {
        pixel_values,
        image_grid_thw: grid,
        num_placeholders,
    })
}

/// 构建带有视觉占位符的 prompt
pub fn build_vision_prompt(prompt: &str, num_placeholders: usize) -> String {
    let image_tokens: String = (0..num_placeholders).map(|_| "<|image_pad|>").collect();
    format!("<|vision_start|>{}<|vision_end|>{}", image_tokens, prompt)
}

/// 在 token 序列中查找连续的 image token 区间
pub fn find_image_token_span(tokens: &[u32], image_token_id: u32) -> Vec<Vec<(usize, usize)>> {
    let positions: Vec<usize> = tokens.iter().enumerate()
        .filter(|&(_, id)| *id == image_token_id)
        .map(|(i, _)| i).collect();

    if positions.is_empty() { return vec![]; }
    let start = positions[0];
    let end = positions[positions.len() - 1] + 1;
    vec![vec![(start, end)]]
}

fn smart_resize(width: u32, height: u32, factor: u32) -> (u32, u32) {
    let h = ((height + factor - 1) / factor) * factor;
    let w = ((width + factor - 1) / factor) * factor;
    (w.max(factor), h.max(factor))
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
}
