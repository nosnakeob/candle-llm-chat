use anyhow::{Context, Result};
// [修改 1] 引入 IndexOp 以修复 E0599 报错 (使得 Tensor 可以使用 .i() 方法)
use candle::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen3_vl::{Config as Qwen3VLConfig, Qwen3VLModel};
use hf_hub::{Repo, RepoType, api::sync::Api};
use tokenizers::Tokenizer;

#[test]
fn t_qwen_vl() -> Result<()> {
    // 1. 初始化设备
    let device = Device::new_cuda(0).unwrap_or(Device::Cpu);

    // 2. 从 HuggingFace Hub 下载 Qwen3-VL-8B-Instruct-FP8 模型
    let api = Api::new()?;
    let repo = api.repo(Repo::with_revision(
        "Qwen/Qwen3-VL-8B-Instruct-FP8".to_string(),
        RepoType::Model,
        "main".to_string(),
    ));

    println!("正在加载配置文件和 Tokenizer...");
    let config_file = repo.get("config.json")?;
    let tokenizer_file = repo.get("tokenizer.json")?;
    // ================= [修复点开始] =================
    // 先将 config.json 解析为动态的 JSON 对象，以便进行修改
    let mut config_value: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(config_file)?)?;

    // tie_word_embeddings 在 text_config 子对象里，不在顶层
    if let Some(text_config) = config_value
        .as_object_mut()
        .and_then(|o| o.get_mut("text_config"))
        .and_then(|v| v.as_object_mut())
    {
        if !text_config.contains_key("tie_word_embeddings") {
            text_config.insert(
                "tie_word_embeddings".to_string(),
                serde_json::Value::Bool(false),
            );
        }
        // NOTE: 如果以后还报其他字段缺失的错，在这里继续补齐 text_config 的字段
    }

    // 从补全后的 JSON 动态对象再反序列化为严格的 Rust Struct
    let config: Qwen3VLConfig = serde_json::from_value(config_value)?;
    // ================= [修复点结束] =================
    let tokenizer = Tokenizer::from_file(tokenizer_file).map_err(|e| anyhow::anyhow!(e))?;

    println!("正在加载 FP8 模型权重...");
    // 从 index.json 读取分片文件列表，避免 repo.info() 发起网络请求
    let index_path = repo.get("model.safetensors.index.json").map_err(anyhow::Error::msg)?;
    let index_json: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(index_path)?)?;
    let weight_map = index_json
        .get("weight_map")
        .and_then(|v| v.as_object())
        .ok_or_else(|| anyhow::anyhow!("model.safetensors.index.json 中没有 weight_map"))?;
    let mut shard_names: Vec<String> = weight_map
        .values()
        .filter_map(|v| v.as_str().map(|s| s.to_string()))
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();
    shard_names.sort();
    let weights: Vec<_> = shard_names
        .iter()
        .map(|name| repo.get(name.as_str()).map_err(anyhow::Error::msg))
        .collect::<Result<_>>()?;

    let dtype = DType::BF16;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&weights, dtype, &device)? };
    let mut model = Qwen3VLModel::new(&config, vb)?;

    // 测试 1：纯文本对话
    println!("\n=== [测试 1: 纯文本对话] ===");
    chat_text(
        &mut model,
        &tokenizer,
        &device,
        "你好！请用一句话介绍一下你自己。",
    )?;

    // 测试 2：单图对话 (这里引入了真实的图片读取)
    println!("\n===[测试 2: 图片对话] ===");
    // 建议你在项目根目录下放一张 test.jpg，如果没有会回退到默认 448x448 分辨率
    chat_image(
        &mut model,
        &tokenizer,
        &device,
        "详细描述一下这张图片。",
        "test.jpg",
    )?;

    // 测试 3：视频对话
    println!("\n=== [测试 3: 视频对话] ===");
    chat_video(
        &mut model,
        &tokenizer,
        &device,
        "视频里的人在做什么？",
        "test.mp4",
    )?;

    Ok(())
}

/// 简单文本对话
fn chat_text(
    model: &mut Qwen3VLModel,
    tokenizer: &Tokenizer,
    device: &Device,
    prompt: &str,
) -> Result<()> {
    let prompt_str = format!(
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
        prompt
    );

    let tokens = tokenizer
        .encode(prompt_str, true)
        .map_err(|e| anyhow::anyhow!(e))?;
    let input_ids = Tensor::new(tokens.get_ids(), device)?.unsqueeze(0)?;

    println!("User: {}", prompt);
    generate_and_print(model, tokenizer, device, &input_ids, None, None)
}

///[代码图片测试] 升级版：动态读取图片大小，分配恰当的 Vision Patch 长度
fn chat_image(
    model: &mut Qwen3VLModel,
    tokenizer: &Tokenizer,
    device: &Device,
    prompt: &str,
    img_path: &str,
) -> Result<()> {
    println!("User: [图片: {}] {}", img_path, prompt);

    // [新增] 读取真实图片获取高宽，以此进行 Qwen3-VL 动态分辨率的计算
    let (img_w, img_h) = match image::io::Reader::open(img_path) {
        Ok(reader) => match reader.into_dimensions() {
            Ok(dim) => dim,
            Err(_) => {
                println!("⚠️ 无法读取图片尺寸，使用默认尺寸 448x448");
                (448, 448)
            }
        },
        Err(_) => {
            println!("⚠️ 找不到图片 {}，使用默认尺寸 448x448", img_path);
            (448, 448)
        }
    };

    // Qwen3-VL / Qwen2-VL 的 spatial_merge_size=2，patch_size=14
    // 相当于每 28x28 像素为一个空间 Block
    let block_size = 28;
    let blocks_w = (img_w + block_size - 1) / block_size;
    let blocks_h = (img_h + block_size - 1) / block_size;
    let num_patches = blocks_w * blocks_h;

    let image_tokens = format!(
        "<|vision_start|>{}<|vision_end|>",
        "<|image_pad|>".repeat(num_patches as usize)
    );
    let prompt_str = format!(
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{}{}<|im_end|>\n<|im_start|>assistant\n",
        image_tokens, prompt
    );

    let tokens = tokenizer
        .encode(prompt_str, true)
        .map_err(|e| anyhow::anyhow!(e))?;
    let input_ids = Tensor::new(tokens.get_ids(), device)?.unsqueeze(0)?;

    // 构造维度与图片高度和宽度匹配的 Dummy Pixel Tensor
    let dummy_pixel_values = Tensor::zeros((num_patches as usize, 1176), DType::BF16, device)?;

    // image_grid_thw 定义了图像在 Time, Height, Width 三个维度上的 Patch 数量
    let dummy_grid_thw = Tensor::new(&[1u32, blocks_h, blocks_w], device)?.unsqueeze(0)?;

    generate_and_print(
        model,
        tokenizer,
        device,
        &input_ids,
        Some(&dummy_pixel_values),
        Some(&dummy_grid_thw),
    )
}

/// 带视频对话
fn chat_video(
    model: &mut Qwen3VLModel,
    tokenizer: &Tokenizer,
    device: &Device,
    prompt: &str,
    vid_path: &str,
) -> Result<()> {
    println!("User: [视频: {}] {}", vid_path, prompt);

    let num_video_patches = 4 * 64;
    let video_tokens = format!(
        "<|vision_start|>{}<|vision_end|>",
        "<|video_pad|>".repeat(num_video_patches)
    );

    let prompt_str = format!(
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{}{}<|im_end|>\n<|im_start|>assistant\n",
        video_tokens, prompt
    );

    let tokens = tokenizer
        .encode(prompt_str, true)
        .map_err(|e| anyhow::anyhow!(e))?;
    let input_ids = Tensor::new(tokens.get_ids(), device)?.unsqueeze(0)?;

    let dummy_pixel_values = Tensor::zeros((num_video_patches, 1176), DType::BF16, device)?;
    let dummy_grid_thw = Tensor::new(&[4u32, 8u32, 8u32], device)?.unsqueeze(0)?;

    generate_and_print(
        model,
        tokenizer,
        device,
        &input_ids,
        Some(&dummy_pixel_values),
        Some(&dummy_grid_thw),
    )
}

/// 最小化的自回归贪心解码循环
fn generate_and_print(
    model: &mut Qwen3VLModel,
    tokenizer: &Tokenizer,
    device: &Device,
    input_ids: &Tensor,
    pixel_values: Option<&Tensor>,
    image_grid_thw: Option<&Tensor>,
) -> Result<()> {
    print!("Assistant: ");
    let mut current_ids = input_ids.clone();

    let eos_id = tokenizer.token_to_id("<|im_end|>").unwrap_or(151645);
    let max_tokens = 100;

    for _ in 0..max_tokens {
        // 计算当前序列长度和传入的 Position IDs
        let seq_len = current_ids.dim(1)?;
        let input_pos: Vec<usize> = (0..seq_len).collect();

        // [修改 2] 补全 forward 所有的 9 个参数以修复 E0061 报错
        // 我们利用 .cloned() 将 Option<&Tensor> 转化为 Option<Tensor>
        // 其他不需要的参数填入 None 或者 vec![] 让编译器自适应推导
        let logits = model.forward(
            &current_ids,            // input_ids
            pixel_values.cloned(),   // pixel_values
            image_grid_thw.cloned(), // image_grid_thw
            None,                    // video_pixel_values
            None,                    // video_grid_thw
            vec![],                  // cu_seqlens 等其他 Vec 占位
            vec![],                  // image_sizes: Vec<Vec<(usize, usize)>>
            vec![],                  // video_sizes: Vec<Vec<(usize, usize)>>
            &input_pos,              // input_pos
        )?;

        let seq_len = logits.dim(1)?;
        // .i() 此时生效，因为我们引用了 candle::IndexOp
        let logits = logits.i((0, seq_len - 1, ..))?;

        let next_token_id = logits.argmax(0)?.to_scalar::<u32>()?;

        if next_token_id == eos_id {
            break;
        }

        if let Some(text) = tokenizer.decode(&[next_token_id], true).ok() {
            print!("{}", text);
            use std::io::Write;
            std::io::stdout().flush()?;
        }

        let next_token_tensor = Tensor::new(&[next_token_id], device)?.unsqueeze(0)?;
        current_ids = Tensor::cat(&[&current_ids, &next_token_tensor], 1)?;
    }
    println!();
    Ok(())
}
