pub mod chat;
pub mod env_guard;
pub mod hf;
pub mod load;

use candle::quantized::gguf_file::Content;
use tracing::info;

pub fn format_size(size_in_bytes: usize) -> String {
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

/// 计算并记录 GGUF 文件中张量的总大小信息
pub fn log_tensor_size(ct: &Content) {
    let mut total_size_in_bytes = 0;
    for (_, tensor) in ct.tensor_infos.iter() {
        let elem_count = tensor.shape.elem_count();
        total_size_in_bytes +=
            elem_count * tensor.ggml_dtype.type_size() / tensor.ggml_dtype.block_size();
    }
    let formatted_size = format_size(total_size_in_bytes);
    info!(
        "loaded {:?} tensors ({})",
        ct.tensor_infos.len(),
        &formatted_size
    );
}
