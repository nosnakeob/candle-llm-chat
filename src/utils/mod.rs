pub mod chat;
pub mod load;

use candle::quantized::gguf_file::Content;
use std::io::BufRead;
use std::{env, io};
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

pub fn get_user_prompt() -> String {
    println!("请输入您的问题:");
    let stdin = io::stdin();
    let mut line = String::new();
    stdin
        .lock()
        .read_line(&mut line)
        .expect("Failed to read line");

    // 去除末尾的换行符
    line = line.trim().to_string();

    line
}

pub struct ProxyGuard;

impl ProxyGuard {
    pub fn new(port: &str) -> Self {
        unsafe {
            env::set_var("HTTPS_PROXY", format!("http://127.0.0.1:{port}"));
        }
        Self
    }
}

impl Drop for ProxyGuard {
    fn drop(&mut self) {
        unsafe {
            env::remove_var("HTTPS_PROXY");
        }
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
