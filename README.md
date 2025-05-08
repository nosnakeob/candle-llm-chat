# hf-llm-chat

一个基于 [Candle](https://github.com/huggingface/candle) 机器学习框架的 LLM 聊天机器人。
本项目旨在提供一个易于使用、支持流式输出和 GPU 加速的聊天机器人实现。主要参考了 [candle example for quantized-qwen2-instruct](https://github.com/huggingface/candle/blob/main/candle-examples/examples/quantized-qwen2-instruct/main.rs) 和 [candle_demo_openchat_35](https://github.com/rustai-solutions/candle_demo_openchat_35)，并在此基础上进行了简化和异步化改造。

## ✨ 功能特性

-   **模型支持**: 利用 Candle 框架，可支持多种 GGUF 格式的量化大语言模型。
-   **流式输出**: 实现打字机效果的实时响应，提升用户体验。
-   **GPU 加速**: 支持 CUDA，可利用 NVIDIA GPU 进行高效推理。
-   **异步处理**: 基于 Tokio 的异步设计，确保应用性能。
-   **配置灵活**: 可通过 `BaseConfig` 结构体轻松调整模型参数、采样设置等。

## 🚀 快速开始

### 1. 环境要求

-   Rust 工具链 (推荐最新稳定版)
-   CUDA 工具包 (若需使用 GPU 加速)

### 2. 下载与运行

```bash
git clone https://github.com/your-username/hf-llm-chat.git # 替换为您的仓库地址
cd hf-llm-chat
```

#### 设置代理 (可选)

如果在中国大陆或其他网络受限地区下载 Hugging Face 模型，可能需要设置代理。项目中已包含通过 `ProxyGuard` 设置代理的示例代码 (位于 `src/utils/mod.rs` 和测试用例中)。

```rust
// 示例：在代码中设置代理
let _proxy = ProxyGuard::new("http://127.0.0.1:10808"); // 替换为您的代理地址
```

### 3. 运行交互式聊天测试

```bash
cargo test --package hf-llm-chat --lib tests::test_chat -- --nocapture
```

此命令将启动一个交互式聊天会话，您可以输入文本与模型进行对话。

## ⚙️ 配置

项目的主要配置位于 `src/config.rs` 文件中的 `BaseConfig` 结构体。您可以修改以下参数：

-   `sample_len`: 生成响应的最大 token 数量。
-   `temperature`: 控制生成文本的随机性，较低的值使输出更确定。
-   `seed`: 随机数种子，用于可复现的输出。
-   `repeat_penalty`: 重复惩罚系数，避免生成重复内容。
-   `repeat_last_n`: 计算重复惩罚时考虑的先前 token 数量。
-   `which`: 选择要加载的具体模型 (定义在 `src/models/q_qwen2.rs` 或 `src/models/q_llama.rs` 等文件中)。

## 📦 GGUF 模型与分片处理

本项目支持 GGUF 格式的模型。对于分片的 GGUF 模型文件，需要使用 `llama-gguf-split` 工具进行合并。

### 依赖: `llama-gguf-split`

`llama-gguf-split` 是一个外部运行时依赖。如果需要加载分片模型，请确保已按照以下步骤安装并将其添加到系统 PATH：

1.  克隆 `llama.cpp` 仓库:
    ```bash
    git clone --recursive https://github.com/ggerganov/llama.cpp
    ```
2.  编译安装:
    ```bash
    cd llama.cpp
    cmake -S . -B build
    cmake --build build --config Release
    ```
3.  将生成的可执行文件 (通常在 `build/bin` 目录下) 添加到系统 PATH。

### 自动合并

程序在下载模型时，如果检测到模型文件是分片的，会自动调用 `llama-gguf-split` 进行合并。合并后的完整模型文件将保存在与分片文件相同的目录下。

参考资料:
- [How to use the gguf-split / Model sharding demo](https://github.com/ggml-org/llama.cpp/discussions/6404)

## 📝 许可证

本项目采用 MIT 许可证。详情请参阅 [LICENSE](LICENSE) 文件。

