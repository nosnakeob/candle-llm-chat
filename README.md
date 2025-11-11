# candle-llm-chat

一个基于 [Candle](https://github.com/huggingface/candle) 机器学习框架的 LLM 聊天机器人。
本项目提供了一个易于使用、支持流式输出和 GPU 加速的聊天机器人实现，支持多种 GGUF 格式的量化大语言模型。采用现代化的 Rust 异步设计，具有简洁的模型配置系统。

## ✨ 功能特性

- **多模型支持**: 支持 Qwen2/Qwen3/Llama 系列模型，通过 `models.toml` 配置文件管理
- **简洁的 API**: 基于字符串标识符的模型选择，支持 `"qwen3"` 或 `"qwen3.W3_14b"` 格式
- **流式输出**: 实现打字机效果的实时响应，提升用户体验
- **GPU 加速**: 支持 CUDA，可利用 NVIDIA GPU 进行高效推理
- **异步处理**: 基于 Tokio 的异步设计，确保应用性能
- **智能聊天上下文**: 自动角色切换和思考过程过滤的 `ChatContext` 管理
- **配置灵活**: 通过 `InferenceConfig` 结构体和 TOML 文件轻松调整模型参数

## 🚀 快速开始

### 1. 环境要求

- Rust 工具链 (推荐最新稳定版)
- CUDA 工具包 (若需使用 GPU 加速)

### 2. 下载与运行

```bash
git clone https://github.com/your-username/candle-llm-chat.git # 替换为您的仓库地址
cd candle-llm-chat
```

#### 设置代理 (可选)

如果在中国大陆或其他网络受限地区下载 Hugging Face 模型，可能需要设置代理。项目提供了 `ProxyGuard` 工具类：

```rust
use candle_llm_chat::utils::proxy::ProxyGuard;

// 设置代理，ProxyGuard 会在作用域结束时自动清理
let _proxy = ProxyGuard::new("7890"); // 端口号，完整地址为 http://127.0.0.1:7890
```

`ProxyGuard` 实现了 RAII 模式，会在析构时自动清理环境变量。

### 3. 运行测试

**交互式聊天测试**：

```bash
cargo test --package candle-llm-chat --lib pipe::tests::test_pipeline -- --nocapture
```

**预设对话测试**：

```bash
cargo test --package candle-llm-chat --lib pipe::tests::test_prompt -- --nocapture
```

这些测试将演示模型加载、聊天上下文管理和流式输出功能。

### 4. 模型配置

项目支持多种预配置模型，在 `models.toml` 中定义：

- **Qwen2 系列**: 1.5B, 7B, 14B 参数模型
- **Qwen3 系列**: 4B, 8B, 14B, 32B 参数模型
- **Llama 系列**: 包含 DeepSeek-R1-Distill-Llama-8B

默认使用 Qwen3-8B 模型，可通过模型标识符切换：

```rust
// 使用默认模型 (qwen3-8B)
let text_gen = TextGeneration::default().await?;

// 使用架构的默认模型
let text_gen = TextGeneration::with_default_config("qwen2").await?;

// 使用特定模型变体
let text_gen = TextGeneration::with_default_config("qwen3.W3_14b").await?;
```

## ⚙️ 配置

### 主要配置文件

**`src/model/config.rs`** - 核心配置结构：

```rust
pub struct InferenceConfig {
    pub sample_len: usize,      // 生成响应的最大 token 数量 (默认: 1000)
    pub temperature: f64,       // 控制随机性 (默认: 0.8)
    pub top_p: Option<f64>,     // Nucleus 采样概率
    pub seed: u64,              // 随机种子 (默认: 299792458)
    pub repeat_penalty: f32,    // 重复惩罚系数 (默认: 1.1)
    pub repeat_last_n: usize,   // 重复惩罚上下文长度 (默认: 64)
    pub device: Device,         // 计算设备 (CPU/CUDA)
}
```

**`models.toml`** - 模型仓库配置：

```toml
[qwen3.W3_8b]
model_repo = "Qwen/Qwen3-8B-GGUF"
model_file = "Qwen3-8B-Q4_K_M"
tokenizer_repo = "Qwen/Qwen3-8B"
default = true
```

**`config.toml`** - HuggingFace 访问令牌等全局配置

### 使用示例

**基本聊天流式输出**：

```rust
use candle_llm_chat::pipe::TextGeneration;
use futures_util::{StreamExt, pin_mut};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 使用默认配置 (Qwen3-8B)
    let mut text_gen = TextGeneration::default().await?;

    // 流式聊天
    let stream = text_gen.chat("你好，请介绍一下自己");
    pin_mut!(stream);

    while let Some(Ok(token)) = stream.next().await {
        print!("{}", token);
    }

    Ok(())
}
```

**选择特定模型**：

```rust
use candle_llm_chat::pipe::TextGeneration;

// 使用 Qwen2 默认模型 (7B)
let text_gen = TextGeneration::with_default_config("qwen2").await?;

// 使用 Qwen3-14B 模型
let text_gen = TextGeneration::with_default_config("qwen3.W3_14b").await?;

// 使用 DeepSeek-R1-Llama-8B 模型
let text_gen = TextGeneration::with_default_config("llama.DeepseekR1Llama8b").await?;
```

**自定义生成参数**：

```rust
use candle_llm_chat::model::config::InferenceConfig;
use candle_llm_chat::pipe::TextGeneration;

let mut config = InferenceConfig::default();
config.temperature = 0.7;
config.sample_len = 2000;
config.repeat_penalty = 1.2;

let mut text_gen = TextGeneration::new("qwen3", config).await?;
```

## 📦 GGUF 模型与分片处理

本项目支持 GGUF 格式的模型。对于分片的 GGUF 模型文件，需要使用 `gguf-utils` 工具进行合并。

### 依赖: `gguf-utils`

`gguf-utils` 是一个外部运行时依赖。如果需要加载分片模型，请确保已安装：

```bash
cargo install gguf-utils
```

安装后，`gguf-utils` 命令将自动添加到系统 PATH。

### 自动合并

程序在下载模型时，如果检测到模型文件是分片的，会自动调用 `gguf-utils merge` 进行合并。合并后的完整模型文件将保存在与分片文件相同的目录下。

## 🏗️ 项目架构

```mermaid
graph TB
    subgraph "用户交互层"
        A[用户输入] --> B[TextGeneration::chat]
    end

    subgraph "配置管理层"
        MR[ModelRegistry<br/>模型注册表] --> HI[HubInfo<br/>模型仓库信息]
        MT[models.toml] --> MR
        HI --> ML[ModelLoader<br/>模型加载器]
    end

    subgraph "核心组件"
        C[ChatContext<br/>聊天上下文管理] --> TG[TextGeneration<br/>文本生成管道]
        ML --> TG
        IC[InferenceConfig<br/>推理配置] --> TG
        F[TokenOutputStream<br/>Token流处理] --> TG
        G[LogitsProcessor<br/>采样处理] --> TG
    end

    subgraph "模型抽象层"
        FW[Forward Trait<br/>统一推理接口] --> MW[ModelWeights实现]
        MW --> MW1[quantized_qwen2::ModelWeights]
        MW --> MW2[quantized_qwen3::ModelWeights]
        MW --> MW3[quantized_llama::ModelWeights]
    end

    subgraph "模型实现层"
        MW1 --> H1[Qwen2 GGUF模型文件]
        MW2 --> H2[Qwen3 GGUF模型文件]
        MW3 --> H3[Llama GGUF模型文件]
        I[Tokenizer<br/>分词器] --> TG
    end

    subgraph "底层框架"
        K[Candle Framework<br/>机器学习框架]
        L[CUDA Support<br/>GPU加速]
        M[HuggingFace Hub<br/>模型仓库]
    end

    subgraph "工具组件"
        N[ProxyGuard<br/>代理设置] --> M
        O[gguf-utils<br/>模型分片合并] --> H1
        O --> H2
        O --> H3
    end

    B --> C
    TG --> P[Stream Output<br/>流式输出]
    P --> Q[实时响应显示]

    H1 --> M
    H2 --> M
    H3 --> M
    I --> M
    MW1 --> K
    MW2 --> K
    MW3 --> K
    K --> L

    style MR fill:#fff3e0
    style HI fill:#e3f2fd
    style FW fill:#f1f8e9
    style C fill:#e1f5fe
    style TG fill:#f3e5f5
    style P fill:#e8f5e8
```

### 架构说明

#### 核心设计模式

**1. 简化的配置系统**

- `ModelRegistry`: 从 `models.toml` 加载模型配置的注册表系统
- `HubInfo`: 包含模型仓库、文件名和分词器仓库的配置结构
- `InferenceConfig`: 推理参数配置，包含温度、采样长度等
- `ModelLoader`: 统一的模型加载器，负责加载模型、分词器和元数据

**2. 模型标识符系统**

```rust
// 支持两种格式：
// 1. 架构名 - 使用该架构的默认模型
let text_gen = TextGeneration::with_default_config("qwen3").await?;

// 2. 架构名.变体名 - 使用特定模型变体
let text_gen = TextGeneration::with_default_config("qwen3.W3_14b").await?;
```

**3. 统一推理接口**

```rust
pub trait Forward {
    fn forward(&mut self, x: &Tensor, index_pos: usize) -> Result<Tensor>;
}

// 通过宏为所有模型权重实现 Forward trait
impl_model_traits!(quantized_llama, quantized_qwen2, quantized_qwen3);
```

#### 核心流程

1. **配置加载** → `ModelRegistry` 从 `models.toml` 读取模型配置
2. **模型选择** → 通过字符串标识符 (如 `"qwen3"` 或 `"qwen3.W3_14b"`) 选择模型
3. **异步加载** → `ModelLoader::load()` 异步加载 GGUF 模型、分词器和元数据
4. **推理执行** → `Forward` trait 统一推理接口
5. **流式输出** → `TextGeneration::chat()` 返回异步流

#### 关键组件

- **ModelRegistry**: TOML 配置文件驱动的模型注册表，支持默认模型和变体选择
- **HubInfo**: 封装 HuggingFace 模型仓库信息，负责下载模型和分词器
- **ModelLoader**: 统一的模型加载器，返回 `(Box<dyn Forward>, Tokenizer, ModelInfo)` 元组
- **ModelInfo**: 从 GGUF 文件元数据提取的模型信息 (架构、EOS token、聊天模板)
- **InferenceConfig**: 推理参数配置，支持自定义温度、采样长度等
- **ChatContext**: 智能聊天上下文，自动角色切换和思考过程过滤
- **TextGeneration**: 核心文本生成管道，支持流式输出
- **宏系统**: `impl_model_traits!` 自动为模型实现必要 trait

#### 技术特性

- 🎯 **简洁 API**: 基于字符串的模型选择，无需复杂的枚举类型
- 🔧 **配置驱动**: 通过 TOML 文件管理模型，易于扩展新模型
- 🔄 **异步优先**: 全异步设计，模型加载和推理均为异步
- 🚀 **GPU 加速**: 自动检测 CUDA 设备，提升推理性能
- 📡 **流式输出**: 基于 async-stream 的实时响应
- 🛠️ **简化架构**: 移除复杂的泛型系统，采用更直观的字符串标识符

## 📁 项目结构

```
src/
├── lib.rs                 # 库入口
├── pipe.rs                # TextGeneration 核心管道
├── model/
│   ├── mod.rs            # Forward trait 和宏定义
│   ├── config.rs         # InferenceConfig 和 ModelLoader
│   ├── registry.rs       # ModelRegistry 模型注册表
│   └── hub.rs            # HubInfo 和 ModelInfo
├── utils/
│   ├── mod.rs            # 工具函数
│   ├── load.rs           # 模型和分词器下载
│   ├── chat.rs           # ChatContext 聊天上下文
│   └── proxy.rs          # ProxyGuard 代理管理

配置文件:
├── models.toml           # 模型仓库配置
├── config.toml           # 全局配置 (HF token 等)
└── Cargo.toml            # 项目依赖
```

## 🔧 扩展新模型

添加新模型只需要两步：

1. **在 `models.toml` 中添加配置**：

```toml
[qwen3.W3_72b]
model_repo = "Qwen/Qwen3-72B-GGUF"
model_file = "Qwen3-72B-Q4_K_M"
tokenizer_repo = "Qwen/Qwen3-72B"
```

2. **在代码中使用**：

```rust
let text_gen = TextGeneration::with_default_config("qwen3.W3_72b").await?;
```

对于新的模型架构，需要：

- 在 `models.toml` 中添加新的架构部分 (如 `[new_arch.variant]`)
- 在 `ModelLoader::load()` 中添加对应的加载逻辑
- 确保 Candle 框架支持该模型架构

## 📝 许可证

本项目采用 MIT 许可证。详情请参阅 [LICENSE](LICENSE) 文件。
