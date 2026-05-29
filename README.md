# candle-llm-chat

基于 [Candle](https://github.com/huggingface/candle) 框架的 **Rust 本地 LLM 推理引擎**，支持 Qwen3 系列的 GGUF 量化版和 Safetensors 完整版推理，内置 Agent 工具调用循环，以及 Qwen3-VL 多模态图文对话。

[![Rust](https://img.shields.io/badge/rust-2024%20edition-orange)](https://www.rust-lang.org/)
[![Candle](https://img.shields.io/badge/candle-0.10-blue)](https://github.com/huggingface/candle)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 特性

| 特性 | 说明 |
|------|------|
| **简洁 API** | 字符串标识符选择模型：`"qwen3"` / `"qwen3.8b_q4"` |
| **多模型支持** | Qwen3 系列（4B/8B/14B/32B），通过 `models.toml` 配置 |
| **双格式支持** | GGUF 量化 + Safetensors 完整版，自动识别分发 |
| **流式输出** | 基于 `async-stream` 的实时 Token 流 |
| **GPU 加速** | CUDA / cuDNN / flash-attn |
| **异步设计** | 全栈 Tokio 异步，模型加载和推理均非阻塞 |
| **聊天上下文** | MiniJinja 模板渲染，自动角色切换，`<think>` 隔离 |
| **Agent 工具调用** | `<tool_call>JSON</tool_call>` 解析，最多 10 轮循环 |
| **内置工具** | 文件读取（路径穿越防护）、网页抓取（HTML → Markdown） |
| **多模态** | Qwen3-VL 文本推理已就绪，图像输入接口预留 |

---

## 快速开始

### 环境要求

- Rust 工具链（edition 2024）
- CUDA 工具包（可选，GPU 加速用）
- `gguf-utils`（可选，分片 GGUF 合并用）：`cargo install gguf-utils`

### 环境变量（国内用户）

```powershell
# Windows PowerShell
$env:HF_ENDPOINT = "https://hf-mirror.com"
```

```bash
# Linux / macOS
export HF_ENDPOINT="https://hf-mirror.com"
```

### 基本使用

```rust
use candle_llm_chat::pipe::TextGeneration;
use futures_util::{StreamExt, pin_mut};

// 使用默认模型（qwen3.8b_q4）
let mut gen = TextGeneration::default().await?;

let stream = gen.chat("介绍一下自己");
pin_mut!(stream);

while let Some(Ok(token)) = stream.next().await {
    print!("{}", token);
}
```

### Agent 模式

```rust
use candle_llm_chat::agent::Agent;
use futures_util::{StreamExt, pin_mut};

let mut agent = Agent::new().await?;

let stream = agent.chat("访问 https://example.com 并总结页面内容");
pin_mut!(stream);

while let Some(Ok(token)) = stream.next().await {
    print!("{token}");
}
// Agent 内部自动执行：模型输出 tool_call → 系统抓取网页 → 结果注入 → 模型继续输出
```

---

## 数据流

### Agent + 工具调用

```
User prompt → Agent::chat()
  → pipe.inject_system_prompt(tool_descriptions)   # 仅一次
  → pipe.push_user_message(prompt)
  → pipe.chat_full()                                # 模型生成
  → ToolCallParser::parse(raw_response)
  → 有工具调用：
      → registry.execute(name, args)
      → pipe.push_tool_result(result)
      → pipe.push_assistant_continuation()
      → 继续 chat_full()
  → 无工具调用：
      → ToolCallParser::strip(raw_response)
      → yield 清洗后的文本给用户
```

### 纯聊天

```
User prompt → TextGeneration::chat()
  → ChatContext::push_msg()             # 自动角色切换，剥离 <think>
  → ChatContext::render()               # MiniJinja 模板
  → tokenize → 自回归解码循环
  → yield tokens via async-stream
  → ChatContext::push_msg(answer)       # 追加到历史
```

---

## 项目结构

```
src/
├── lib.rs              # 库入口
├── pipe.rs             # TextGeneration — 推理主管线（流式 + 完整响应）
├── agent.rs            # Agent — 工具调用循环，最大 10 轮
├── qwen3_vl.rs         # Qwen3-VL 多模态独立管线
├── model/
│   ├── mod.rs          # ModelInference trait + impl_model_traits! 宏
│   ├── config.rs       # ModelLoader — GGUF / Safetensors 自动分发
│   ├── registry.rs     # ModelRegistry — models.toml 解析
│   └── hub.rs          # HubInfo, ModelArch 类型定义
├── tools/
│   ├── mod.rs          # Tool trait + ToolDef
│   ├── parse.rs        # ToolCallParser — 正则 + 裸 JSON 回退
│   ├── registry.rs     # ToolRegistry — 注册 + 执行
│   ├── file_read.rs    # FileReadTool — 路径穿越防护
│   └── web_fetch.rs    # WebFetchTool — HTML → Markdown
└── utils/
    ├── chat.rs         # ChatContext — MiniJinja 模板渲染
    ├── load.rs         # 模型/Tokenizer 下载，分片合并
    └── env_guard.rs    # ProxyGuard / HfEndpointGuard RAII
```

---

## 配置

### 选择模型

```rust
// 架构默认模型
let gen = TextGeneration::with_default_config("qwen3").await?;

// GGUF 量化版（体积小、速度快）
let gen = TextGeneration::with_default_config("qwen3.4b_q4").await?;

// Safetensors 完整版（精度更高）
let gen = TextGeneration::with_default_config("qwen3.8b_base").await?;

// 用 Agent 指定模型
let mut agent = Agent::with_model("qwen3.4b_q4").await?;
```

### 自定义推理参数

```rust
use candle_llm_chat::model::config::InferenceConfig;

let config = InferenceConfig {
    temperature: 0.7,
    sample_len: 2000,
    repeat_penalty: 1.1,
    ..Default::default()
};

let mut gen = TextGeneration::new("qwen3.8b_q4", config).await?;
```

### 自定义工具

```rust
use candle_llm_chat::tools::{Tool, ToolDef};

struct MyTool;

impl Tool for MyTool {
    fn name(&self) -> &str { "my_tool" }
    fn definition(&self) -> ToolDef {
        ToolDef::new("my_tool", "我的自定义工具", serde_json::json!({
            "type": "object",
            "properties": {
                "input": {"type": "string"}
            }
        }))
    }
    fn execute(&self, args: &serde_json::Value) -> anyhow::Result<String> {
        Ok(format!("收到: {}", args))
    }
}

let mut agent = Agent::with_default_tools().await?;
agent.register_tool(MyTool)?;
```

### 网络代理

```rust
use candle_llm_chat::utils::env_guard::ProxyGuard;

let _proxy = ProxyGuard::new(7897);  // Drop 时自动恢复
```

---

## 实现状态

### ✅ 已实现

- **模型加载**：GGUF 量化 + Safetensors 完整版，自动识别分发
- **模型注册表**：`models.toml` → `ModelRegistry`，`tokenizer_repo` 自动继承
- **流式聊天**：`TextGeneration::chat()` 基于 `async-stream`
- **完整响应**：`chat_full()` 供 Agent 内部调用
- **多轮对话**：`ChatContext` + MiniJinja 模板，自动角色切换，`<think>` 剥离
- **推理参数**：温度、采样长度、重复惩罚、KV Cache
- **Agent 工具调用**：`Agent::chat()` 自动解析 `<tool_call>`，最多 10 轮循环，无工具时退化为普通聊天
- **文件读取工具**：路径穿越防护（canonicalize），支持 `.eml`
- **网页抓取工具**：HTML → Markdown 提取
- **Qwen3-VL 文本推理**：`Qwen3VL::chat()` 纯文本路径已通
- **网络工具**：`ProxyGuard` / `HfEndpointGuard` RAII 模式

### 🚧 部分实现

- **Llama 系列**：配置节已预留（`models.toml`），`ModelLoader` 中 bail 尚未解除
- **Qwen3-VL 图像输入**：`model.forward()` 接口就绪，`pixel_values` 预处理管线缺失

### ⏳ 待验证（需模型文件 + GPU）

- 跑通第一个模型（4B GGUF 量化版）
- Agent 端到端验证（集成测试已写，`#[ignore]`）

---

## 测试

```bash
# 单元测试（无需网络/GPU）
cargo test

# 集成测试（需模型文件 + GPU + 网络）
cargo test -- --ignored

# 带输出
cargo test -- --nocapture
```

所有单元测试都不需要模型文件、GPU 或网络。集成测试有 `#[ignore]` 标记，安全隔离。

---

## 许可证

MIT
