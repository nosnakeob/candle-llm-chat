# CLAUDE.md

## 思维规则

遇到问题时先用中文思考和分析，理清思路后再输出英文代码或文档。分析过程使用中文，保持思维连贯性。

## 项目概述

基于 [candle](https://github.com/huggingface/candle) 框架的 Rust 本地 LLM 推理引擎，支持 Qwen3 系列模型的 GGUF 量化版和 Safetensors 完整版推理，集成 Agent 工具调用和 Qwen3-VL 多模态图文对话。

## 项目边界（vibe coding 纲领）

本项目是 vibe coding 实验，AI 与开发者协作推进。以下三层边界定义了什么值得做、什么必须做、什么已经能做：

### 要做的事（Goals — 方向感）

探索 Rust + candle 框架在本地 LLM 推理上的可能性，以最小可行路径验证想法，不追求生产级完备性。具体来说：

1. **玩模型** — 把各种架构的模型（GGUF / Safetensors）跑起来，观察效果，理解差异
2. **玩工具调用** — 让模型使用外部工具（读文件、抓网页等），探索 Agent 能力边界
3. **玩多模态** — 图文理解、视觉推理，拓展模型输入维度
4. **积累可复用的实现片段** — 代码是为学习和实验服务，不是为了构建产品

### 需要做到的事（Requirements — 底线）

不做"系统"或"平台"，但要满足以下条件才算实验成功：

1. **模型推理正确** — 至少一个模型可完整跑通：prompt → 推理 → 输出，结果合理
2. **Agent 循环可用** — 模型能识别工具调用、执行、拿到结果后继续对话
3. **不依赖外部 API** — 推理完全本地运行（HuggingFace 只用于下载模型文件）
4. **代码可编译** — `cargo build` 和 `cargo test` 通过（集成测试除外）

### 能做到的事（Capabilities — 当前状态）

以下能力已经实现，可以直接使用或在此基础上扩展：

- **模型加载**：自动识别 GGUF / Safetensors 格式，Qwen3 全系列 4B~32B
- **硬件加速**：CUDA + flash-attn（`InferenceConfig` 默认 GPU）
- **聊天管线**：`TextGeneration::chat()` 流式输出，`chat_full()` 完整响应
- **多轮对话**：`ChatContext` 管理历史 + MiniJinja chat template 渲染，自动角色切换
- **Agent 工具调用**：`Agent::chat()` 解析 `<tool_call>`，最大 10 轮循环，带退化为普通聊天的兜底
- **文件读取工具**：路径穿越防护，支持 `.eml` 等格式
- **网页抓取工具**：HTML → Markdown 提取
- **模型配置注册表**：`models.toml` + `ModelRegistry`，tokenizer_repo 自动继承

### 不做什么（Anti-goals）

以下方向明确排除在项目边界之外：

- **不做 web 服务 / API 服务器** — 没有 HTTP 接口，没有数据库，没有用户管理
- **不做生产部署** — 不需要高并发、负载均衡、监控告警、容器化
- **不做模型训练 / 微调 / LoRA** — 只做推理
- **不做平台级抽象** — 不需要插件系统、动态加载、热更新
- **不做跨模型兼容适配层** — 能跑哪个模型就写哪个模型的代码，不追求统一接口覆盖所有架构

## 技术栈

- 语言：Rust (edition 2024)
- ML 框架：candle-core 0.10 + candle-transformers 0.10
- 模型格式：GGUF（量化）、Safetensors（完整）、BF16
- 硬件加速：CUDA（主）、CUDNN、flash-attn
- 异步运行时：Tokio
- 模板引擎：MiniJinja（Chat template 渲染）
- 包管理：Cargo（无 Cargo.lock）

## 常用命令

```bash
cargo build                    # 编译库
cargo test                     # 运行单元测试（无需网络/GPU）
cargo test -- --ignored        # 运行集成测试（需 GPU + 网络 + 模型文件）
cargo test -- --nocapture      # 显示 stdout/stderr
cargo test --lib <module>::tests::<test_name> -- --nocapture  # 单测
```

## 架构总览

**分层栈：** Agent → TextGeneration → ModelInference → Candle/HF Hub

### 核心组件

| 组件 | 文件 | 职责 |
|------|------|------|
| `TextGeneration` | `src/pipe.rs` | 推理主管线：chat 模板 → tokenize → 自回归解码 → TokenOutputStream |
| `Agent` | `src/agent.rs` | 工具调用循环，解析 `<tool_call>JSON</tool_call>`，最大 10 轮 |
| `ModelInference` trait | `src/model/mod.rs` | GGUF + Safetensors 统一推理 trait |
| `ModelLoader` | `src/model/config.rs` | 根据仓库名含 "gguf" 自动分发到 load_gguf / load_safetensors |
| `ModelRegistry` | `src/model/registry.rs` | 解析 models.toml，tokenizer_repo 自动继承（base → variant） |
| `ChatContext` | `src/utils/chat.rs` | 多轮对话历史 + MiniJinja 模板渲染，自动角色切换，剥离 `<think>` |
| `Qwen3VL` | `src/qwen3_vl.rs` | 多模态独立管线，图像预处理 + 图文对话，不通过 ModelInference trait |
| `Tool` trait + `ToolRegistry` | `src/tools/*` | 可插拔工具接口，FileReadTool（路径穿越防护）+ WebFetchTool |
| `ToolCallParser` | `src/tools/parse.rs` | 正则提取 + 裸 JSON 回退解析工具调用 |

### 数据流（Agent + 工具）

```
User prompt → Agent::chat()
  → pipe.inject_system_prompt(tool_descriptions)   # 仅一次
  → pipe.push_user_message(prompt)
  → pipe.chat_full()                                # 模型生成
  → ToolCallParser::parse(raw_response)
  → 如果有工具调用：
      → registry.execute(name, args)
      → pipe.push_tool_result(result)
      → pipe.push_assistant_continuation()
      → 继续 chat_full()
  → 如果没有工具调用：
      → ToolCallParser::strip(raw_response)
      → yield 清洗后的文本给用户
```

### 数据流（纯聊天）

```
User prompt → TextGeneration::chat()
  → ChatContext::push_msg()             # 自动角色切换，剥离 <think>
  → ChatContext::render()               # MiniJinja 模板
  → tokenize → 自回归解码循环
  → yield tokens via async-stream
  → ChatContext::push_msg(answer)       # 追加到历史
```

## 编码规范

- 命名：Rust 标准命名风格（snake_case 变量/函数，PascalCase 类型/ trait）
- 错误处理：使用 `anyhow::Result` / `anyhow::Error`，避免 `unwrap()`/`expect()`
- 注释：不写 WHAT（代码本身表达意图），仅在 WHY 不明确时加注释（隐藏约束、微妙不变量、workaround）
- 导入顺序：标准库 → 第三方 crate → 内部模块（空行分隔）
- 新增功能不要引入不必要的抽象、重构或超前设计

## 测试策略

- **单元测试**（无 `#[ignore]`）：无需 GPU/网络，`cargo test` 可运行。覆盖 registry 解析、工具调用解析、文件读取、HTML 提取、chat 模板渲染
- **集成测试**（`#[ignore]`）：需实际模型下载、GPU 或网络。手动运行 `cargo test -- --ignored`

## 模型配置

定义在 `models.toml`，命名格式：`arch.size_variant`。支持后缀：
- `*_base` — Safetensors 完整版
- `*_q4` / `*_q8` — GGUF 量化版
- `*_abliterated` — 社区去审查版

Tokenizer repo 自动从对应 `*_base` 继承。

## 操作禁区

- 不要修改 `models.toml` 中非当前任务相关的模型配置
- 不要提交 `.env`、模型权重文件、大型二进制文件到 git
- 不要运行 git push --force，除非用户明确要求
- 不要修改 `[lints.rust]` 配置（全局抑制 unused 警告是故意的）
- 不要删除 `Cargo.lock` 的 `.gitignore` 条目（不锁定依赖版本）

## 注意事项

- 所有模型加载测试都是 `#[ignore]`（需下载 GB 级文件 + CUDA）
- `Cargo.lock` 不提交（在 `.gitignore` 中）
- `gguf-utils` CLI 需要单独安装：`cargo install gguf-utils`
- 关键环境变量：`HF_ENDPOINT`（镜像）、`HF_TOKEN`（认证）、`HTTPS_PROXY`（代理）
- 集成测试中使用 `ProxyGuard` / `HfEndpointGuard`（RAII 模式）管理环境变量
- Qwen3-VL 不通过 `ModelInference` trait 集成（forward 签名不兼容），作为独立管线
- flash-attn feature 需要代理才能编译（需 clone NVIDIA cutlass）
