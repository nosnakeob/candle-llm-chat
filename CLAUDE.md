# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test Commands

```bash
cargo build                    # build library
cargo test                     # run unit tests only (no model downloads)
cargo test -- --ignored        # run integration tests (requires GPU + network + model files)
cargo test -- --nocapture      # show stdout/stderr during tests
cargo test --lib <module>::tests::<test_name> -- --nocapture  # single test

# Run interactive pipeline test (manual input)
cargo test --lib pipe::tests::test_pipeline -- --nocapture --ignored

# Run env guard mirror test
cargo test --lib utils::env_guard::tests::test_hf_endpoint_mirror_reachable -- --nocapture --ignored
```

No linter/formatter is enforced currently. `[lints.rust] unused = "allow"` in Cargo.toml suppresses unused code warnings globally.

## Architecture Overview

**Layer stack:** Agent → TextGeneration → ModelInference → Candle/HF Hub

### Key Components

| Component | File | Role |
|-----------|------|------|
| `TextGeneration` | `src/pipe.rs` | Core inference pipeline: chat template rendering → tokenization → autoregressive decode → TokenOutputStream. Provides `chat()` (streaming) and `chat_full()` (complete response for Agent loops). |
| `Agent` | `src/agent.rs` | Wraps TextGeneration with tool-calling loop. Parses `<tool_call>JSON</tool_call>` from model output, executes tools, injects results back into context, loops until no more tool calls (max 10 rounds). |
| `ModelInference` trait | `src/model/mod.rs` | Trait + macro unifying GGUF (`quantized_qwen3`) and Safetensors (`qwen3::ModelForCausalLM`) inference. |
| `ModelLoader` | `src/model/config.rs` | Dispatches to `load_gguf()` or `load_safetensors()` based on repo name containing "gguf". |
| `ModelRegistry` | `src/model/registry.rs` | Parses `models.toml` with smart tokenizer_repo auto-fill (base → variant inheritance). |
| `ChatContext` | `src/utils/chat.rs` | Multi-turn message history + MiniJinja template rendering. Auto-role-switching (User→Assistant→User). Strips `<think>` blocks on push. |
| `Tool` trait + `ToolRegistry` | `src/tools/*` | Pluggable tool interface. `FileReadTool` (with path traversal protection), `WebFetchTool`. |
| `ToolCallParser` | `src/tools/parse.rs` | Regex-based extraction of tool call JSON from model output, with bare-JSON fallback. |
| `Qwen3VL` | `src/qwen3_vl.rs` | Multimodal stub — text path works, image pipeline not yet implemented (uses dummy zero tensors). |

### Data Flow (Agent with tools)

```
User prompt → Agent::chat()
  → pipe.inject_system_prompt(tool_descriptions)   # once
  → pipe.push_user_message(prompt)
  → pipe.chat_full()                                # model generates
  → ToolCallParser::parse(raw_response)
  → if tool calls found:
      → registry.execute(name, args)
      → pipe.push_tool_result(result)
      → pipe.push_assistant_continuation()
      → loop back to chat_full()
  → if no tool calls:
      → ToolCallParser::strip(raw_response)
      → yield cleaned text to user
```

### Data Flow (plain chat)

```
User prompt → TextGeneration::chat()
  → ChatContext::push_msg()             # auto-role, strip <think>
  → ChatContext::render()               # MiniJinja template
  → tokenize → autoregressive loop
  → yield tokens via async-stream
  → ChatContext::push_msg(answer)       # append to history
```

### Test Strategy

- **Unit tests** (no `#[ignore]`): runnable without GPU/network — registry parsing, tool call parsing, file read, HTML extraction, chat template rendering. These run in CI with `cargo test`.
- **Integration tests** (`#[ignore]`): require actual model downloads, GPU, or network access. Run manually with `cargo test -- --ignored`.

### Model Configuration

Models defined in `models.toml` with naming: `arch.size_variant`. Supported suffixes: `*_base` (Safetensors), `*_q4`/`*_q8` (GGUF quantized). Tokenizer repos auto-inherit from corresponding `*_base`.

Key env vars: `HF_ENDPOINT` (mirror), `HF_TOKEN` (auth), `HTTPS_PROXY` (proxy). Use `ProxyGuard` / `HfEndpointGuard` (RAII) in tests.

### Important gotchas

- All model-loading tests are `#[ignore]` because they download GB-scale files and require CUDA
- `Cargo.lock` is in `.gitignore` (not committed)
- `gguf-utils` CLI binary is required for GGUF shard merging (installed separately via `cargo install gguf-utils`)
- The `[lints.rust] unused = "allow"` in Cargo.toml suppresses dead code warnings globally
