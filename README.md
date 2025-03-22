# Rust OpenChat Bot

基于 [Candle](https://github.com/huggingface/candle) 机器学习框架的OpenChat聊天机器人实现。本项目在 [candle_demo_openchat_35](https://github.com/rustai-solutions/candle_demo_openchat_35) 的基础上进行了进一步封装,将其简化为易用的机器人结构体。

## 功能特性

- 基于OpenChat 3.5模型
- 支持CUDA加速
- 流式输出响应
- 简单易用的API接口

## 快速开始

### 1. 环境要求

- Rust 工具链
- Python 3.x
- CUDA 工具包 (用于GPU加速)

### 2. 下载权重

1. 修改`download_model.py`, `arg::Args::model`中的openchat文件名
2. 运行`python download_model.py`

### 3. 运行

```
cargo run --release
```

## 配置说明

可以通过修改 `src/arg.rs` 中的 `Args` 结构体来自定义以下参数:

- `sample_len`: 生成的最大token数量
- `temperature`: 采样温度
- `seed`: 随机数种子
- `repeat_penalty`: 重复惩罚系数
- `repeat_last_n`: 重复检查的历史长度

## GGUF 模型合并

某些大型模型文件可能会被分片存储，需要在使用前进行合并。本项目使用 llama-gguf-split 工具来处理模型合并。

### 安装 llama-gguf-split

1. 克隆 llama.cpp 仓库:
```bash
git clone --recursive https://github.com/ggerganov/llama.cpp
```

2. 编译安装:
```bash
cd llama.cpp
cmake -S . -B build
cmake --build build --config Release
```

3. 将生成的可执行文件添加到系统 PATH:
- Windows: 将 `build/bin/Release` 目录添加到系统环境变量
- Linux/Mac: 将可执行文件链接到 `/usr/local/bin`

### 自动合并

程序会自动检测分片文件并进行合并:
1. 首次下载模型时，如果检测到分片文件
2. 自动调用 llama-gguf-split 进行合并
3. 合并后的文件将保存在相同目录下

合并完成后即可正常使用模型。

参考资料:
- [How to use the gguf-split / Model sharding demo](https://github.com/ggml-org/llama.cpp/discussions/6404)

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

