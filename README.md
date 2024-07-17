## Rust OpenChat Bot

参考 [candle](https://github.com/huggingface/candle)/example/quantized,
在[candle_demo_openchat_35](https://github.com/rustai-solutions/candle_demo_openchat_35)基础上进一步封装简化成机器人结构体

### 下载权重

1. 修改`download_model.py`, `arg::Args::model`中的openchat文件名
2. 运行`python download_model.py`

### 运行

```
cargo run --release
```

