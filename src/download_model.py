import os

# 设置镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from huggingface_hub import snapshot_download

# 下载第一个模型
# snapshot_download(repo_id="TheBloke/Llama-2-7B-GGML", allow_patterns=["llama-2-7b.ggmlv3.q4_0.bin"])
snapshot_download(repo_id="TheBloke/openchat_3.5-GGUF", allow_patterns=["openchat_3.5.Q2_K.gguf"])

# 下载第二个模型
# snapshot_download(repo_id="hf-internal-testing/llama-tokenizer")
snapshot_download(repo_id="openchat/openchat_3.5", allow_patterns=["tokenizer.json"])
