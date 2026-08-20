#!/bin/bash
# 启动 vLLM OpenAI 兼容端点（Qwen2.5-7B-Instruct），供 llm/vllm_backend.py 调用。
#
# 用法（服务器上）：
#   bash scripts/start_vllm.sh                # 前台
#   nohup bash scripts/start_vllm.sh > /root/autodl-tmp/log_vllm.log 2>&1 &   # 后台
#
# 客户端配置（config.py 或环境变量）：
#   LLM_BACKEND=vllm  VLLM_URL=http://127.0.0.1:8000/v1
#
# 模型路径：优先 models_cache（旧项目同款目录），回退 HF 缓存。
set -e
export HF_ENDPOINT=https://hf-mirror.com
MODELS_DIR=/root/autodl-tmp/models_cache

QWEN=$(ls -d \
  "$MODELS_DIR/Qwen/Qwen2.5-7B-Instruct" \
  "$MODELS_DIR/models--Qwen--Qwen2.5-7B-Instruct/snapshots/"*/  \
  ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/*/ \
  2>/dev/null | head -1 || true)
if [ -z "$QWEN" ]; then
  echo "Qwen2.5-7B-Instruct not found. Run scripts/setup_server.sh first."
  exit 1
fi

PORT=${VLLM_PORT:-8000}
echo "serving $QWEN on :$PORT (model name: default)"

vllm serve "$QWEN" \
  --served-model-name default \
  --port "$PORT" \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.9 \
  --enforce-eager \
  "$@"

# 启动后可自检：
#   curl http://127.0.0.1:8000/v1/models
