#!/bin/bash
# AutoDL 服务器环境初始化（在服务器上执行，先同步项目再跑本脚本）。
#
# 用法：
#   bash scripts/setup_server.sh [--skip-model-download]
#
# 约定（与旧项目 RAG-ConGuard 保持一致，方便复用旧机器上的模型缓存）：
#   conda 环境   : rag_env
#   模型缓存     : /root/autodl-tmp/models_cache（BGE + Qwen2.5-7B-Instruct）
#   HF_ENDPOINT  : https://hf-mirror.com（国内服务器必需）
set -e

PY=/root/miniconda3/envs/rag_env/bin/python
export HF_ENDPOINT=https://hf-mirror.com
MODELS_DIR=/root/autodl-tmp/models_cache

echo "=== [1/4] conda 环境 ==="
if [ ! -x "$PY" ]; then
  conda create -y -n rag_env python=3.11
else
  echo "env rag_env already exists"
fi

echo "=== [2/4] pip 依赖（GPU 版：torch cu121 / faiss-gpu / vllm） ==="
"$PY" -m pip install --upgrade pip
"$PY" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
"$PY" -m pip install faiss-gpu rank-bm25 sentence-transformers transformers datasets openai numpy scikit-learn
"$PY" -m pip install vllm

echo "=== [3/4] 生成模型 Qwen2.5-7B-Instruct（vLLM 服务用） ==="
if [ "$1" == "--skip-model-download" ]; then
  echo "skip model download"
elif [ -d "$MODELS_DIR/Qwen/Qwen2.5-7B-Instruct" ] || [ -d "$MODELS_DIR/models--Qwen--Qwen2.5-7B-Instruct" ]; then
  echo "Qwen2.5-7B-Instruct already in $MODELS_DIR"
else
  "$PY" -m huggingface_hub snapshot_download \
    --repo-type model Qwen/Qwen2.5-7B-Instruct \
    --local-dir "$MODELS_DIR/Qwen/Qwen2.5-7B-Instruct"
fi

echo "=== [4/4] embedding 模型 BGE-base-en-v1.5（检索用） ==="
if [ -d "$MODELS_DIR/BAAI/bge-base-en-v1.5" ] || [ -d "$MODELS_DIR/models--BAAI--bge-base-en-v1.5" ]; then
  echo "bge-base-en-v1.5 already in $MODELS_DIR"
else
  "$PY" -m huggingface_hub snapshot_download \
    --repo-type model BAAI/bge-base-en-v1.5 \
    --local-dir "$MODELS_DIR/BAAI/bge-base-en-v1.5"
fi

echo "=== 完成。后续步骤："
echo "  1) 构建语料:    $PY scripts/download_datasets.py --corpus 全量 --queries 全量"
echo "  2) 构建 KB:      $PY scripts/build_kb.py"
echo "  3) 起 vLLM:      bash scripts/start_vllm.sh"
echo "  4) 跑评估:       LLM_BACKEND=vllm $PY scripts/evaluate_pipeline.py --gen 1"
