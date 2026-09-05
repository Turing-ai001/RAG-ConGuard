#!/bin/bash
# signed_semantic 变体：clean_val 特征 → tune（与 p1 同网格）
set -u
cd /root/autodl-tmp/RAG/experiments/final_protocol
export HF_OFFLINE=1 LLM_BACKEND=hf
export HF_EMBEDDING_MODEL=/root/autodl-tmp/models_cache/BAAI/bge-base-en-v1.5
export HF_MODEL=/root/autodl-tmp/models_cache/Qwen/Qwen2.5-7B-Instruct
export HF_NLI_MODEL=/root/autodl-tmp/models_cache/cross-encoder/nli-deberta-v3-base
PY=/root/miniconda3/envs/rag/bin/python

echo "[semsem] clean_val features"
$PY -m fp.runner --stage features --split clean_val --run-dir results/p1semsem --graph-mode signed_semantic > logs/p1semsem_cleanval.log 2>&1
echo "[semsem] tune"
$PY -m fp.tune --run-dir results/p1semsem > logs/p1semsem_tune.log 2>&1
echo TUNE_DONE > results/p1semsem/TUNE_DONE
