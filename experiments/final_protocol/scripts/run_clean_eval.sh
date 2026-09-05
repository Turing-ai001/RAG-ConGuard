#!/bin/bash
# 干净效用评估（v1.2 signed 口径）：
#   clean_test 特征 → clean_test 上 B0/G0(tau=0.25)/G2(lam=0.2) → clean_val 上同三方法
set -u
cd /root/autodl-tmp/RAG/experiments/final_protocol
export HF_OFFLINE=1 LLM_BACKEND=hf
export HF_EMBEDDING_MODEL=/root/autodl-tmp/models_cache/BAAI/bge-base-en-v1.5
export HF_MODEL=/root/autodl-tmp/models_cache/Qwen/Qwen2.5-7B-Instruct
export HF_NLI_MODEL=/root/autodl-tmp/models_cache/cross-encoder/nli-deberta-v3-base
PY=/root/miniconda3/envs/rag/bin/python

echo "[clean] collecting clean_test features"
$PY -m fp.runner --stage features --split clean_test --run-dir results/p1 > logs/p1_clean_test_feat.log 2>&1
echo "[clean] eval clean_val (B0,G0,G2)"
$PY -m fp.runner --stage eval --split clean_val --methods B0,G0,G2 --run-dir results/p1 \
  --methods-params '{"G0": {"tau": 0.25}, "G2": {"lam": 0.2}}' > logs/p1_cleanval_eval.log 2>&1
echo "[clean] eval clean_test (B0,G0,G2)"
$PY -m fp.runner --stage eval --split clean_test --methods B0,G0,G2 --run-dir results/p1 \
  --methods-params '{"G0": {"tau": 0.25}, "G2": {"lam": 0.2}}' > logs/p1_cleantest_eval.log 2>&1
echo CLEAN_DONE > results/p1/CLEAN_DONE
