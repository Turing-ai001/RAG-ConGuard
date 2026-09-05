#!/bin/bash
# Phase 5 主结果：冻结 test split，8 方法同池，val 选定的 operating points
set -u
cd /root/autodl-tmp/RAG/experiments/final_protocol
export HF_OFFLINE=1 LLM_BACKEND=hf
export HF_EMBEDDING_MODEL=/root/autodl-tmp/models_cache/BAAI/bge-base-en-v1.5
export HF_MODEL=/root/autodl-tmp/models_cache/Qwen/Qwen2.5-7B-Instruct
export HF_NLI_MODEL=/root/autodl-tmp/models_cache/cross-encoder/nli-deberta-v3-base
PY=/root/miniconda3/envs/rag/bin/python

$PY -m fp.runner --stage eval --split test \
  --methods B0,B1,B2,B4,B5,B6,G0,G1 \
  --run-dir results/p1 \
  --methods-params '{"B1":{"tau_rel":0.30},"B2":{"tau_dup":0.7},"B6":{"tau_loo":0.5},"G0":{"selector":"s0","tau":0.05},"G1":{"selector":"s2","lam":0.2}}' \
  > logs/p5_main_results.log 2>&1
echo MAIN_DONE > results/p1/MAIN_DONE
