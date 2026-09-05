#!/bin/bash
# Phase 1 执行链（在 9 个特征文件齐全后自动接续）：
#   1) clean_val 特征  2) 风险学习器训练（train+clean_trn 负例池）  3) test 上 B0+G0 评估
set -u
cd /root/autodl-tmp/RAG/experiments/final_protocol
export HF_OFFLINE=1 LLM_BACKEND=hf
export HF_EMBEDDING_MODEL=/root/autodl-tmp/models_cache/BAAI/bge-base-en-v1.5
export HF_MODEL=/root/autodl-tmp/models_cache/Qwen/Qwen2.5-7B-Instruct
export HF_NLI_MODEL=/root/autodl-tmp/models_cache/cross-encoder/nli-deberta-v3-base
PY=/root/miniconda3/envs/rag/bin/python

echo "[chain] waiting for 9 feature files ..."
while [ "$(ls results/p1/features_*.jsonl 2>/dev/null | wc -l)" -lt 9 ]; do sleep 60; done
# 等采集进程退出（防止通道冲突）
while ps aux | grep -v grep | grep -q 'fp.runner'; do sleep 60; done
echo "[chain] starting clean_val features"
$PY -m fp.runner --stage features --split clean_val --run-dir results/p1 > logs/p1_clean_val.log 2>&1
echo "[chain] training risk learner"
$PY -m fp.runner --stage train --run-dir results/p1 > logs/p1_train.log 2>&1
echo "[chain] eval test B0+G0"
$PY -m fp.runner --stage eval --split test --methods B0,G0 --run-dir results/p1 > logs/p1_eval.log 2>&1
echo "[chain] P1 DONE" > results/p1/P1_DONE
