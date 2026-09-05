#!/bin/bash
# Phase 8 消融（冻结 test）：
#   R1 rule-risk 防御：G0 选择器不变（S0 τ=0.05），risk 换 RuleRisk（无训练）
#   R2 rule-risk @ 可行点：S2 λ=0.2 + RuleRisk
set -u
cd /root/autodl-tmp/RAG/experiments/final_protocol
export HF_OFFLINE=1 LLM_BACKEND=hf
export HF_EMBEDDING_MODEL=/root/autodl-tmp/models_cache/BAAI/bge-base-en-v1.5
export HF_MODEL=/root/autodl-tmp/models_cache/Qwen/Qwen2.5-7B-Instruct
export HF_NLI_MODEL=/root/autodl-tmp/models_cache/cross-encoder/nli-deberta-v3-base
PY=/root/miniconda3/envs/rag/bin/python
PARAMS='{"G0":{"selector":"s0","tau":0.05},"G1":{"selector":"s2","lam":0.2}}'

$PY -m fp.runner --stage eval --split test --methods G0,G1 --run-dir results/p1 \
  --methods-params "$PARAMS" --risk-kind rule > logs/p8_rule_ablation.log 2>&1
echo ABLATION_DONE > results/p1/ABLATION_DONE
