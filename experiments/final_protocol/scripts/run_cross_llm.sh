#!/bin/bash
# Phase 7：跨 LLM 泛化（冻结 test / clean_val，同 split 同配置）
#   对 Llama-3.1-8B-Instruct 与 Mistral-7B-Instruct-v0.3：
#     test:    B0, G0(S0 τ=0.05 强保护点), G1(S2 λ=0.2 可行点)
#     clean_val: B0, G1
#   （risk learner / graph 全部复用 Qwen 主链——跨模型泛化考察生成侧）
set -u
cd /root/autodl-tmp/RAG/experiments/final_protocol
export HF_OFFLINE=1 LLM_BACKEND=hf
export HF_EMBEDDING_MODEL=/root/autodl-tmp/models_cache/BAAI/bge-base-en-v1.5
export HF_NLI_MODEL=/root/autodl-tmp/models_cache/cross-encoder/nli-deberta-v3-base
PY=/root/miniconda3/envs/rag/bin/python
PARAMS='{"G0":{"selector":"s0","tau":0.05},"G1":{"selector":"s2","lam":0.2}}'

for MODEL in "/root/autodl-tmp/models_cache/Llama/Meta-Llama-3.1-8B-Instruct:llama31" \
             "/root/autodl-tmp/models_cache/Mistral/Mistral-7B-Instruct-v0.3:mistral7"; do
  M="${MODEL%%:*}"; TAG="${MODEL##*:}"
  export HF_MODEL="$M"
  echo "[$TAG] test: B0,G0,G1"
  $PY -m fp.runner --stage eval --split test --methods B0,G0,G1 --run-dir results/p1 \
    --methods-params "$PARAMS" > "logs/p7_${TAG}_test.log" 2>&1
  echo "[$TAG] clean_val: B0,G1"
  $PY -m fp.runner --stage eval --split clean_val --methods B0,G1 --run-dir results/p1 \
    --methods-params "$PARAMS" > "logs/p7_${TAG}_clean.log" 2>&1
done
echo CROSS_LLM_DONE > results/p1/CROSS_LLM_DONE
