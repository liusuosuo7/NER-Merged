#!/bin/bash

# 使用组合器功能的LinkNER推理脚本
# 该脚本展示如何使用多个预训练的spanNER模型进行组合推理，然后链接到大语言模型

cd /workspace/LinkNER

# 设置数据和模型路径
DATA_DIR="dataset/conll03"
BERT_CONFIG_DIR="models/bert-base-cased"
MODELS_CONFIG="config/models_config.json"
RESULTS_DIR="results/"
LOG_DIR="log/"

echo "==================== 组合器模式推理 ===================="
echo "使用多个spanNER模型进行组合推理..."

# 第一步：使用组合器进行本地推理
python main.py \
    --state inference \
    --use_combination true \
    --models_config_path "$MODELS_CONFIG" \
    --combination_method "voting_with_probabilities" \
    --use_f1_weights true \
    --use_prob_scores true \
    --f1_weight 1.0 \
    --prob_weight 0.8 \
    --data_dir "$DATA_DIR" \
    --bert_config_dir "$BERT_CONFIG_DIR" \
    --results_dir "$RESULTS_DIR" \
    --logger_dir "$LOG_DIR" \
    --max_spanLen 5 \
    --n_class 5 \
    --test_mode ori \
    --threshold 0.4 \
    --dataname conll03 \
    --batch_size 32

echo "==================== 组合器推理完成 ===================="
echo "结果已保存到: ${RESULTS_DIR}conll03_combined_local_model.jsonl"
echo "不确定span已保存到: ${RESULTS_DIR}conll03_uncertain_spans.json"

echo ""
echo "==================== 准备链接到LLM ===================="
echo "基于不确定性阈值，准备需要LLM进一步处理的span..."

# 第二步：链接到LLM（需要根据实际情况配置LLM相关参数）
# python main.py \
#     --state llm_classify \
#     --input_file "${RESULTS_DIR}conll03_uncertain_spans.json" \
#     --save_file "${RESULTS_DIR}llm_enhanced_results.json" \
#     --linkDataName conll03 \
#     --llm_name 'your_llm_name' \
#     --llm_ckpt 'path_to_your_llm' \
#     --threshold 0.4 \
#     --shot 3

echo "脚本执行完成！"
echo ""
echo "组合器配置说明："
echo "- combination_method: voting_with_probabilities (基于概率分布的投票)"
echo "- use_f1_weights: true (使用F1分数权重)"
echo "- use_prob_scores: true (使用概率分数)"
echo "- f1_weight: 1.0 (F1权重系数)"
echo "- prob_weight: 0.8 (概率权重系数)"
echo "- threshold: 0.4 (不确定性阈值)"