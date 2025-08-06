# LinkNER 组合器功能使用说明

## 概述

该修改版本的LinkNER支持使用多个预训练的spanNER模型进行组合，而不是只使用单一的spanNER作为本地模型。组合器通过不确定估计方法将多个模型的预测结果融合，然后连接到大型语言模型进行进一步的命名实体识别。

## 新增功能

### 1. 组合器SpanNER (CombinedSpanNER)
- 支持加载和管理多个预训练的spanNER模型
- 提供多种组合策略：多数投票、F1加权投票、概率分布组合
- 输出组合后的预测结果和不确定性估计

### 2. 组合框架 (CombinedFewShotNERFramework)  
- 集成组合器功能到LinkNER工作流程
- 兼容原有的单模型模式
- 支持不确定span的自动识别和保存

### 3. 新增参数配置
- `--use_combination`: 是否启用组合器模式
- `--models_config_path`: 模型配置文件路径
- `--combination_method`: 组合方法选择
- `--f1_weight`, `--prob_weight`: 组合权重配置

## 使用方法

### 步骤1: 准备模型配置文件

创建一个JSON配置文件来指定要组合的模型：

```json
[
  {
    "name": "spanNER_model1",
    "path": "models/spanNER_model1.pkl",
    "f1_score": 0.92,
    "description": "SpanNER模型1 - BERT-base训练"
  },
  {
    "name": "spanNER_model2", 
    "path": "models/spanNER_model2.pkl",
    "f1_score": 0.90,
    "description": "SpanNER模型2 - RoBERTa-base训练"
  }
]
```

### 步骤2: 使用组合器进行推理

```bash
python main.py \
    --state inference \
    --use_combination true \
    --models_config_path "config/models_config.json" \
    --combination_method "voting_with_probabilities" \
    --use_f1_weights true \
    --use_prob_scores true \
    --f1_weight 1.0 \
    --prob_weight 0.8 \
    --data_dir "dataset/conll03" \
    --threshold 0.4 \
    --dataname conll03
```

### 步骤3: 链接到LLM（可选）

对于高不确定性的span，可以进一步链接到大语言模型：

```bash
python main.py \
    --state llm_classify \
    --input_file "results/conll03_uncertain_spans.json" \
    --save_file "results/llm_enhanced_results.json" \
    --llm_name 'your_llm_name' \
    --threshold 0.4
```

## 组合方法详解

### 1. 多数投票 (voting_majority)
- 选择获得最多模型支持的标签
- 适用于模型性能相近的情况

### 2. F1加权投票 (voting_weighted_f1)  
- 根据各模型的F1分数进行加权投票
- 性能更好的模型具有更大的权重

### 3. 概率分布组合 (voting_with_probabilities)
- 基于概率分布进行加权组合
- 同时考虑F1分数和预测置信度
- **推荐使用**，通常获得最佳性能

## 参数说明

### 组合器相关参数
- `--use_combination`: 启用组合器模式 (default: false)
- `--models_config_path`: 模型配置文件路径 (default: "config/models_config.json")
- `--combination_method`: 组合方法 (choices: voting_majority, voting_weighted_f1, voting_with_probabilities)
- `--use_f1_weights`: 是否使用F1分数作为权重 (default: true)
- `--use_prob_scores`: 是否使用概率分数 (default: true)
- `--f1_weight`: F1分数权重系数 (default: 1.0)
- `--prob_weight`: 概率分数权重系数 (default: 0.8)

### 原有参数保持不变
- `--threshold`: 不确定性阈值 (default: 0.4)
- `--dataname`: 数据集名称
- `--data_dir`: 数据目录
- 等等...

## 输出文件

### 1. 组合结果文件
- 文件名: `{dataname}_combined_local_model.jsonl`
- 内容: 每行包含一个句子的NER标注结果

### 2. 不确定span文件
- 文件名: `{dataname}_uncertain_spans.json`  
- 内容: 超过不确定性阈值的span信息，用于LLM进一步处理

## 性能优化建议

### 1. 模型选择
- 选择具有互补性的模型（不同架构、不同训练数据）
- 确保各模型都有合理的性能（F1 > 0.8）
- 通常3-5个模型的组合效果最佳

### 2. 参数调优
- `threshold`: 根据下游任务调整，平衡精度和召回率
- `f1_weight` vs `prob_weight`: 根据模型置信度可靠性调整
- `combination_method`: 建议先尝试`voting_with_probabilities`

### 3. 计算资源
- 组合器模式需要加载多个模型，确保有足够的GPU内存
- 推理时间会相应增加，可以考虑并行优化

## 故障排除

### 常见问题

1. **模型加载失败**
   - 检查模型文件路径是否正确
   - 确保模型文件与当前代码版本兼容

2. **内存不足**
   - 减少batch_size
   - 减少同时加载的模型数量

3. **组合效果不佳**
   - 检查各模型的单独性能
   - 尝试不同的组合方法
   - 调整权重参数

## 示例脚本

详细的使用示例请参考：
- `scripts/combined_inference.sh`: 组合器推理脚本
- `config/models_config.json`: 模型配置示例

## 扩展功能

该组合器设计具有良好的扩展性，可以方便地：
- 添加新的组合策略
- 支持不同类型的NER模型
- 集成更复杂的不确定性估计方法

如需自定义组合策略，可以在`CombinedSpanNER`类中添加新的组合方法。