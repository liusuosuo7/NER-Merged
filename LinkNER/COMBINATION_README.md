# LinkNER 模型组合器功能

## 概述

LinkNER现在支持模型组合器功能，可以将多个命名实体识别模型的预测结果进行组合，以提高整体的识别性能。该功能基于spanNER的组合器实现，并针对LinkNER的不确定性估计机制进行了优化。

## 主要特性

1. **多种组合方法**: 支持多种投票和加权策略
2. **灵活配置**: 可通过命令行参数灵活配置组合参数
3. **不确定性感知**: 结合LinkNER的不确定性估计进行智能组合
4. **结果保存**: 自动保存组合结果用于后续分析

## 支持的组合方法

### 1. 最优潜在结果 (best_potential)
使用真实标签信息来确定理论上的最优组合结果，主要用于上界分析。

### 2. 多数投票 (voting_majority)
每个模型对同一位置的实体类型进行投票，选择得票最多的类型作为最终结果。

### 3. 基于整体F1的加权投票 (voting_weightByOverallF1)
根据每个模型的整体F1分数进行加权投票，性能更好的模型权重更高。

### 4. 基于类别F1的加权投票 (voting_weightByCategotyF1)
根据每个模型在特定类别上的F1分数进行加权投票，针对不同类别使用不同权重。

### 5. 基于概率的投票 (voting_prob)
结合模型预测概率和不确定性信息进行投票，特别适用于LinkNER场景。

## 使用方法

### 1. 训练时启用组合器功能

```bash
python main.py \
    --state train \
    --enable_combination True \
    --dataname conll03 \
    --combination_method voting_majority
```

### 2. 推理时使用组合器

```bash
python main.py \
    --state inference \
    --inference_model path/to/model.pkl \
    --enable_combination True \
    --model_files model1.pkl model2.pkl model3.pkl \
    --model_f1s 0.85 0.88 0.82 \
    --combination_method voting_weightByOverallF1
```

### 3. 仅运行模型组合

```bash
python main.py \
    --state combination \
    --enable_combination True \
    --model_files model1.pkl model2.pkl model3.pkl \
    --model_f1s 0.85 0.88 0.82 \
    --combination_method voting_weightByOverallF1 \
    --dataname conll03
```

### 4. 使用示例脚本

```bash
# 基本多数投票
python run_combination_example.py --model_files model1.pkl model2.pkl model3.pkl

# 加权投票
python run_combination_example.py \
    --model_files model1.pkl model2.pkl model3.pkl \
    --model_f1s 0.85 0.88 0.82 \
    --combination_method voting_weightByOverallF1

# 基于概率的投票
python run_combination_example.py \
    --model_files model1.pkl model2.pkl model3.pkl \
    --combination_method voting_prob \
    --prob_file prob_data.pkl
```

## 配置参数说明

### 组合器相关参数

- `--enable_combination`: 是否启用组合器功能 (默认: False)
- `--combination_method`: 组合方法选择 (默认: voting_majority)
- `--model_files`: 要组合的模型结果文件列表
- `--model_f1s`: 各模型的F1分数（用于加权投票）
- `--combination_result_dir`: 组合结果保存目录 (默认: combination/comb_result)
- `--prob_file`: 概率信息文件（用于概率投票）

## 文件格式要求

### 模型结果文件格式

模型结果文件应该是包含预测结果的pickle文件，支持以下格式：

1. **标准格式**: `(pred_chunks, true_chunks)`
2. **LinkNER格式**: `(pred_chunks, true_chunks, uncertainties)`
3. **字典格式**: `{'pred_chunks': [...], 'true_chunks': [...], 'uncertainties': [...]}`

### Chunk格式

每个chunk应该是一个元组：`(label, start_idx, end_idx, sent_id)`

- `label`: 实体类型标签
- `start_idx`: 实体起始位置
- `end_idx`: 实体结束位置
- `sent_id`: 句子ID

## 输出结果

组合器会输出以下信息：

1. **性能指标**:
   - F1分数
   - 精确率 (Precision)
   - 召回率 (Recall)
   - 正确预测数量
   - 总预测数量
   - 总正确数量

2. **结果文件**:
   - 组合后的预测结果保存为pickle文件
   - 文件名格式: `{方法名}_combine_{F1分数}.pkl`

## 实际使用示例

### 场景1: 组合多个spanNER模型

假设你有三个训练好的spanNER模型，可以这样组合：

```bash
python main.py \
    --state combination \
    --enable_combination True \
    --model_files results/model_bert_1.pkl results/model_bert_2.pkl results/model_roberta.pkl \
    --model_f1s 0.87 0.85 0.89 \
    --combination_method voting_weightByOverallF1 \
    --dataname conll03 \
    --combination_result_dir combination_results
```

### 场景2: LinkNER + spanNER模型组合

```bash
python main.py \
    --state combination \
    --enable_combination True \
    --model_files linkner_results.pkl spanner_model1.pkl spanner_model2.pkl \
    --combination_method voting_prob \
    --prob_file linkner_probabilities.pkl \
    --dataname conll03
```

## 注意事项

1. **文件路径**: 确保所有模型文件路径正确且文件存在
2. **数据格式**: 模型结果文件必须包含预测的chunk信息
3. **类别一致性**: 所有模型应该使用相同的标签体系
4. **F1分数**: 如果使用加权投票，建议提供准确的F1分数以获得更好的组合效果

## 扩展功能

### 自定义组合方法

可以在`combination/comb_voting.py`中添加新的组合方法：

```python
def custom_voting_method(self):
    """自定义投票方法"""
    pchunk_plb_ms, keep_pref_upchunks = self.get_unique_pchunk_labs()
    
    # 实现你的组合逻辑
    comb_kchunks = []
    
    # ... 你的代码 ...
    
    f1, p, r, correct_preds, total_preds, total_correct = evaluate_chunk_level(
        comb_kchunks, self.tchunks_unique)
    
    return [f1, p, r, correct_preds, total_preds, total_correct]
```

然后在`combine_results`方法中添加映射。

## 疑难解答

### 常见错误

1. **文件不存在**: 检查模型文件路径是否正确
2. **格式错误**: 确保模型结果文件格式符合要求
3. **内存不足**: 大量模型组合时可能需要更多内存
4. **标签不匹配**: 确保所有模型使用相同的标签体系

### 性能优化

1. **并行处理**: 大量模型时考虑并行处理
2. **内存管理**: 适时释放不需要的数据
3. **结果缓存**: 避免重复计算相同的组合

## 贡献指南

欢迎提交改进建议和新功能！请确保：

1. 代码风格与现有代码一致
2. 添加适当的注释和文档
3. 提供使用示例
4. 经过充分测试

## 许可证

此功能遵循项目的原始许可证。