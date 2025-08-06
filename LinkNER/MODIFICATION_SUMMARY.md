# LinkNER 组合器功能修改总结

## 修改概述

根据用户需求，本次修改成功将LinkNER从使用单一spanNER模型改为使用具有组合器功能的spanNER系统。新系统可以组合多个不同的命名实体识别模型，通过不确定估计方法生成更可靠的预测结果，然后连接到大型语言模型进行进一步处理。

## 主要修改内容

### 1. 核心组合器模块 (`src/combined_spanner.py`)

**新增类：`CombinedSpanNER`**
- 支持加载和管理多个预训练的spanNER模型
- 实现三种组合策略：
  - `voting_majority`: 多数投票
  - `voting_weighted_f1`: F1分数加权投票  
  - `voting_with_probabilities`: 概率分布组合（推荐）
- 生成组合后的预测结果、置信度和不确定性

**主要功能：**
- `load_model()`: 加载单个模型
- `load_models_from_config()`: 从配置文件批量加载模型
- `predict_all_models()`: 使用所有模型进行预测
- `combine_predictions()`: 组合多模型预测结果
- `get_uncertainty_for_linking()`: 为LLM链接准备不确定span

### 2. 组合框架 (`src/combined_framework.py`)

**新增类：`CombinedFewShotNERFramework`**
- 继承并扩展原有框架功能
- 集成组合器到完整的NER工作流程
- 兼容原有的单模型训练和推理模式
- 自动识别和保存高不确定性span用于LLM处理

**核心方法：**
- `metric_combined()`: 使用组合器进行评估
- `eval()`: 根据模式选择评估策略
- `train()`: 组合器模式下跳过训练（使用预训练模型）

### 3. 改进的不确定性估计 (`src/combined_uncertainty.py`)

**新增类：`CombinedUncertaintyEstimator`**
- 专门针对多模型组合的不确定性估计
- 支持多种不确定性度量方法：
  - `entropy`: 基于熵的不确定性
  - `variance`: 模型间预测方差
  - `disagreement`: 模型分歧度
  - `confidence`: 基于置信度的不确定性
  - `mutual_info`: 互信息

**主要功能：**
- `compute_combined_uncertainty()`: 计算综合不确定性
- `should_link_to_llm()`: 判断是否需要LLM介入
- `rank_uncertain_spans()`: 按不确定性排序span

### 4. 参数配置扩展 (`args_config.py`)

**新增参数：**
```python
--use_combination         # 启用组合器模式
--models_config_path      # 模型配置文件路径
--combination_method      # 组合方法选择
--use_f1_weights         # 是否使用F1权重
--use_prob_scores        # 是否使用概率分数
--f1_weight              # F1权重系数
--prob_weight            # 概率权重系数
```

### 5. 主程序修改 (`main.py`)

**修改要点：**
- 导入新的组合框架
- 根据`use_combination`参数选择使用原框架或组合框架
- 保持向后兼容性

### 6. 配置和脚本文件

**新增文件：**
- `config/models_config.json`: 模型配置示例
- `scripts/combined_inference.sh`: 组合器推理脚本
- `COMBINED_USAGE.md`: 详细使用说明

## 工作流程对比

### 原始工作流程：
```
单一spanNER → 不确定估计 → LLM链接
```

### 修改后工作流程：
```
多个spanNER模型 → 组合器 → 改进的不确定估计 → LLM链接
    ↓
[模型1, 模型2, 模型3] → 投票/加权组合 → 多维度不确定性 → 智能筛选
```

## 技术特点

### 1. 模块化设计
- 各组件独立可复用
- 易于扩展新的组合策略
- 保持原有接口兼容性

### 2. 灵活的配置
- 支持不同数量的模型组合
- 可调节的组合权重
- 多种不确定性计算方法

### 3. 性能优化
- 合理的批处理策略
- GPU内存管理
- 并行预测支持

### 4. 鲁棒性
- 模型加载失败处理
- 不确定性计算回退机制
- 详细的日志记录

## 使用示例

### 1. 基本使用
```bash
python main.py \
    --state inference \
    --use_combination true \
    --models_config_path config/models_config.json \
    --combination_method voting_with_probabilities
```

### 2. 模型配置
```json
[
  {
    "name": "spanNER_model1",
    "path": "models/model1.pkl", 
    "f1_score": 0.92
  }
]
```

## 预期效果

### 1. 性能提升
- 更高的F1分数（通过模型互补）
- 更可靠的不确定性估计
- 减少LLM调用次数（更精准的筛选）

### 2. 鲁棒性增强
- 降低单模型错误影响
- 提高边界case处理能力
- 更稳定的预测结果

### 3. 可扩展性
- 易于添加新模型
- 支持不同架构的模型组合
- 灵活的组合策略

## 注意事项

### 1. 资源需求
- 需要更多GPU内存来加载多个模型
- 推理时间会相应增加
- 需要准备多个预训练模型

### 2. 配置要求
- 确保模型文件路径正确
- 各模型需要兼容当前代码版本
- 合理设置不确定性阈值

### 3. 性能调优
- 根据具体任务调整组合方法
- 优化权重参数设置
- 监控内存和计算资源使用

## 总结

本次修改成功实现了用户的需求，将LinkNER从单模型架构升级为多模型组合架构。新系统在保持原有功能的基础上，提供了更强大的模型组合能力和更精确的不确定性估计，为后续的LLM链接奠定了更好的基础。整个修改具有良好的模块化设计、向后兼容性和可扩展性。