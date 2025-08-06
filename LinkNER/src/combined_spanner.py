import torch
import numpy as np
import os
import pickle
import json
from typing import Dict, List, Tuple, Optional
from models.bert_model_spanner import BertNER
from models.Evidential_woker import Span_Evidence
from src.combined_uncertainty import CombinedUncertaintyEstimator
import logging

logger = logging.getLogger(__name__)

class ModelResult:
    """存储单个模型的预测结果"""
    def __init__(self, model_name: str, f1_score: float):
        self.model_name = model_name
        self.f1_score = f1_score
        self.predictions = {}  # 存储span到label的映射
        self.probabilities = {}  # 存储span到概率分布的映射
        self.uncertainties = {}  # 存储span到不确定性的映射

class CombinedSpanNER:
    """具有组合器功能的SpanNER模块"""
    
    def __init__(self, args, num_labels):
        self.args = args
        self.num_labels = num_labels
        self.models = {}  # 存储加载的模型
        self.model_results = {}  # 存储模型结果
        self.combination_method = getattr(args, 'combination_method', 'voting_majority')
        self.use_f1_weights = getattr(args, 'use_f1_weights', True)
        self.use_prob_scores = getattr(args, 'use_prob_scores', True)
        self.f1_weight = getattr(args, 'f1_weight', 1.0)
        self.prob_weight = getattr(args, 'prob_weight', 0.8)
        
        # 初始化Evidence模块用于不确定估计
        self.evidential_worker = Span_Evidence(args, num_labels)
        
        # 初始化组合不确定性估计器
        uncertainty_methods = getattr(args, 'uncertainty_methods', ['entropy', 'variance', 'disagreement'])
        self.uncertainty_estimator = CombinedUncertaintyEstimator(uncertainty_methods)
        
    def load_model(self, model_path: str, model_name: str, f1_score: float = 0.0):
        """加载单个NER模型"""
        try:
            model = torch.load(model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
            model.eval()
            self.models[model_name] = model
            self.model_results[model_name] = ModelResult(model_name, f1_score)
            logger.info(f"成功加载模型 {model_name}, F1: {f1_score}")
        except Exception as e:
            logger.error(f"加载模型 {model_name} 失败: {e}")
            
    def load_models_from_config(self, models_config: List[Dict]):
        """从配置文件加载多个模型"""
        for config in models_config:
            model_path = config['path']
            model_name = config['name']
            f1_score = config.get('f1_score', 0.0)
            self.load_model(model_path, model_name, f1_score)
    
    def predict_single_model(self, model_name: str, data_loader):
        """使用单个模型进行预测"""
        if model_name not in self.models:
            raise ValueError(f"模型 {model_name} 未加载")
            
        model = self.models[model_name]
        model_result = self.model_results[model_name]
        
        span_predictions = {}
        span_probabilities = {}
        span_uncertainties = {}
        
        with torch.no_grad():
            for batch_idx, data in enumerate(data_loader):
                # 解包数据
                tokens, token_type_ids, all_span_idxs_ltoken, morph_idxs, span_label_ltoken, all_span_lens, all_span_weights, real_span_mask_ltoken, words, all_span_word, all_span_idxs = data
                loadall = [tokens, token_type_ids, all_span_idxs_ltoken, morph_idxs, span_label_ltoken, all_span_lens, all_span_weights, real_span_mask_ltoken, words, all_span_word, all_span_idxs]
                attention_mask = (tokens != 0).long()
                
                # 模型前向传播
                logits = model(loadall, all_span_lens, all_span_idxs_ltoken, tokens, attention_mask, token_type_ids)
                
                # 获取预测和不确定性
                predicts, uncertainty = self.evidential_worker.pred(logits)
                prob, pred_id = torch.max(predicts, 2)
                
                # 处理每个批次的span
                batch_size = tokens.size(0)
                for batch_i in range(batch_size):
                    for span_i, span_idx in enumerate(all_span_idxs[batch_i]):
                        if real_span_mask_ltoken[batch_i][span_i] == 0:
                            continue
                            
                        span_key = (batch_idx, batch_i, span_i, tuple(span_idx))
                        
                        # 获取预测标签
                        label_id = pred_id[batch_i][span_i].item()
                        if hasattr(self.args, 'label2idx_list') and self.args.label2idx_list:
                            label = self.args.label2idx_list[label_id][0] if label_id < len(self.args.label2idx_list) else 'O'
                        else:
                            label = f"LABEL_{label_id}"
                        
                        # 存储结果
                        span_predictions[span_key] = label
                        span_probabilities[span_key] = predicts[batch_i][span_i].cpu().numpy()
                        span_uncertainties[span_key] = uncertainty[batch_i][span_i].item()
        
        # 更新模型结果
        model_result.predictions = span_predictions
        model_result.probabilities = span_probabilities  
        model_result.uncertainties = span_uncertainties
        
        return span_predictions, span_probabilities, span_uncertainties
    
    def predict_all_models(self, data_loader):
        """使用所有加载的模型进行预测"""
        all_predictions = {}
        
        for model_name in self.models.keys():
            logger.info(f"使用模型 {model_name} 进行预测...")
            predictions, probabilities, uncertainties = self.predict_single_model(model_name, data_loader)
            all_predictions[model_name] = {
                'predictions': predictions,
                'probabilities': probabilities,
                'uncertainties': uncertainties
            }
            
        return all_predictions
    
    def voting_majority(self, span_key: Tuple) -> Tuple[str, float]:
        """多数投票组合方法"""
        votes = {}
        
        for model_name, model_result in self.model_results.items():
            if span_key in model_result.predictions:
                label = model_result.predictions[span_key]
                votes[label] = votes.get(label, 0) + 1
        
        if not votes:
            return 'O', 1.0
            
        # 选择得票最多的标签
        best_label = max(votes.keys(), key=lambda x: votes[x])
        confidence = votes[best_label] / len(self.model_results)
        
        return best_label, confidence
    
    def voting_weighted_by_f1(self, span_key: Tuple) -> Tuple[str, float]:
        """基于F1分数加权投票"""
        weighted_votes = {}
        
        for model_name, model_result in self.model_results.items():
            if span_key in model_result.predictions:
                label = model_result.predictions[span_key]
                weight = model_result.f1_score
                weighted_votes[label] = weighted_votes.get(label, 0.0) + weight
        
        if not weighted_votes:
            return 'O', 1.0
            
        # 选择加权得分最高的标签
        best_label = max(weighted_votes.keys(), key=lambda x: weighted_votes[x])
        total_weight = sum(weighted_votes.values())
        confidence = weighted_votes[best_label] / total_weight if total_weight > 0 else 0.0
        
        return best_label, confidence
    
    def voting_with_probabilities(self, span_key: Tuple) -> Tuple[str, float]:
        """基于概率分布的组合方法"""
        combined_probs = np.zeros(self.num_labels)
        total_weight = 0.0
        
        for model_name, model_result in self.model_results.items():
            if span_key in model_result.probabilities:
                probs = model_result.probabilities[span_key]
                f1_weight = model_result.f1_score if self.use_f1_weights else 1.0
                weight = self.f1_weight * f1_weight + self.prob_weight * np.max(probs)
                
                combined_probs += weight * probs
                total_weight += weight
        
        if total_weight == 0:
            return 'O', 1.0
            
        # 归一化
        combined_probs /= total_weight
        
        # 获取最佳标签
        best_label_id = np.argmax(combined_probs)
        confidence = combined_probs[best_label_id]
        
        if hasattr(self.args, 'label2idx_list') and self.args.label2idx_list:
            best_label = self.args.label2idx_list[best_label_id][0] if best_label_id < len(self.args.label2idx_list) else 'O'
        else:
            best_label = f"LABEL_{best_label_id}"
            
        return best_label, confidence
    
    def combine_predictions(self, all_span_keys: List[Tuple]) -> Tuple[Dict, Dict, Dict]:
        """组合所有模型的预测结果"""
        combined_predictions = {}
        combined_confidences = {}
        combined_uncertainties = {}
        
        for span_key in all_span_keys:
            # 根据组合方法选择组合策略
            if self.combination_method == 'voting_majority':
                label, confidence = self.voting_majority(span_key)
            elif self.combination_method == 'voting_weighted_f1':
                label, confidence = self.voting_weighted_by_f1(span_key)
            elif self.combination_method == 'voting_with_probabilities':
                label, confidence = self.voting_with_probabilities(span_key)
            else:
                # 默认使用多数投票
                label, confidence = self.voting_majority(span_key)
            
            # 收集各模型的预测和概率
            span_predictions = []
            span_probabilities = []
            
            for model_name, model_result in self.model_results.items():
                if span_key in model_result.predictions:
                    span_predictions.append(model_result.predictions[span_key])
                if span_key in model_result.probabilities:
                    span_probabilities.append(model_result.probabilities[span_key])
            
            # 使用改进的不确定性估计
            if span_predictions and span_probabilities:
                uncertainty_dict = self.uncertainty_estimator.compute_combined_uncertainty(
                    span_predictions, span_probabilities
                )
                combined_uncertainty = uncertainty_dict['combined']
            else:
                # 回退到原始方法
                uncertainties = []
                for model_name, model_result in self.model_results.items():
                    if span_key in model_result.uncertainties:
                        uncertainties.append(model_result.uncertainties[span_key])
                combined_uncertainty = np.mean(uncertainties) if uncertainties else 1.0
            
            combined_predictions[span_key] = label
            combined_confidences[span_key] = confidence
            combined_uncertainties[span_key] = combined_uncertainty
        
        return combined_predictions, combined_confidences, combined_uncertainties
    
    def forward(self, data_loader):
        """组合器的前向传播，返回组合后的预测结果"""
        # 使用所有模型进行预测
        all_predictions = self.predict_all_models(data_loader)
        
        # 收集所有span keys
        all_span_keys = set()
        for model_predictions in all_predictions.values():
            all_span_keys.update(model_predictions['predictions'].keys())
        
        # 组合预测结果
        combined_predictions, combined_confidences, combined_uncertainties = self.combine_predictions(list(all_span_keys))
        
        return {
            'predictions': combined_predictions,
            'confidences': combined_confidences,
            'uncertainties': combined_uncertainties,
            'individual_predictions': all_predictions
        }
    
    def get_uncertainty_for_linking(self, results: Dict, threshold: float = 0.4) -> List[Dict]:
        """为链接到LLM准备不确定性数据"""
        uncertain_spans = []
        
        for span_key, uncertainty in results['uncertainties'].items():
            if uncertainty > threshold:
                span_data = {
                    'span_key': span_key,
                    'uncertainty': uncertainty,
                    'prediction': results['predictions'].get(span_key, 'O'),
                    'confidence': results['confidences'].get(span_key, 0.0),
                    'individual_predictions': {}
                }
                
                # 添加各个模型的预测
                for model_name in self.models.keys():
                    if span_key in self.model_results[model_name].predictions:
                        span_data['individual_predictions'][model_name] = {
                            'prediction': self.model_results[model_name].predictions[span_key],
                            'uncertainty': self.model_results[model_name].uncertainties.get(span_key, 1.0)
                        }
                
                uncertain_spans.append(span_data)
        
        return uncertain_spans