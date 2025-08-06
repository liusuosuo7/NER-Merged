import torch
import numpy as np
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class CombinedUncertaintyEstimator:
    """组合器专用的不确定性估计器"""
    
    def __init__(self, uncertainty_methods: List[str] = None):
        """
        初始化不确定性估计器
        
        Args:
            uncertainty_methods: 不确定性估计方法列表
                - 'entropy': 基于熵的不确定性
                - 'variance': 模型间预测方差
                - 'disagreement': 模型分歧度
                - 'confidence': 基于置信度的不确定性
        """
        if uncertainty_methods is None:
            uncertainty_methods = ['entropy', 'variance', 'disagreement']
        self.uncertainty_methods = uncertainty_methods
    
    def entropy_uncertainty(self, probabilities: List[np.ndarray]) -> float:
        """
        计算基于熵的不确定性
        
        Args:
            probabilities: 各模型的概率分布列表
            
        Returns:
            熵值，越大表示不确定性越高
        """
        if not probabilities:
            return 1.0
            
        # 计算平均概率分布
        avg_probs = np.mean(probabilities, axis=0)
        
        # 避免log(0)
        avg_probs = np.clip(avg_probs, 1e-10, 1.0)
        
        # 计算熵
        entropy = -np.sum(avg_probs * np.log(avg_probs))
        
        # 归一化到[0,1]范围
        max_entropy = np.log(len(avg_probs))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        
        return normalized_entropy
    
    def variance_uncertainty(self, probabilities: List[np.ndarray]) -> float:
        """
        计算模型间预测方差作为不确定性
        
        Args:
            probabilities: 各模型的概率分布列表
            
        Returns:
            方差值，越大表示模型间分歧越大
        """
        if len(probabilities) < 2:
            return 0.0
            
        # 计算各类别概率的方差
        prob_matrix = np.array(probabilities)  # shape: (n_models, n_classes)
        variances = np.var(prob_matrix, axis=0)
        
        # 使用平均方差作为不确定性
        avg_variance = np.mean(variances)
        
        return avg_variance
    
    def disagreement_uncertainty(self, predictions: List[str]) -> float:
        """
        计算模型间预测分歧度
        
        Args:
            predictions: 各模型的预测标签列表
            
        Returns:
            分歧度，0表示完全一致，1表示完全分歧
        """
        if not predictions:
            return 1.0
            
        # 统计不同预测的数量
        unique_predictions = set(predictions)
        n_unique = len(unique_predictions)
        n_total = len(predictions)
        
        if n_total <= 1:
            return 0.0
            
        # 计算分歧度
        disagreement = (n_unique - 1) / (n_total - 1) if n_total > 1 else 0.0
        
        return disagreement
    
    def confidence_uncertainty(self, probabilities: List[np.ndarray]) -> float:
        """
        基于置信度的不确定性
        
        Args:
            probabilities: 各模型的概率分布列表
            
        Returns:
            不确定性值，越大表示置信度越低
        """
        if not probabilities:
            return 1.0
            
        # 计算各模型的最大概率（置信度）
        confidences = [np.max(prob) for prob in probabilities]
        
        # 使用平均置信度的倒数作为不确定性
        avg_confidence = np.mean(confidences)
        uncertainty = 1.0 - avg_confidence
        
        return uncertainty
    
    def mutual_information_uncertainty(self, probabilities: List[np.ndarray]) -> float:
        """
        计算互信息作为不确定性度量
        
        Args:
            probabilities: 各模型的概率分布列表
            
        Returns:
            互信息值
        """
        if len(probabilities) < 2:
            return 0.0
            
        # 计算总不确定性（熵）
        avg_probs = np.mean(probabilities, axis=0)
        avg_probs = np.clip(avg_probs, 1e-10, 1.0)
        total_uncertainty = -np.sum(avg_probs * np.log(avg_probs))
        
        # 计算期望的数据不确定性
        expected_uncertainty = 0.0
        for prob in probabilities:
            prob = np.clip(prob, 1e-10, 1.0)
            entropy = -np.sum(prob * np.log(prob))
            expected_uncertainty += entropy
        expected_uncertainty /= len(probabilities)
        
        # 互信息 = 总不确定性 - 期望数据不确定性
        mutual_info = total_uncertainty - expected_uncertainty
        
        return mutual_info
    
    def compute_combined_uncertainty(self, 
                                   predictions: List[str], 
                                   probabilities: List[np.ndarray],
                                   weights: List[float] = None) -> Dict[str, float]:
        """
        计算组合不确定性
        
        Args:
            predictions: 各模型的预测标签列表
            probabilities: 各模型的概率分布列表
            weights: 各不确定性方法的权重
            
        Returns:
            包含各种不确定性度量的字典
        """
        if weights is None:
            weights = [1.0] * len(self.uncertainty_methods)
        
        uncertainties = {}
        
        # 计算各种不确定性
        if 'entropy' in self.uncertainty_methods:
            uncertainties['entropy'] = self.entropy_uncertainty(probabilities)
            
        if 'variance' in self.uncertainty_methods:
            uncertainties['variance'] = self.variance_uncertainty(probabilities)
            
        if 'disagreement' in self.uncertainty_methods:
            uncertainties['disagreement'] = self.disagreement_uncertainty(predictions)
            
        if 'confidence' in self.uncertainty_methods:
            uncertainties['confidence'] = self.confidence_uncertainty(probabilities)
            
        if 'mutual_info' in self.uncertainty_methods:
            uncertainties['mutual_info'] = self.mutual_information_uncertainty(probabilities)
        
        # 计算加权平均不确定性
        method_weights = dict(zip(self.uncertainty_methods, weights))
        total_weight = sum(method_weights.values())
        
        combined_uncertainty = 0.0
        for method, uncertainty in uncertainties.items():
            if method in method_weights:
                weight = method_weights[method] / total_weight
                combined_uncertainty += weight * uncertainty
        
        uncertainties['combined'] = combined_uncertainty
        
        return uncertainties
    
    def should_link_to_llm(self, uncertainties: Dict[str, float], 
                          threshold: float = 0.4,
                          method: str = 'combined') -> bool:
        """
        判断是否应该链接到LLM
        
        Args:
            uncertainties: 不确定性度量字典
            threshold: 不确定性阈值
            method: 使用哪种不确定性度量进行判断
            
        Returns:
            True表示应该链接到LLM
        """
        if method not in uncertainties:
            logger.warning(f"不确定性方法 {method} 不存在，使用combined方法")
            method = 'combined'
            
        uncertainty = uncertainties.get(method, 1.0)
        return uncertainty > threshold
    
    def rank_uncertain_spans(self, span_uncertainties: Dict, 
                           top_k: int = None) -> List[Tuple]:
        """
        根据不确定性对span进行排序
        
        Args:
            span_uncertainties: span到不确定性的映射
            top_k: 返回前k个最不确定的span
            
        Returns:
            按不确定性降序排列的span列表
        """
        # 按不确定性排序
        sorted_spans = sorted(span_uncertainties.items(), 
                            key=lambda x: x[1].get('combined', 0.0), 
                            reverse=True)
        
        if top_k is not None:
            sorted_spans = sorted_spans[:top_k]
            
        return sorted_spans