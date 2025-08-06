#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LinkNER组合器功能测试脚本

此脚本用于测试组合器功能的基本功能是否正常工作。
"""

import os
import pickle
import tempfile
import shutil
from combination.comb_voting import CombByVoting
from combination.evaluate_metric import evaluate_chunk_level


def create_test_data():
    """创建测试数据"""
    # 模拟三个模型的预测结果
    
    # 模型1的结果
    pred_chunks_1 = [
        ('PER', 0, 2, 0),   # John Smith
        ('LOC', 5, 7, 0),   # New York
        ('ORG', 10, 12, 1), # Google Inc
    ]
    true_chunks_1 = [
        ('PER', 0, 2, 0),
        ('LOC', 5, 7, 0),
        ('ORG', 10, 12, 1),
        ('MISC', 15, 16, 1),  # 模型1漏检了这个
    ]
    
    # 模型2的结果
    pred_chunks_2 = [
        ('PER', 0, 2, 0),   # John Smith - 正确
        ('LOC', 5, 7, 0),   # New York - 正确
        ('MISC', 15, 16, 1), # 模型2检测到了这个
        ('PER', 18, 19, 1),  # 模型2的误检
    ]
    true_chunks_2 = true_chunks_1  # 真实标签相同
    
    # 模型3的结果
    pred_chunks_3 = [
        ('PER', 0, 2, 0),   # John Smith - 正确
        ('ORG', 10, 12, 1), # Google Inc - 正确
        ('MISC', 15, 16, 1), # 正确检测
        ('LOC', 20, 21, 1),  # 另一个位置的误检
    ]
    true_chunks_3 = true_chunks_1  # 真实标签相同
    
    return [
        (pred_chunks_1, true_chunks_1),
        (pred_chunks_2, true_chunks_2),
        (pred_chunks_3, true_chunks_3)
    ]


def save_test_models(test_data, temp_dir):
    """保存测试模型结果到临时文件"""
    model_files = []
    for i, (pred_chunks, true_chunks) in enumerate(test_data):
        file_path = os.path.join(temp_dir, f"test_model_{i+1}.pkl")
        with open(file_path, 'wb') as f:
            pickle.dump((pred_chunks, true_chunks), f)
        model_files.append(file_path)
    return model_files


def calculate_f1_scores(test_data):
    """计算每个模型的F1分数"""
    f1_scores = []
    for pred_chunks, true_chunks in test_data:
        f1, _, _, _, _, _ = evaluate_chunk_level(pred_chunks, true_chunks)
        f1_scores.append(f1)
    return f1_scores


def test_combination_methods(temp_dir, model_files, f1_scores):
    """测试所有组合方法"""
    print("Testing combination methods...")
    print("=" * 50)
    
    # 测试参数
    dataname = "conll03"
    classes = ['PER', 'LOC', 'ORG', 'MISC']
    cmodelname = "TestCombination"
    fn_stand_res = ""
    fn_prob = ""
    result_dir = os.path.join(temp_dir, "results")
    
    # 创建组合器
    combiner = CombByVoting(
        dataname=dataname,
        file_dir=temp_dir,
        fmodels=[os.path.basename(f) for f in model_files],
        f1s=f1_scores,
        cmodelname=cmodelname,
        classes=classes,
        fn_stand_res=fn_stand_res,
        fn_prob=fn_prob,
        result_dir=result_dir
    )
    
    # 测试所有组合方法
    methods = [
        'best_potential',
        'voting_majority',
        'voting_weightByOverallF1',
        'voting_weightByCategotyF1'
    ]
    
    results = {}
    for method in methods:
        print(f"\nTesting method: {method}")
        try:
            result = combiner.combine_results(method=method)
            f1, p, r, correct_preds, total_preds, total_correct = result
            results[method] = result
            print(f"  F1: {f1:.4f}, Precision: {p:.4f}, Recall: {r:.4f}")
            print(f"  Correct: {correct_preds}, Predicted: {total_preds}, True: {total_correct}")
        except Exception as e:
            print(f"  Error: {e}")
            results[method] = None
    
    return results


def test_individual_models(test_data):
    """测试单个模型的性能"""
    print("\nIndividual model performance:")
    print("-" * 30)
    
    for i, (pred_chunks, true_chunks) in enumerate(test_data):
        f1, p, r, correct_preds, total_preds, total_correct = evaluate_chunk_level(
            pred_chunks, true_chunks)
        print(f"Model {i+1}: F1={f1:.4f}, P={p:.4f}, R={r:.4f}")


def main():
    """主测试函数"""
    print("LinkNER Combination Functionality Test")
    print("=" * 50)
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    print(f"Using temporary directory: {temp_dir}")
    
    try:
        # 创建测试数据
        test_data = create_test_data()
        print(f"Created test data with {len(test_data)} models")
        
        # 测试单个模型性能
        test_individual_models(test_data)
        
        # 保存测试模型
        model_files = save_test_models(test_data, temp_dir)
        print(f"\nSaved test models: {[os.path.basename(f) for f in model_files]}")
        
        # 计算F1分数
        f1_scores = calculate_f1_scores(test_data)
        print(f"Individual F1 scores: {[f'{f:.4f}' for f in f1_scores]}")
        
        # 测试组合方法
        results = test_combination_methods(temp_dir, model_files, f1_scores)
        
        # 总结结果
        print("\n" + "=" * 50)
        print("SUMMARY")
        print("=" * 50)
        
        successful_methods = [method for method, result in results.items() if result is not None]
        failed_methods = [method for method, result in results.items() if result is None]
        
        print(f"Successful methods: {len(successful_methods)}")
        for method in successful_methods:
            f1 = results[method][0]
            print(f"  - {method}: F1={f1:.4f}")
        
        if failed_methods:
            print(f"\nFailed methods: {len(failed_methods)}")
            for method in failed_methods:
                print(f"  - {method}")
        
        # 检查是否有改进
        best_individual_f1 = max(f1_scores)
        best_combination_f1 = max([result[0] for result in results.values() if result is not None])
        
        print(f"\nBest individual model F1: {best_individual_f1:.4f}")
        print(f"Best combination F1: {best_combination_f1:.4f}")
        print(f"Improvement: {best_combination_f1 - best_individual_f1:.4f}")
        
        if best_combination_f1 > best_individual_f1:
            print("✅ Combination shows improvement!")
        else:
            print("⚠️  Combination did not improve over individual models")
        
        print("\n🎉 Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # 清理临时文件
        try:
            shutil.rmtree(temp_dir)
            print(f"Cleaned up temporary directory: {temp_dir}")
        except Exception as e:
            print(f"Warning: Could not clean up temporary directory: {e}")


if __name__ == "__main__":
    main()