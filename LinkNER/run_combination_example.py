#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LinkNER模型组合示例脚本

此脚本演示如何使用LinkNER的组合器功能来组合多个模型的预测结果。
"""

import os
import sys
import argparse
from main import run_combination_only
from args_config import get_args


def create_example_args():
    """创建示例参数配置"""
    parser = argparse.ArgumentParser(description="LinkNER Model Combination Example")
    
    # 基本参数
    parser.add_argument("--dataname", default="conll03", type=str)
    parser.add_argument("--state", default="combination", type=str)
    parser.add_argument("--enable_combination", default=True, type=bool)
    
    # 组合器参数
    parser.add_argument("--combination_method", default="voting_majority", type=str,
                       choices=["best_potential", "voting_majority", "voting_weightByOverallF1", 
                               "voting_weightByCategotyF1", "voting_prob"])
    parser.add_argument("--model_files", nargs='+', default=[], 
                       help="List of model result files to combine")
    parser.add_argument("--model_f1s", type=float, nargs='+', default=[],
                       help="F1 scores of individual models")
    parser.add_argument("--combination_result_dir", default="combination/comb_result", type=str)
    parser.add_argument("--prob_file", default="", type=str)
    
    return parser.parse_args()


def main():
    print("LinkNER Model Combination Example")
    print("=" * 50)
    
    # 检查是否提供了模型文件
    if len(sys.argv) < 2:
        print("Usage examples:")
        print()
        print("1. Basic majority voting:")
        print(f"python {sys.argv[0]} --model_files model1.pkl model2.pkl model3.pkl")
        print()
        print("2. Weighted voting by overall F1:")
        print(f"python {sys.argv[0]} --model_files model1.pkl model2.pkl model3.pkl \\")
        print("    --model_f1s 0.85 0.88 0.82 --combination_method voting_weightByOverallF1")
        print()
        print("3. Probability-based voting:")
        print(f"python {sys.argv[0]} --model_files model1.pkl model2.pkl model3.pkl \\")
        print("    --combination_method voting_prob --prob_file prob_data.pkl")
        print()
        print("Available combination methods:")
        print("  - best_potential: 使用最优潜在结果")
        print("  - voting_majority: 多数投票")
        print("  - voting_weightByOverallF1: 基于整体F1的加权投票")
        print("  - voting_weightByCategotyF1: 基于类别F1的加权投票")
        print("  - voting_prob: 基于概率的投票")
        print()
        return
    
    # 解析参数
    args = create_example_args()
    
    if not args.model_files:
        print("Error: No model files specified!")
        print("Use --model_files to specify model result files.")
        return
    
    # 检查文件是否存在
    missing_files = []
    for file_path in args.model_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"Error: The following model files do not exist:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        print()
        print("请确保模型结果文件存在。")
        print("模型结果文件应该是通过LinkNER训练或推理生成的.pkl文件。")
        return
    
    print(f"Configuration:")
    print(f"  Dataset: {args.dataname}")
    print(f"  Combination method: {args.combination_method}")
    print(f"  Model files: {args.model_files}")
    print(f"  Model F1s: {args.model_f1s if args.model_f1s else 'Using default (1.0 for each)'}")
    print(f"  Result directory: {args.combination_result_dir}")
    if args.prob_file:
        print(f"  Probability file: {args.prob_file}")
    print()
    
    # 运行组合器
    try:
        run_combination_only(args)
        print()
        print("Model combination completed successfully! 🎉")
        print(f"Results saved to: {args.combination_result_dir}")
    except Exception as e:
        print(f"Error during combination: {e}")
        print("\n可能的解决方案:")
        print("1. 检查模型文件格式是否正确")
        print("2. 确保模型文件包含预测结果")
        print("3. 检查文件路径是否正确")
        print("4. 确保所有依赖包已安装")


if __name__ == "__main__":
    main()