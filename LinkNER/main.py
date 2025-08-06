# encoding: utf-8
import argparse
import os
import random
import logging
import torch
import json
from src.framework import FewShotNERFramework
from dataloaders.spanner_dataset import get_span_labels, get_loader
from src.bert_model_spanner import BertNER
from transformers import AutoTokenizer, AutoModel, AutoConfig
from src.config_spanner import BertNerConfig
from src.Evidential_woker import Span_Evidence
from metrics.mtrics_LinkResult import *
from args_config import get_args
from run_llm import *
from combination.comb_voting import CombByVoting

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def setup_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_logger(args, seed):
    path = os.path.join(args.logger_dir, f"{args.etrans_func}{seed}_{time.strftime('%m-%d_%H-%M-%S')}.txt")
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s", datefmt='%Y-%m-%d %H:%M:%S')

    file_handler = logging.FileHandler(path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger

def run_combination_only(args):
    """仅运行模型组合功能"""
    if not args.enable_combination or not args.model_files:
        print("Error: Combination mode requires enable_combination=True and model_files to be specified")
        return
    
    print(f"Running combination with method: {args.combination_method}")
    print(f"Model files: {args.model_files}")
    print(f"Model F1s: {args.model_f1s}")
    
    # 获取数据集类别（简化版本，实际应该从数据中推断）
    if args.dataname == 'conll03':
        classes = ['PER', 'LOC', 'ORG', 'MISC']
    elif args.dataname == 'wnut17':
        classes = ['person', 'location', 'group', 'creative-work', 'product']
    else:
        classes = ['PER', 'LOC', 'ORG']  # 默认类别
    
    # 使用默认F1分数如果没有提供
    model_f1s = args.model_f1s if args.model_f1s else [1.0] * len(args.model_files)
    
    # 创建组合器
    combiner = CombByVoting(
        dataname=args.dataname,
        file_dir=os.path.dirname(args.model_files[0]) if args.model_files else ".",
        fmodels=[os.path.basename(f) for f in args.model_files],
        f1s=model_f1s,
        cmodelname=f"LinkNER_Combined_{args.combination_method}",
        classes=classes,
        fn_stand_res="",
        fn_prob=args.prob_file if hasattr(args, 'prob_file') else "",
        result_dir=args.combination_result_dir
    )
    
    # 执行组合
    results = combiner.combine_results(method=args.combination_method)
    f1, p, r, correct_preds, total_preds, total_correct = results
    
    print(f"Combination Results:")
    print(f"F1: {f1:.4f}")
    print(f"Precision: {p:.4f}")
    print(f"Recall: {r:.4f}")
    print(f"Correct predictions: {correct_preds}")
    print(f"Total predictions: {total_preds}")
    print(f"Total correct: {total_correct}")


def main():
    args = get_args()
    
    if args.seed == -1:
        seed_num = random.randint(0, 100000000)
    else:
        seed_num = int(args.seed)

    print('random_int:', seed_num)
    print("Seed num:", seed_num)
    setup_seed(seed_num)

    logging.info(str(args.__dict__ if isinstance(args, argparse.ArgumentParser) else args))

    if args.state == 'llm_classify':
        dic = json.load(open(args.selectShot_dir)) if args.selectShot_dir and args.selectShot_dir != 'None' else None
        linkToLLM(args.input_file, args.save_file, dic, args)
        return
    
    # 如果是组合器模式，直接运行组合器
    if args.state == 'combination':
        run_combination_only(args)
        return

    num_labels = args.n_class
    task_idx2label = None
    args.label2idx_list, args.morph2idx_list = get_span_labels(args)
    
    bert_config = BertNerConfig.from_pretrained(
        args.bert_config_dir,
        hidden_dropout_prob=args.bert_dropout,
        attention_probs_dropout_prob=args.bert_dropout,
        model_dropout=args.model_dropout
    )
    model = BertNER.from_pretrained(args.bert_config_dir, config=bert_config, args=args)
    model.cuda()

    train_data_loader = get_loader(args, args.data_dir, "train", True)
    dev_data_loader = get_loader(args, args.data_dir, "dev", False)
    test_data_loader = get_loader(args, args.data_dir, "test", False)
    if args.test_mode == 'ori':
        test_data_loader = get_loader(args, args.data_dir, "test", False)
    elif args.test_mode == 'typos' and args.dataname == 'conll03':
        test_data_loader = get_loader(args, args.data_dir_typos, "test", False)
    elif args.test_mode == 'oov' and args.dataname == 'conll03':
        test_data_loader = get_loader(args, args.data_dir_oov, "test", False)
    elif args.test_mode == 'ood' and args.dataname == 'conll03':
        test_data_loader = get_loader(args, args.data_dir_ood, "test", False)
    else:
        raise Exception("Invalid dataname or test_mode! Please check")

    edl = Span_Evidence(args, num_labels)
    logger = get_logger(args, seed_num)
    framework = FewShotNERFramework(
        args, 
        logger, 
        task_idx2label, 
        train_data_loader, 
        dev_data_loader, 
        test_data_loader, 
        edl, 
        seed_num, 
        num_labels=num_labels
    )

    if args.state == 'train':
        framework.train(model)
        logger.info("training is ended! 🎉")

    if args.state == 'inference':
        model = torch.load(args.inference_model)
        if args.enable_combination:
            framework.inference_with_combination(model)
        else:
            framework.inference(model)
        logger.info("inference is ended!! 🎉")

if __name__ == '__main__':
    main()
