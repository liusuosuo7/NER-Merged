import torch
import numpy as np
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.optim import SGD
import time
import prettytable as pt
import os
import json
from metrics.function_metrics import span_f1_prune, ECE_Scores, get_predict_prune
from src.combined_spanner import CombinedSpanNER
import logging

logger = logging.getLogger(__name__)

class CombinedFewShotNERFramework:
    """集成组合器功能的Few-Shot NER框架"""
    
    def __init__(self, args, logger, task_idx2label, train_data_loader, val_data_loader, test_data_loader, edl, seed_num, num_labels):
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.test_data_loader = test_data_loader
        self.logger = logger
        self.seed = seed_num
        self.args = args
        self.eps = 1e-10
        self.learning_rate = args.lr
        self.load_ckpt = args.load_ckpt
        self.optimizer = args.optimizer
        self.annealing_start = 1e-6
        self.epoch_num = args.iteration
        self.edl = edl
        self.num_labels = num_labels
        self.task_idx2label = task_idx2label
        
        # 初始化组合器
        self.combined_spanner = CombinedSpanNER(args, num_labels)
        
        # 是否使用组合器模式
        self.use_combination = getattr(args, 'use_combination', False)
        self.models_config_path = getattr(args, 'models_config_path', None)
        
        if self.use_combination and self.models_config_path:
            self.load_combination_models()
    
    def load_combination_models(self):
        """加载组合器所需的多个模型"""
        if os.path.exists(self.models_config_path):
            with open(self.models_config_path, 'r', encoding='utf-8') as f:
                models_config = json.load(f)
            self.combined_spanner.load_models_from_config(models_config)
            self.logger.info(f"成功加载 {len(models_config)} 个模型用于组合")
        else:
            self.logger.warning(f"模型配置文件 {self.models_config_path} 不存在")
    
    def item(self, x):
        return x.item()

    def metric_combined(self, eval_dataset, mode):
        """使用组合器进行评估的度量方法"""
        pred_cnt = 0
        label_cnt = 0
        correct_cnt = 0
        context_results = []
        predict_results = []
        prob_results = []
        uncertainty_results = []

        # 使用组合器进行预测
        combined_results = self.combined_spanner.forward(eval_dataset)
        
        # 处理组合后的结果
        predictions = combined_results['predictions']
        uncertainties = combined_results['uncertainties']
        confidences = combined_results['confidences']
        
        # 转换结果格式以兼容原有的评估流程
        with torch.no_grad():
            for it, data in enumerate(eval_dataset):
                tokens, token_type_ids, all_span_idxs_ltoken, morph_idxs, span_label_ltoken, all_span_lens, all_span_weights, real_span_mask_ltoken, words, all_span_word, all_span_idxs = data
                
                batch_size = tokens.size(0)
                pred_tensor = torch.zeros_like(span_label_ltoken, dtype=torch.long)
                uncertainty_tensor = torch.ones_like(span_label_ltoken, dtype=torch.float)
                prob_tensor = torch.ones_like(span_label_ltoken, dtype=torch.float)
                
                for batch_i in range(batch_size):
                    for span_i, span_idx in enumerate(all_span_idxs[batch_i]):
                        if real_span_mask_ltoken[batch_i][span_i] == 0:
                            continue
                            
                        span_key = (it, batch_i, span_i, tuple(span_idx))
                        
                        if span_key in predictions:
                            # 获取标签ID
                            label = predictions[span_key]
                            if hasattr(self.args, 'label2idx_list') and self.args.label2idx_list:
                                label_id = next((idx for idx, (l, _) in enumerate(self.args.label2idx_list) if l == label), 0)
                            else:
                                label_id = 0
                            
                            pred_tensor[batch_i][span_i] = label_id
                            uncertainty_tensor[batch_i][span_i] = uncertainties.get(span_key, 1.0)
                            prob_tensor[batch_i][span_i] = confidences.get(span_key, 0.0)
                
                # 计算准确率指标
                correct, tmp_pred_cnt, tmp_label_cnt = span_f1_prune(pred_tensor, span_label_ltoken, real_span_mask_ltoken)
                pred_cnt += tmp_pred_cnt
                label_cnt += tmp_label_cnt
                correct_cnt += correct
                
                # 收集结果用于输出
                batch_results = get_predict_prune(self.args.label2idx_list, all_span_word, words, pred_tensor, span_label_ltoken, all_span_idxs, prob_tensor, uncertainty_tensor)
                context_results += batch_results
                
                predict_results.append(pred_tensor)
                prob_results.append(prob_tensor)
                uncertainty_results.append(uncertainty_tensor)

        # 计算最终指标
        precision = correct_cnt / (pred_cnt + 0.0) if pred_cnt > 0 else 0.0
        recall = correct_cnt / (label_cnt + 0.0) if label_cnt > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall + float("1e-8")) if (precision + recall) > 0 else 0.0
        
        # 计算ECE（期望校准误差）
        ece = 0.0  # 组合器的ECE计算需要特殊处理
        
        if mode == 'test':
            results_dir = os.path.join(self.args.results_dir, f"{self.args.dataname}_combined_local_model.jsonl")
            sent_num = len(context_results)

            with open(results_dir, 'w', encoding='utf-8') as fout:
                for idx in range(sent_num):
                    json.dump(context_results[idx], fout, ensure_ascii=False)
                    fout.write('\n')
            
            # 保存不确定span用于LLM链接
            uncertain_spans = self.combined_spanner.get_uncertainty_for_linking(combined_results, self.args.threshold)
            uncertain_file = os.path.join(self.args.results_dir, f"{self.args.dataname}_uncertain_spans.json")
            with open(uncertain_file, 'w', encoding='utf-8') as f:
                json.dump(uncertain_spans, f, ensure_ascii=False, indent=2)
            self.logger.info(f"保存了 {len(uncertain_spans)} 个不确定span到 {uncertain_file}")

        return precision, recall, f1, ece

    def metric_single(self, model, eval_dataset, mode):
        """使用单个模型进行评估的度量方法（原有方法）"""
        pred_cnt = 0
        label_cnt = 0
        correct_cnt = 0
        context_results = []
        predict_results = []
        prob_results = []
        uncertainty_results = []

        with torch.no_grad():
            for it, data in enumerate(eval_dataset):
                gold_tokens_list = []
                pred_scores_list = []
                pred_list = []
                batch_soft = []

                tokens, token_type_ids, all_span_idxs_ltoken, morph_idxs, span_label_ltoken, all_span_lens, all_span_weights, real_span_mask_ltoken, words, all_span_word, all_span_idxs = data
                loadall = [tokens, token_type_ids, all_span_idxs_ltoken, morph_idxs, span_label_ltoken, all_span_lens, all_span_weights, real_span_mask_ltoken, words, all_span_word, all_span_idxs]
                attention_mask = (tokens != 0).long()
                logits = model(loadall, all_span_lens, all_span_idxs_ltoken, tokens, attention_mask, token_type_ids)
                predicts, uncertainty = self.edl.pred(logits)
                correct, tmp_pred_cnt, tmp_label_cnt = span_f1_prune(predicts, span_label_ltoken, real_span_mask_ltoken)
                pred_cls, pred_scores, tgt_cls = self.edl.ece_value(logits, span_label_ltoken, real_span_mask_ltoken)

                prob, pred_id = torch.max(predicts, 2)
                batch_results = get_predict_prune(self.args.label2idx_list, all_span_word, words, pred_id, span_label_ltoken, all_span_idxs, prob, uncertainty)
                pred_cnt += tmp_pred_cnt
                label_cnt += tmp_label_cnt
                correct_cnt += correct
                batch_soft += span_label_ltoken
                prob_results += prob

                predict_results += pred_id
                uncertainty_results += uncertainty

                context_results += batch_results

                pred_list.append(pred_cls)
                pred_scores_list.append(pred_scores)
                gold_tokens_list.append(tgt_cls)

            gold_tokens_cat = torch.cat(gold_tokens_list, dim=0)
            pred_scores_cat = torch.cat(pred_scores_list, dim=0)
            pred_cat = torch.cat(pred_list, dim=0)

            ece = ECE_Scores(pred_cat, gold_tokens_cat, pred_scores_cat)
            precision = correct_cnt / (pred_cnt + 0.0)
            recall = correct_cnt / (label_cnt + 0.0)
            f1 = 2 * precision * recall / (precision + recall + float("1e-8"))
            if mode == 'test':
                results_dir = os.path.join(self.args.results_dir, f"{self.args.dataname}_{self.args.uncertainty_type}_local_model.jsonl")
                sent_num = len(context_results)

                with open(results_dir, 'w', encoding='utf-8') as fout:
                    for idx in range(sent_num):
                        json.dump(context_results[idx], fout, ensure_ascii=False)
                        fout.write('\n')

            return precision, recall, f1, ece

    def eval(self, model=None, mode=None):
        """评估方法，根据是否使用组合器选择不同的评估策略"""
        if self.use_combination:
            # 使用组合器评估
            if mode == 'dev':
                self.logger.info("使用组合器在验证集上评估")
                precision, recall, f1, ece = self.metric_combined(self.val_data_loader, mode='dev')
                self.logger.info('{} Combined F1 {}'.format("dev", f1))
                table = pt.PrettyTable(["{}".format("dev"), "Precision", "Recall", 'F1', 'ECE'])

            elif mode == 'test':
                self.logger.info("使用组合器在测试集上评估: " + str(self.args.test_mode))
                precision, recall, f1, ece = self.metric_combined(self.test_data_loader, mode='test')
                self.logger.info('{} Combined F1 {}'.format("test", f1))
                table = pt.PrettyTable(["{}".format("test"), "Precision", "Recall", 'F1', 'ECE'])
        else:
            # 使用单个模型评估
            if mode == 'dev':
                self.logger.info("Use val dataset")
                precision, recall, f1, ece = self.metric_single(model, self.val_data_loader, mode='dev')
                self.logger.info('{} Label F1 {}'.format("dev", f1))
                table = pt.PrettyTable(["{}".format("dev"), "Precision", "Recall", 'F1', 'ECE'])

            elif mode == 'test':
                self.logger.info("Use " + str(self.args.test_mode) + " test dataset")
                precision, recall, f1, ece = self.metric_single(model, self.test_data_loader, mode='test')
                self.logger.info('{} Label F1 {}'.format("test", f1))
                table = pt.PrettyTable(["{}".format("test"), "Precision", "Recall", 'F1', 'ECE'])

        table.add_row(["Entity"] + ["{:3.4f}".format(x) for x in [precision, recall, f1, ece]])
        self.logger.info("\n{}".format(table))
        return f1, ece

    def train(self, model):
        """训练方法 - 仅在非组合器模式下使用"""
        if self.use_combination:
            self.logger.info("组合器模式不需要训练，所有子模型已预训练")
            return
            
        self.logger.info("Start training...")

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        if self.optimizer == "adamw":
            optimizer = AdamW(optimizer_grouped_parameters, betas=(0.9, 0.98), lr=self.learning_rate, eps=self.args.adam_epsilon)
        elif self.optimizer == "sgd":
            optimizer = SGD(optimizer_grouped_parameters, self.learning_rate, momentum=0.9)
        elif self.optimizer == "torch.adam":
            optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.args.lr, eps=self.args.adam_epsilon, weight_decay=self.args.weight_decay)

        t_total = len(self.train_data_loader) * self.args.iteration
        warmup_steps = int(self.args.warmup_proportion * t_total)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)

        model.train()
        best_f1 = 0.0
        best_step = 0
        iter_loss = 0.0

        for idx in range(self.args.iteration):
            pred_cnt = 0
            label_cnt = 0
            correct_cnt = 0
            epoch_start = time.time()
            self.logger.info("training...")

            for it in range(len(self.train_data_loader)):
                loss = 0
                tokens, token_type_ids, all_span_idxs_ltoken, morph_idxs, span_label_ltoken, all_span_lens, all_span_weights, real_span_mask_ltoken, words, all_span_word, all_span_idxs = next(iter(self.train_data_loader))
                loadall = [tokens, token_type_ids, all_span_idxs_ltoken, morph_idxs, span_label_ltoken, all_span_lens, all_span_weights, real_span_mask_ltoken, words, all_span_word, all_span_idxs]
                attention_mask = (tokens != 0).long()
                logits = model(loadall, all_span_lens, all_span_idxs_ltoken, tokens, attention_mask, token_type_ids)
                loss, pred = self.edl.loss(logits, loadall, span_label_ltoken, real_span_mask_ltoken, idx)
                correct, tmp_pred_cnt, tmp_label_cnt = span_f1_prune(pred, span_label_ltoken, real_span_mask_ltoken)

                pred_cnt += tmp_pred_cnt
                label_cnt += tmp_label_cnt
                correct_cnt += correct

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                iter_loss += self.item(loss.data)

            epoch_finish = time.time()
            epoch_cost = epoch_finish - epoch_start
            precision = correct_cnt / (pred_cnt + 0.)
            recall = correct_cnt / (label_cnt + 0.)
            f1 = 2 * precision * recall / (precision + recall + float("1e-8"))

            self.logger.info("Time '%.2f's" % epoch_cost)
            self.logger.info('step: {0:4} | loss: {1:2.6f} | [ENTITY] precision: {2:3.4f}, recall: {3:3.4f}, f1: {4:3.4f}'.format(idx + 1, iter_loss, precision, recall, f1))

            if (idx + 1) % 1 == 0:
                f1, ece = self.eval(model, mode='dev')
                self.inference(model)
                if f1 > best_f1:
                    best_step = idx + 1
                    best_f1 = f1
                    if self.args.load_ckpt:
                        torch.save(model, self.args.results_dir + self.args.loss + str(self.args.seed) + '_model.pkl')

                if (idx + 1) > best_step + self.args.early_stop:
                    self.logger.info('Early stop!')
                    return

            iter_loss = 0.
            pred_cnt = 0
            label_cnt = 0
            correct_cnt = 0

    def inference(self, model=None):
        """推理方法"""
        f1, ece = self.eval(model, mode='test')