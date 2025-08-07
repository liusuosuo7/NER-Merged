import torch
import numpy as np
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.optim import SGD
import time
import prettytable as pt
import os
import json
import pickle
from metrics.function_metrics import span_f1_prune, ECE_Scores, get_predict_prune
from combination.comb_voting import CombByVoting

class FewShotNERFramework:
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

    def item(self, x):
        return x.item()

    def metric(self, model, eval_dataset, mode):
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

                # 保存预测结果用于组合器
                self.save_predictions_for_combination(context_results, mode)

            return precision, recall, f1, ece

    def eval(self, model, mode=None):
        if mode == 'dev':
            self.logger.info("Use val dataset")
            precision, recall, f1, ece = self.metric(model, self.val_data_loader, mode='dev')
            self.logger.info('{} Label F1 {}'.format("dev", f1))
            table = pt.PrettyTable(["{}".format("dev"), "Precision", "Recall", 'F1', 'ECE'])

        elif mode == 'test':
            self.logger.info("Use " + str(self.args.test_mode) + " test dataset")
            precision, recall, f1, ece = self.metric(model, self.test_data_loader, mode='test')
            self.logger.info('{} Label F1 {}'.format("test", f1))
            table = pt.PrettyTable(["{}".format("test"), "Precision", "Recall", 'F1', 'ECE'])

        table.add_row(["Entity"] + ["{:3.4f}".format(x) for x in [precision, recall, f1, ece]])
        self.logger.info("\n{}".format(table))
        return f1, ece

    def train(self, model):
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

    def inference(self, model):
        f1, ece = self.eval(model, mode='test')

    def save_predictions_for_combination(self, context_results, mode):
        """保存预测结果用于模型组合"""
        if not self.args.enable_combination:
            return
        
        pred_chunks = []
        true_chunks = []
        
        for result in context_results:
            # 从context_results中提取chunk信息
            if 'pred_entities' in result:
                for entity in result['pred_entities']:
                    label = entity.get('label', 'O')
                    start = entity.get('start', 0)
                    end = entity.get('end', 0)
                    sent_id = result.get('sent_id', 0)
                    pred_chunks.append((label, start, end, sent_id))
            
            if 'true_entities' in result:
                for entity in result['true_entities']:
                    label = entity.get('label', 'O')
                    start = entity.get('start', 0)
                    end = entity.get('end', 0)
                    sent_id = result.get('sent_id', 0)
                    true_chunks.append((label, start, end, sent_id))
        
        # 保存结果
        save_dir = os.path.join(self.args.results_dir, "combination_data")
        os.makedirs(save_dir, exist_ok=True)
        
        save_file = os.path.join(save_dir, f"linkner_{mode}_results.pkl")
        with open(save_file, 'wb') as f:
            pickle.dump((pred_chunks, true_chunks), f)
        
        self.logger.info(f"Saved predictions for combination: {save_file}")

    def run_model_combination(self, model_files=None, model_f1s=None, method='voting_majority'):
        """运行模型组合"""
        if not self.args.enable_combination:
            self.logger.info("Model combination is disabled")
            return None
        
        # 使用参数或者配置文件中的模型列表
        if model_files is None:
            model_files = self.args.model_files
        if model_f1s is None:
            model_f1s = self.args.model_f1s
        
        if not model_files:
            self.logger.warning("No model files specified for combination")
            return None
        
        # 确保F1分数列表长度与模型文件列表长度一致
        if not model_f1s or len(model_f1s) != len(model_files):
            self.logger.warning("Using default F1 scores for combination")
            model_f1s = [1.0] * len(model_files)
        
        # 获取数据集类别
        classes = [self.args.label2idx_list[i][0] for i in range(len(self.args.label2idx_list)) 
                  if self.args.label2idx_list[i][0] != 'O']
        
        # 创建组合器
        combiner = CombByVoting(
            dataname=self.args.dataname,
            file_dir=os.path.dirname(model_files[0]) if model_files else self.args.results_dir,
            fmodels=[os.path.basename(f) for f in model_files],
            f1s=model_f1s,
            cmodelname=f"LinkNER_Combined_{method}",
            classes=classes,
            fn_stand_res="",  # 不需要标准结果文件
            fn_prob=self.args.prob_file if hasattr(self.args, 'prob_file') else "",
            result_dir=self.args.combination_result_dir
        )
        
        # 执行组合
        self.logger.info(f"Running model combination with method: {method}")
        results = combiner.combine_results(method=method)
        
        f1, p, r, correct_preds, total_preds, total_correct = results
        self.logger.info(f"Combination results - F1: {f1:.4f}, Precision: {p:.4f}, Recall: {r:.4f}")
        
        return results

    def inference_with_combination(self, model):
        """推理时使用组合器功能"""
        # 首先运行标准推理
        f1, ece = self.eval(model, mode='test')
        
        # 如果启用了组合器功能，运行模型组合
        if self.args.enable_combination and self.args.model_files:
            self.logger.info("Running model combination...")
            combination_results = self.run_model_combination(
                method=self.args.combination_method
            )
            
            if combination_results:
                comb_f1 = combination_results[0]
                self.logger.info(f"Individual model F1: {f1:.4f}")
                self.logger.info(f"Combined model F1: {comb_f1:.4f}")
                self.logger.info(f"Improvement: {comb_f1 - f1:.4f}")
        
        return f1, ece