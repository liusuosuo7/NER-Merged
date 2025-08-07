# -*- coding: utf-8 -*-

def get_chunks_onesent(seq):
    """从一个句子的标签序列中提取chunks"""
    prev_tag, prev_type = 'O', ''
    chunks = []
    begin_idx = 0
    for i, chunk in enumerate(seq + ['O']):
        tag = chunk[0]
        type_ = chunk.split('-')[-1]

        if end_of_chunk(prev_tag, tag, prev_type, type_):
            chunks.append((prev_type, begin_idx, i-1))
        if start_of_chunk(prev_tag, tag, prev_type, type_):
            begin_idx = i
        prev_tag, prev_type = tag, type_

    return chunks


def end_of_chunk(prev_tag, tag, prev_type, type_):
    """判断是否为chunk的结束"""
    chunk_end = False

    if prev_tag == 'E': chunk_end = True
    if prev_tag == 'S': chunk_end = True
    if prev_tag == 'B' and tag == 'B': chunk_end = True
    if prev_tag == 'B' and tag == 'O': chunk_end = True
    if prev_tag == 'I' and tag == 'B': chunk_end = True
    if prev_tag == 'I' and tag == 'O': chunk_end = True

    if prev_tag != 'O' and prev_type != type_ and tag == 'I':
        chunk_end = True

    return chunk_end


def start_of_chunk(prev_tag, tag, prev_type, type_):
    """判断是否为chunk的开始"""
    chunk_start = False

    if tag == 'B': chunk_start = True
    if tag == 'S': chunk_start = True
    if prev_tag == 'O' and tag == 'I': chunk_start = True
    if tag != 'O' and tag != '.' and prev_type != type_:
        chunk_start = True

    return chunk_start


def evaluate_chunk_level(pred_chunks, true_chunks):
    """chunk级别的评估"""
    correct_preds, total_correct, total_preds = 0., 0., 0.
    correct_preds += len(set(true_chunks) & set(pred_chunks))
    total_preds += len(pred_chunks)
    total_correct += len(true_chunks)
    p = correct_preds / total_preds if correct_preds > 0 else 0
    r = correct_preds / total_correct if correct_preds > 0 else 0
    f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0

    return f1, p, r, correct_preds, total_preds, total_correct


def evaluate_ByCategory(pred_chunks, true_chunks, classes):
    """按类别评估F1分数"""
    class2f1 = {}
    
    for class_name in classes:
        pred_class_chunks = [chunk for chunk in pred_chunks if chunk[0] == class_name]
        true_class_chunks = [chunk for chunk in true_chunks if chunk[0] == class_name]
        
        f1, p, r, correct_preds, total_preds, total_correct = evaluate_chunk_level(
            pred_class_chunks, true_class_chunks)
        
        class2f1[class_name] = f1
    
    return class2f1


def f1_score(y_true, y_pred):
    """计算F1分数"""
    def safe_division(numr, denr, on_err=0.0):
        return on_err if denr == 0.0 else numr / denr

    assert len(y_true) == len(y_pred)
    
    true_entities = set(y_true)
    pred_entities = set(y_pred)
    
    nb_correct = len(true_entities & pred_entities)
    nb_pred = len(pred_entities)
    nb_true = len(true_entities)

    p = safe_division(nb_correct, nb_pred)
    r = safe_division(nb_correct, nb_true)
    score = safe_division(2 * p * r, p + r)
    
    return score, p, r


def precision_recall_f1(y_true, y_pred, average='micro'):
    """计算精确率、召回率和F1分数"""
    if average == 'micro':
        return f1_score(y_true, y_pred)
    else:
        # 如果需要宏平均等其他方式，可以在这里扩展
        raise NotImplementedError(f"Average method '{average}' not implemented")


def classification_report(y_true, y_pred, classes):
    """生成分类报告"""
    report = {}
    
    for class_name in classes:
        true_class = [chunk for chunk in y_true if chunk[0] == class_name]
        pred_class = [chunk for chunk in y_pred if chunk[0] == class_name]
        
        f1, p, r = f1_score(true_class, pred_class)
        
        report[class_name] = {
            'precision': p,
            'recall': r,
            'f1-score': f1,
            'support': len(true_class)
        }
    
    # 计算总体指标
    overall_f1, overall_p, overall_r = f1_score(y_true, y_pred)
    report['overall'] = {
        'precision': overall_p,
        'recall': overall_r,
        'f1-score': overall_f1,
        'support': len(y_true)
    }
    
    return report