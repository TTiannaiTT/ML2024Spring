import numpy as np
from sklearn.metrics import roc_auc_score

# python sklearn包计算auc
def get_auc(y_labels, y_scores):
    auc = roc_auc_score(y_labels, y_scores)
    print('AUC calculated by sklearn tool is {}'.format(auc))
    return auc

# 方法1计算auc
def calculate_auc_func1(y_labels, y_scores):
    pos_sample_ids = [i for i in range(len(y_labels)) if y_labels[i] == 1]
    neg_sample_ids = [i for i in range(len(y_labels)) if y_labels[i] == 0]

    sum_indicator_value = 0
    for i in pos_sample_ids:
        for j in neg_sample_ids:
            if y_scores[i] > y_scores[j]:
                sum_indicator_value += 1
            elif y_scores[i] == y_scores[j]:
                sum_indicator_value += 0.5

    auc = sum_indicator_value/(len(pos_sample_ids) * len(neg_sample_ids))
    print('AUC calculated by function1 is {:.2f}'.format(auc))
    return auc

y_labels = np.array([0, 0, 1, 1, 0,1,0,1,1,0])
y1_score = np.array([0.38,0.28,0.67,0.38,0.11,0.43,0.88,0.54,0.29,0.75])
y2_score = np.array([0.19,0.89,0.47,0.89,0.95,0.49,0.23,0.66,0.15,0.66])
get_auc(y_labels, y1_score)
get_auc(y_labels, y2_score)
