import torch
import numpy as np

def update_conf_matrix(confusion_matrix, labels, preds):
    for l, p in zip(labels.view(-1), preds.view(-1)):
        confusion_matrix[l, p] += 1
    return confusion_matrix

def calc_macro_f1_score(confusion_matrix):
    precision = confusion_matrix.diagonal()/confusion_matrix.sum(dim=0)
    recall = confusion_matrix.diagonal()/confusion_matrix.sum(dim=1)
    f1 = (2*(precision*recall)/(precision+recall))
    f1[f1 != f1] = 0
    return f1.mean()

def calc_accuracy(confusion_matrix):
    return confusion_matrix.diagonal().sum()/confusion_matrix.sum()

def calc_weighted_quadratic_kappa(confusion_matrix):
    # Adapated from
    # https://github.com/scikit-learn/scikit-learn/blob/15a949460/sklearn/metrics/_classification.py#L563
    n_classes = confusion_matrix.size()[0]
    sum0 = confusion_matrix.sum(dim=0)
    sum1 = confusion_matrix.sum(dim=1)
    expected = torch.outer(sum0, sum1) / confusion_matrix.sum()

    w_mat = torch.zeros([n_classes, n_classes])
    w_mat += torch.arange(n_classes)
    w_mat = (w_mat - w_mat.T) ** 2

    k = torch.sum(w_mat * confusion_matrix) / torch.sum(w_mat * expected)
    return 1 - k.item()

if __name__ == "__main__":
    cm = torch.tensor([[7,8,9],[1,2,3],[3,2,1]])
    print(calc_macro_f1_score(cm))
    plt.show()