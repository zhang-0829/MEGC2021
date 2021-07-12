import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def accuracy(output, target):
    with torch.no_grad():
        pred = (output > 0.5).long()
        pred = pred.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        return accuracy_score(y_pred=pred, y_true=target)

def precision(output, target):
    with torch.no_grad():
        pred = (output > 0.5).long()
        pred = pred.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        return precision_score(y_pred=pred, y_true=target)

def recall(output, target):
    with torch.no_grad():
        pred = (output > 0.5).long()
        pred = pred.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        return recall_score(y_pred=pred, y_true=target)

def conf_mat(output, target):
    with torch.no_grad():
        pred = (output > 0.5).long()
        pred = pred.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        return confusion_matrix(y_pred=pred, y_true=target)

def f1(output, target):
    with torch.no_grad():
        pred = (output > 0.5).long()
        pred = pred.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        return f1_score(y_pred=pred, y_true=target)
