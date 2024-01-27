from sklearn.metrics import roc_curve, auc, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import logging

logger = logging.getLogger(__name__)

# plot
def plot_confusion_matrix(y_true, y_pred, *model):

    if model and model[0]:
        y_pred = y_pred[model[0]]  #
        cm = confusion_matrix(y_true, y_pred)
    else:
        cm = confusion_matrix(y_true, y_pred)

    fig = plt.figure(figsize=(10, 5))
    group_names = ["TP", "FP", "FN", "TP"]
    group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
    group_perc = ["{0:.2%}".format(value) for value in cm.flatten() / np.sum(cm)]
    label = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_perc)]
    label = np.asarray(label).reshape(2, 2)
    heatmap = sns.heatmap(cm, annot=label, annot_kws={'size': 12}, fmt='', cmap='YlGnBu')
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=0, ha='center', fontsize=10)
    plt.title('Confusion Matrix', fontsize=12, color='darkblue')
    plt.ylabel('True label', fontsize=10)
    plt.xlabel('Predicted label', fontsize=10)
    return fig

def plot_roc_curve(y_true, y_prob, model = None):

    if model:
        y_prob = y_prob[model]

    false_positive_rate, true_positive_rate, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    fig = plt.figure(figsize=(10, 5))
    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate, true_positive_rate, color='red', label='AUC = %0.2f' % roc_auc, linestyle="--")
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.axis('tight')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    return fig