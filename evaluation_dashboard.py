"""
Basic evaluation dashboard for comparing different techniques.
"""

import matplotlib.pyplot as plt

def plot_metrics(metrics_dict):
    """
    metrics_dict = {
        'Meta-Learner': {'f1': 0.82, 'auc': 0.91},
        'SMOTE': {'f1': 0.76, 'auc': 0.89},
        ...
    }
    """
    labels = list(metrics_dict.keys())
    f1s = [v['f1'] for v in metrics_dict.values()]
    aucs = [v['auc'] for v in metrics_dict.values()]

    x = range(len(labels))
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.bar(x, f1s, tick_label=labels)
    plt.title("F1 Scores")
    plt.subplot(1, 2, 2)
    plt.bar(x, aucs, tick_label=labels)
    plt.title("AUC Scores")
    plt.tight_layout()
    plt.show()
