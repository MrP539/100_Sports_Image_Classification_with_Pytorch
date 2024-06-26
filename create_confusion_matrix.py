#import sklearn
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sklearn
def CREATE_CONFUSION_MATRICS(y_actual,y_pred):
    cm =sklearn.metrics.confusion_matrix(y_true=y_actual, y_pred=y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap='Blues', xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Normalized Confusion Matrix')
    plt.show()

