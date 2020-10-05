from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import itertools
import os


def plot_confusion_matrix(cls_pred, cls_true, num_classes, out_path):
    """
    Plot confution matrix and export the figure at .png file
    :param cls_pred: List of predicted results ex:[0, 1, 0, 1]
    :param cls_true: List of labels ex: [1, 1, 0, 0]
    :param num_classes: The number of classes
    :param out_path: The directory of exporting .png file
    :return:
    """
    plt.clf()
    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred)
    print(cm)

    thresh = cm.max() / 2.
    plt.matshow(cm, interpolation='nearest')
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="black" if cm[i, j] > thresh else "white")

    # Make various adjustments to the plot.
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(out_path, 'confusion_matrix.png'))

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


def plot_roc_curve(Y_test, y_score, n_classes, pltlabel='ROC curve', color=[1, .5, .5], out_path="../checkpoints"):
    """
    Plot ROC curve and export the figure at .png file
    :param Y_test: Groundtruths ex: for two classes [[1, 0], [0, 1], ....]
    :param y_score: The confidence ex: for two classes [[0.3, 0.7], [0.25, .75], ...]
    :param n_classes: Number of classes
    :param pltlabel: Label of figure
    :param color: Curve color
    :param out_path: The directory of exporting .png file
    :return:
    """
    from sklearn.metrics import roc_curve, auc
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(Y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    # Compute micro-average ROC curve and ROC area
    # fpr["micro"], tpr["micro"], _ = roc_curve(Y_test.ravel(), y_score.ravel())
    # roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    # plt.figure()
    lw = 2

    for i in range(n_classes):
        plt.clf()
        plt.plot(fpr[i], tpr[i], color=color,
                 lw=lw, label=pltlabel + '(area = %0.2f)' % roc_auc[i])
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic Curve')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(out_path, 'ROC_cls_' + str(i) + '.png'))
        print(roc_auc[i])
