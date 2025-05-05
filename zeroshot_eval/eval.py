import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    matthews_corrcoef,
    confusion_matrix,
    accuracy_score,
    auc,
    roc_auc_score,
    roc_curve,
    classification_report,
)
from sklearn.metrics import precision_recall_curve, f1_score

""" ROC CURVE """


def plot_roc(y_pred, y_true, roc_name, plot=False):
    # given the test_ground_truth, and test_predictions
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)

    roc_auc = auc(fpr, tpr)

    if plot:
        plt.figure(dpi=100)
        plt.title(roc_name)
        plt.plot(fpr, tpr, "b", label="AUC = %0.2f" % roc_auc)
        plt.legend(loc="lower right")
        plt.plot([0, 1], [0, 1], "r--")
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel("True Positive Rate")
        plt.xlabel("False Positive Rate")
        plt.show()
    return fpr, tpr, thresholds, roc_auc


""" PRECISION-RECALL CURVE """


def plot_pr(y_pred, y_true, pr_name, plot=False):
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    pr_auc = auc(recall, precision)
    # plot the precision-recall curves
    baseline = len(y_true[y_true == 1]) / len(y_true)

    if plot:
        plt.figure(dpi=100)
        plt.title(pr_name)
        plt.plot(recall, precision, "b", label="AUC = %0.2f" % pr_auc)
        # axis labels
        plt.legend(loc="lower right")
        plt.plot([0, 1], [baseline, baseline], "r--")
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        # show the plot
        plt.show()
    return precision, recall, thresholds, pr_auc


# J = TP/(TP+FN) + TN/(TN+FP) - 1 = tpr - fpr
def choose_operating_point(fpr, tpr, thresholds):
    sens = 0
    spec = 0
    J = 0
    for _fpr, _tpr in zip(fpr, tpr):
        if _tpr - _fpr > J:
            sens = _tpr
            spec = 1 - _fpr
            J = _tpr - _fpr
    return sens, spec


def evaluate(
    y_pred,
    y_true,
    cxr_labels,
    cxr_sub_labels,
    roc_name="Receiver Operating Characteristic",
    pr_name="Precision-Recall Curve",
    label_idx_map=None,
    plot=False,
):
    """
    We expect `y_pred` and `y_true` to be numpy arrays, both of shape (num_samples, num_classes)

    `y_pred` is a numpy array consisting of probability scores with all values in range 0-1.

    `y_true` is a numpy array consisting of binary values representing if a class is present in
    the cxr.

    This function provides all relevant evaluation information, ROC, AUROC, Sensitivity, Specificity,
    PR-Curve, Precision, Recall for each class.
    """
    import warnings

    warnings.filterwarnings("ignore")

    num_classes = y_pred.shape[-1]  # number of total labels

    dataframes_roc = []
    dataframes_pr = []
    for i in range(num_classes):
        #         print('{}.'.format(cxr_labels[i]))

        if label_idx_map is None:
            y_pred_i = y_pred[:, i]  # (num_samples,)
            y_true_i = y_true[:, i]  # (num_samples,)

        else:
            y_pred_i = y_pred[:, i]  # (num_samples,)

            true_index = label_idx_map[cxr_labels[i]]
            y_true_i = y_true[:, true_index]  # (num_samples,)

        cxr_label = cxr_labels[i]

        """ ROC CURVE """
        roc_name = cxr_label + " ROC Curve"
        fpr, tpr, thresholds, roc_auc = plot_roc(y_pred_i, y_true_i, roc_name, plot)

        sens, spec = choose_operating_point(fpr, tpr, thresholds)

        results = [[roc_auc]]
        df = pd.DataFrame(results, columns=[cxr_label + "_auc"])
        dataframes_roc.append(df)

        """ PRECISION-RECALL CURVE """
        pr_name = cxr_label + " Precision-Recall Curve"
        precision, recall, thresholds, pr_auc = plot_pr(
            y_pred_i, y_true_i, pr_name, plot
        )
        results = [[pr_auc]]
        df = pd.DataFrame(results, columns=[cxr_label + "_pr_auc"])
        dataframes_pr.append(df)

    dfs_roc = pd.concat(dataframes_roc, axis=1)
    dfs_roc["auc_average"] = dfs_roc.mean(numeric_only=True, axis=1)
    dfs_roc["auc_median"] = dfs_roc.median(numeric_only=True, axis=1)

    dfs_pr = pd.concat(dataframes_pr, axis=1)
    dfs_pr["pr_auc_average"] = dfs_pr.mean(numeric_only=True, axis=1)
    dfs_pr["pr_auc_median"] = dfs_pr.median(numeric_only=True, axis=1)

    results_df = pd.concat([dfs_roc, dfs_pr], axis=1)

    results_df["sub_auc_average"] = results_df[
        list(map(lambda x: x + "_auc", cxr_sub_labels))
    ].mean(axis=1)

    results_df["sub_auc_median"] = results_df[
        list(map(lambda x: x + "_auc", cxr_sub_labels))
    ].median(axis=1)

    results_df["sub_pr_auc_average"] = results_df[
        list(map(lambda x: x + "_pr_auc", cxr_sub_labels))
    ].mean(axis=1)

    results_df["sub_pr_auc_median"] = results_df[
        list(map(lambda x: x + "_pr_auc", cxr_sub_labels))
    ].median(axis=1)

    return results_df
