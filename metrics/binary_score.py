import numpy as np 
import torch 
from sklearn import metrics

from sklearn.utils import column_or_1d, assert_all_finite, check_consistent_length
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.extmath import stable_cumsum

def get_metrics(scores, labels):
    if torch.is_tensor(labels):
        unique = labels.unique()
        labels = labels.numpy()
    else:
        unique = np.unique(labels)
    if torch.is_tensor(scores):
        scores=scores.numpy()
    if len(unique) == 1:
        return 2
    elif len(unique) == 2:
        val_dict = dict()
        try:
            fps, tps, thresholds = metrics._ranking._binary_clf_curve(labels, scores, pos_label=1)
            fpr_tpr_095, thresh_fpr_tpr095= _compute_fpr_tpr_(fps, tps, thresholds)
            dice, thresh_dice = _compute_best_f1(fps, tps, thresholds)
            prec, rec, _ = _compute_prc(fps, tps, thresholds)
            auprc = float(_compute_auprc(prec, rec))
            fpr, tpr, _ = _compute_roc(fps, tps, thresholds)
            auroc = float(metrics.auc(fpr, tpr))

            val_dict['AUROC'] = auroc 
            val_dict['AUPRC'] = auprc 
            val_dict['FPR@TPR=0.95'] = float(fpr_tpr_095)
            val_dict['DICE'] = float(dice)
            return val_dict
        except ValueError:
            val_dict['AUROC'] = np.NaN 
            val_dict['AUPRC'] = np.NaN 
            val_dict['FPR@TPR=0.95'] = np.NaN
            val_dict['DICE'] = np.NaN
            return val_dict
    else:
        raise Exception("Too many different labels for score based auroc")

def _compute_best_f1(fps, tps, thresholds):
    dices = 2*tps / (tps +fps + tps[-1])
    ind = np.argmax(dices)
    return dices[ind], thresholds[ind]

def _compute_fpr_tpr_(fps, tps, thresholds, tpr_rate=0.95):
    tpr = tps/tps[-1]
    ind = np.searchsorted(tpr, tpr_rate)
    return fps[ind]/fps[-1], thresholds[ind]


def _compute_roc(fps, tps, thresholds, drop_intermediate=True):
    if drop_intermediate and len(fps) > 2:
        optimal_idxs = np.where(np.r_[True,
                                      np.logical_or(np.diff(fps, 2),
                                                    np.diff(tps, 2)),
                                      True])[0]
        fps = fps[optimal_idxs]
        tps = tps[optimal_idxs]
        thresholds = thresholds[optimal_idxs]

    # Add an extra threshold position
    # to make sure that the curve starts at (0, 0)
    tps = np.r_[0, tps]
    fps = np.r_[0, fps]
    thresholds = np.r_[thresholds[0] + 1, thresholds]

    if fps[-1] <= 0:
        # warnings.warn("No negative samples in y_true, "
        #               "false positive value should be meaningless",
        #               UndefinedMetricWarning)
        fpr = np.repeat(np.nan, fps.shape)
    else:
        fpr = fps / fps[-1]

    if tps[-1] <= 0:
        # warnings.warn("No positive samples in y_true, "
        #               "true positive value should be meaningless",
        #               UndefinedMetricWarning)
        tpr = np.repeat(np.nan, tps.shape)
    else:
        tpr = tps / tps[-1]

    return fpr, tpr, thresholds

def _compute_auprc(prec, rec):
    return -np.sum(np.diff(rec) * np.array(prec)[:-1])

def _compute_prc(fps, tps, thresholds):
    precision = tps / (tps + fps)
    precision[np.isnan(precision)] = 0
    recall = tps / tps[-1]

    # stop when full recall attained
    # and reverse the outputs so recall is decreasing
    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)
    return np.r_[precision[sl], 1], np.r_[recall[sl], 0], thresholds[sl]