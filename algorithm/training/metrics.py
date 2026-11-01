"""Fixed-vocabulary metrics, including absent classes with zero recall/F1."""
import numpy as np


def classification_metrics(targets, predictions, class_count=5):
    truth, guessed = np.asarray(targets), np.asarray(predictions)
    if truth.ndim != 1 or truth.shape != guessed.shape:
        raise ValueError("targets/predictions must be matching vectors")
    if truth.dtype.kind not in "iu" or guessed.dtype.kind not in "iu":
        raise ValueError("class indices must be integers")
    if np.any((truth < 0) | (truth >= class_count) | (guessed < 0) | (guessed >= class_count)):
        raise ValueError("class index outside vocabulary")
    matrix = np.zeros((class_count, class_count), dtype=np.int64)
    np.add.at(matrix, (truth, guessed), 1)
    return metrics_from_confusion(matrix)


def metrics_from_confusion(matrix):
    """Mergeable sufficient statistic; validation need not retain all guesses."""
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not matrix.shape[0] or matrix.dtype.kind not in "iu" or (matrix < 0).any():
        raise ValueError("confusion matrix must be nonnegative square integers")
    class_count = matrix.shape[0]
    support, predicted = matrix.sum(1), matrix.sum(0)
    precision = np.divide(matrix.diagonal(), predicted, out=np.zeros(class_count), where=predicted > 0)
    recall = np.divide(matrix.diagonal(), support, out=np.zeros(class_count), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(class_count), where=precision + recall > 0)
    return dict(confusion_matrix=matrix.tolist(), support=support.tolist(), precision=precision.tolist(),
                recall=recall.tolist(), f1=f1.tolist(), macro_f1=float(f1.mean()),
                balanced_accuracy=float(recall.mean()), samples=int(matrix.sum()),
                absent_classes=np.flatnonzero(support == 0).tolist())
