"""模型评估指标"""

from torch import Tensor
from sklearn.metrics import average_precision_score, roc_auc_score


def accuracy(logits: Tensor, y: Tensor) -> float:
    """计算多分类准确 (Accuracy)

    Args:
        logits:
            模型输出，形状为 (N, C)

        y:
            真实标签，形状为 (N,)

    Returns:
        Accuracy，取值范围 [0, 1]
    """
    return (logits.argmax(dim=-1) == y).float().mean().item()


def binary_auc(y_true: Tensor, y_score: Tensor) -> float:
    """计算二分类 ROC-AUC

    Args:
        y_true:
            真实二分类标签，形状为 (N,)

        y_score:
            模型输出分数（未经阈值化），形状为 (N,)

    Returns:
        ROC-AUC，取值范围 [0, 1]
    """
    return float(roc_auc_score(y_true.numpy(), y_score.numpy()))


def binary_ap(y_true: Tensor, y_score: Tensor) -> float:
    """计算二分类 Average Precision (AP)

    Args:
        y_true:
            真实二分类标签，形状为 (N,)

        y_score:
            模型输出分数（未经阈值化），形状为 (N,)

    Returns:
        Average Precision，取值范围 [0, 1]
    """
    return float(average_precision_score(y_true.numpy(), y_score.numpy()))
