import torch
import numpy as np

# 分割指标

def dice_coefficient(pred:   torch.Tensor,
                     target: torch.Tensor,
                     threshold: float = 0.5,
                     smooth:    float = 1e-6) -> float:
    """计算 Dice 系数（分割版的 F1 分数）。"""
    pred   = (torch.sigmoid(pred) > threshold).float()
    target = target.float()

    pred   = pred.view(-1)
    target = target.view(-1)

    inter  = (pred * target).sum()
    return ((2.0 * inter + smooth) /
            (pred.sum() + target.sum() + smooth)).item()


def iou_score(pred:   torch.Tensor,
              target: torch.Tensor,
              threshold: float = 0.5,
              smooth:    float = 1e-6) -> float:
    """计算 IoU（Jaccard 指数）。"""
    pred   = (torch.sigmoid(pred) > threshold).float()
    target = target.float()

    pred   = pred.view(-1)
    target = target.view(-1)

    inter  = (pred * target).sum()
    union  = pred.sum() + target.sum() - inter
    return ((inter + smooth) / (union + smooth)).item()


def pixel_accuracy(pred:   torch.Tensor,
                   target: torch.Tensor,
                   threshold: float = 0.5) -> float:
    """计算像素准确率。"""
    pred   = (torch.sigmoid(pred) > threshold).float()
    target = target.float()
    correct = (pred == target).float().sum()
    total   = pred.numel()
    return (correct / total).item()


def sensitivity(pred:   torch.Tensor,
                target: torch.Tensor,
                threshold: float = 0.5,
                smooth:    float = 1e-6) -> float:
    """计算敏感性（召回率/真阳性率）。"""
    pred   = (torch.sigmoid(pred) > threshold).float().view(-1)
    target = target.float().view(-1)
    tp     = (pred * target).sum()
    fn     = ((1 - pred) * target).sum()
    return ((tp + smooth) / (tp + fn + smooth)).item()


def specificity(pred:   torch.Tensor,
                target: torch.Tensor,
                threshold: float = 0.5,
                smooth:    float = 1e-6) -> float:
    """计算特异性（真阴性率）。"""
    pred   = (torch.sigmoid(pred) > threshold).float().view(-1)
    target = target.float().view(-1)
    tn     = ((1 - pred) * (1 - target)).sum()
    fp     = (pred * (1 - target)).sum()
    return ((tn + smooth) / (tn + fp + smooth)).item()


# 分类指标

def classification_metrics(preds: np.ndarray,
                            labels: np.ndarray) -> dict:
    """计算二分类评估指标，返回 accuracy, sensitivity, specificity, precision, f1 等。"""
    preds  = np.asarray(preds)
    labels = np.asarray(labels)

    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    total = len(labels)

    eps = 1e-8
    accuracy    = (tp + tn) / (total + eps)
    sens        = tp / (tp + fn + eps)
    spec        = tn / (tn + fp + eps)
    precision   = tp / (tp + fp + eps)
    f1          = 2 * precision * sens / (precision + sens + eps)

    return dict(
        accuracy   = accuracy,
        sensitivity= sens,
        specificity= spec,
        precision  = precision,
        f1         = f1,
        tp=tp, tn=tn, fp=fp, fn=fn,
    )


# 聚合工具（训练循环使用）

class MetricAccumulator:
    """批次级指标累积器。"""
    def __init__(self):
        self.reset()

    def reset(self):
        self._sum   = 0.0
        self._count = 0

    def update(self, val: float, n: int = 1):
        self._sum   += val * n
        self._count += n

    @property
    def avg(self) -> float:
        return self._sum / max(self._count, 1)