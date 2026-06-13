import torch
import torch.nn as nn
import torch.nn.functional as F

# 分割损失

class DiceLoss(nn.Module):
    """Dice损失，适用于分割任务。"""
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prob  = torch.sigmoid(pred)
        prob  = prob.view(-1)
        tgt   = target.view(-1).float()
        inter = (prob * tgt).sum()
        dice  = (2.0 * inter + self.smooth) / (prob.sum() + tgt.sum() + self.smooth)
        return 1.0 - dice


class BCEDiceLoss(nn.Module):
    """BCE与Dice损失的加权组合。"""
    def __init__(self, dice_weight: float = 0.6, smooth: float = 1e-6):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce  = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return ((1 - self.dice_weight) * self.bce(pred, target.float())
                + self.dice_weight * self.dice(pred, target))


class FocalLoss(nn.Module):
    """Focal损失，关注难分类样本。"""
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.float()
        bce    = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        prob_t = torch.exp(-bce)
        return (self.alpha * (1 - prob_t) ** self.gamma * bce).mean()


# 分类损失

class WeightedCELoss(nn.Module):
    """带类别权重的标准交叉熵。"""
    def __init__(self, class_weights: torch.Tensor = None):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.ce(logits, labels)


class LabelSmoothingCE(nn.Module):
    """标签平滑交叉熵，缓解过拟合，提升校准性。支持类别权重。"""
    def __init__(self,
                 smoothing:   float = 0.1,
                 num_classes: int   = 2,
                 weight: torch.Tensor = None):
        super().__init__()
        assert 0.0 <= smoothing < 1.0, "smoothing 应在 [0, 1)"
        self.smoothing    = smoothing
        self.num_classes  = num_classes
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_prob = F.log_softmax(logits, dim=-1)

        smooth_val = self.smoothing / max(self.num_classes - 1, 1)
        smooth_labels = torch.full_like(log_prob, smooth_val)
        smooth_labels.scatter_(1, targets.unsqueeze(1).long(),
                               1.0 - self.smoothing)

        per_sample_loss = -(smooth_labels * log_prob).sum(dim=-1)

        if self.weight is not None:
            w = self.weight.to(logits.device)[targets]
            per_sample_loss = per_sample_loss * w

        return per_sample_loss.mean()