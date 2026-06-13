import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models

# 骨干工厂

def _build_backbone(name: str, pretrained: bool):
    """构建骨干网络，返回特征提取器和特征维度。"""
    name = name.lower()
    if name == "resnet18":
        from torchvision.models import ResNet18_Weights
        w = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        base = tv_models.resnet18(weights=w)
        feat = 512
    elif name == "resnet34":
        from torchvision.models import ResNet34_Weights
        w = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        base = tv_models.resnet34(weights=w)
        feat = 512
    elif name == "resnet50":
        from torchvision.models import ResNet50_Weights
        w = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        base = tv_models.resnet50(weights=w)
        feat = 2048
    else:
        raise ValueError(f"不支持的骨干: {name}. 可选: resnet18 / resnet34 / resnet50")

    extractor = nn.Sequential(*list(base.children())[:-1])
    return extractor, feat


# 分类头（BN + GELU 风格）

class ClassifierHead(nn.Module):
    """分类头，包含全连接、BN、GELU 和 Dropout。"""
    def __init__(self, in_features: int, n_classes: int = 2):
        super().__init__()
        mid = max(in_features // 2, 256)
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, mid),
            nn.BatchNorm1d(mid),
            nn.GELU(),
            nn.Dropout(0.35),
            nn.Linear(mid, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(128, n_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


# 主模型

class ThyroidClassifier(nn.Module):
    """
    甲状腺结节分类器，支持 ResNet-18/34/50 骨干。
    参数:
        n_classes   : 分类数（默认2）
        pretrained  : 是否使用 ImageNet 预训练
        backbone    : 骨干名称
        freeze_base : 是否冻结骨干（两阶段训练第一阶段）
    """
    def __init__(self,
                 n_classes:   int  = 2,
                 pretrained:  bool = True,
                 backbone:    str  = "resnet34",
                 freeze_base: bool = False):
        super().__init__()
        self.backbone_name = backbone

        self.features, self.feat_dim = _build_backbone(backbone, pretrained)
        self.head = ClassifierHead(self.feat_dim, n_classes)

        if freeze_base:
            for p in self.features.parameters():
                p.requires_grad = False

    def unfreeze_all(self):
        """解冻全部参数。"""
        for p in self.parameters():
            p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播，返回 logits。"""
        feat = self.features(x)
        return self.head(feat)

    @torch.no_grad()
    def predict(self, x: torch.Tensor):
        """返回预测类别和概率数组。"""
        self.eval()
        logits = self.forward(x)
        probs = torch.softmax(logits, dim=1)
        pred = probs.argmax(dim=1)
        return pred, probs