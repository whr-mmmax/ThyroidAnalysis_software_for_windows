import os
import argparse
import time
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import config
from models import ThyroidClassifier
from utils  import ClassificationDataset
from utils.losses  import LabelSmoothingCE
from utils.metrics import classification_metrics, MetricAccumulator


class Logger:
    """日志记录器，同时输出到终端和文件。"""
    def __init__(self, log_file="logs/cls_train.log"):
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        self.f = open(log_file, "w", encoding="utf-8")

    def info(self, msg):
        ts  = time.strftime("[%Y-%m-%d %H:%M:%S]")
        out = f"{ts} {msg}"
        print(out)
        self.f.write(out + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


def train_one_epoch(model, loader, optimizer, criterion, device):
    """训练一个 epoch，返回平均损失和评估指标。"""
    model.train()
    loss_acc = MetricAccumulator()
    all_pred, all_label = [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        loss_acc.update(loss.item(), imgs.size(0))
        all_pred .extend(logits.argmax(1).cpu().numpy().tolist())
        all_label.extend(labels.cpu().numpy().tolist())

    metrics = classification_metrics(np.array(all_pred), np.array(all_label))
    return loss_acc.avg, metrics


@torch.no_grad()
def validate(model, loader, criterion, device):
    """验证一个 epoch，返回平均损失和评估指标。"""
    model.eval()
    loss_acc = MetricAccumulator()
    all_pred, all_label = [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss_acc.update(criterion(logits, labels).item(), imgs.size(0))
        all_pred .extend(logits.argmax(1).cpu().numpy().tolist())
        all_label.extend(labels.cpu().numpy().tolist())

    metrics = classification_metrics(np.array(all_pred), np.array(all_label))
    return loss_acc.avg, metrics


def run_stage(model, train_loader, val_loader, optimizer, criterion,
              scheduler, device, n_epochs, best_acc, logger, label):
    """执行一个训练阶段（阶段1或阶段2），更新最佳准确率。"""
    for ep in range(1, n_epochs + 1):
        t0 = time.time()
        tr_loss, tr_m = train_one_epoch(model, train_loader, optimizer, criterion, device)
        va_loss, va_m = validate(model, val_loader, criterion, device)

        scheduler.step(ep - 1)
        lr = optimizer.param_groups[0]["lr"]

        logger.info(
            f"[{label}] Epoch {ep:03d}/{n_epochs}  lr={lr:.2e} | "
            f"train loss={tr_loss:.4f} acc={tr_m['accuracy']:.4f} | "
            f"val   loss={va_loss:.4f} acc={va_m['accuracy']:.4f} "
            f"sens={va_m['sensitivity']:.4f} spec={va_m['specificity']:.4f} "
            f"f1={va_m['f1']:.4f} | {time.time()-t0:.1f}s"
        )

        if va_m["accuracy"] > best_acc:
            best_acc = va_m["accuracy"]
            torch.save({
                "epoch":      ep,
                "state_dict": model.state_dict(),
                "accuracy":   best_acc,
                "metrics":    va_m,
                "backbone":   model.backbone_name,
            }, config.CLS_MODEL_PATH)
            logger.info(f"  ✓ 保存最优模型  Acc={best_acc:.4f} → {config.CLS_MODEL_PATH}")

    return best_acc


def train(args):
    logger = Logger("logs/cls_train.log")
    device = torch.device(config.DEVICE)
    use_pin = (device.type == "cuda")
    logger.info(f"使用设备: {device}")

    # 加载数据集
    logger.info("加载分类数据集...")
    full_ds = ClassificationDataset(
        root_dir=args.cls_data, img_size=config.CLS_IMG_SIZE,
        enhance_method=config.ENHANCE_METHOD)
    n_val   = max(1, int(len(full_ds) * 0.2))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42))
    logger.info(f"训练集: {n_train}   验证集: {n_val}")

    class_weights = full_ds.class_weights().to(device)
    logger.info(f"类别权重: {class_weights.cpu().numpy()}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=0, pin_memory=use_pin)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0, pin_memory=use_pin)

    # 构建模型
    model = ThyroidClassifier(
        n_classes   = 2,
        pretrained  = args.pretrain,
        backbone    = args.backbone,
        freeze_base = args.pretrain,
    ).to(device)
    logger.info(f"骨干: {args.backbone}  预训练={args.pretrain}  特征维度={model.feat_dim}")

    # 标签平滑交叉熵
    criterion = LabelSmoothingCE(
        smoothing   = config.CLS_LABEL_SMOOTH,
        num_classes = 2,
        weight      = class_weights,
    )

    # 阶段1: 冻结骨干，仅训练分类头
    stage1_epochs = 20 if args.pretrain else 0
    stage2_epochs = args.epochs - stage1_epochs
    best_acc = 0.0

    if stage1_epochs > 0:
        logger.info(f"[阶段 1] 冻结骨干，训练分类头 ({stage1_epochs} epochs)")
        optimizer1 = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr, weight_decay=config.CLS_WEIGHT_DECAY)
        sched1 = CosineAnnealingWarmRestarts(optimizer1, T_0=10, T_mult=1, eta_min=1e-7)
        best_acc = run_stage(model, train_loader, val_loader, optimizer1, criterion,
                             sched1, device, stage1_epochs, best_acc, logger, "阶段1")

    # 阶段2: 解冻全部参数微调
    model.unfreeze_all()
    stage2_lr = args.lr * 0.3
    logger.info(f"[阶段 2] 全网络微调 ({stage2_epochs} epochs)  lr={stage2_lr:.2e}")
    optimizer2 = optim.AdamW(
        model.parameters(), lr=stage2_lr,
        weight_decay=config.CLS_WEIGHT_DECAY)
    sched2 = CosineAnnealingWarmRestarts(optimizer2, T_0=20, T_mult=2, eta_min=1e-7)
    best_acc = run_stage(model, train_loader, val_loader, optimizer2, criterion,
                         sched2, device, stage2_epochs, best_acc, logger, "阶段2")

    logger.info("=" * 60)
    logger.info(f"训练完成  最优验证准确率 = {best_acc:.4f}")
    logger.close()


def parse_args():
    p = argparse.ArgumentParser(description="甲状腺结节分类训练")
    p.add_argument("--cls_data",   type=str,   default=config.CLS_DATA_ROOT)
    p.add_argument("--epochs",     type=int,   default=config.CLS_EPOCHS)
    p.add_argument("--batch_size", type=int,   default=config.CLS_BATCH_SIZE)
    p.add_argument("--lr",         type=float, default=config.CLS_LR)
    p.add_argument("--backbone",   type=str,   default=config.CLS_BACKBONE,
                   choices=["resnet18", "resnet34", "resnet50"])
    p.add_argument("--no_pretrain", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.pretrain = not args.no_pretrain
    train(args)