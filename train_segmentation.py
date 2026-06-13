import os
import argparse
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

import config
from models import UNet
from utils  import SegmentationDataset, BCEDiceLoss, dice_coefficient, iou_score
from utils.metrics import MetricAccumulator


class Logger:
    """日志记录器，同时输出到终端和文件。"""
    def __init__(self, log_file: str = "logs/seg_train.log"):
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        self.f = open(log_file, "w", encoding="utf-8")

    def info(self, msg: str):
        ts  = time.strftime("[%Y-%m-%d %H:%M:%S]")
        out = f"{ts} {msg}"
        print(out)
        self.f.write(out + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


def train_one_epoch(model, loader, optimizer, criterion, device):
    """训练一个 epoch，返回平均损失和平均 Dice。"""
    model.train()
    loss_acc = MetricAccumulator()
    dice_acc = MetricAccumulator()

    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        preds = model(imgs)
        loss  = criterion(preds, masks)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        bs = imgs.size(0)
        with torch.no_grad():
            d = dice_coefficient(preds.detach(), masks.detach(),
                                 threshold=config.SEG_THRESHOLD)
        loss_acc.update(loss.item(), bs)
        dice_acc.update(d, bs)

    return loss_acc.avg, dice_acc.avg


@torch.no_grad()
def validate(model, loader, criterion, device):
    """验证一个 epoch，返回平均损失、平均 Dice 和平均 IoU。"""
    model.eval()
    loss_acc = MetricAccumulator()
    dice_acc = MetricAccumulator()
    iou_acc  = MetricAccumulator()

    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        preds = model(imgs)
        loss  = criterion(preds, masks)
        bs    = imgs.size(0)
        loss_acc.update(loss.item(), bs)
        dice_acc.update(dice_coefficient(preds, masks, config.SEG_THRESHOLD), bs)
        iou_acc .update(iou_score(preds, masks, config.SEG_THRESHOLD), bs)

    return loss_acc.avg, dice_acc.avg, iou_acc.avg


def train(args):
    logger = Logger("logs/seg_train.log")
    device = torch.device(config.DEVICE)
    use_pin = (device.type == "cuda")
    logger.info(f"使用设备: {device}")

    # 加载数据集
    logger.info("加载分割数据集...")
    train_ds = SegmentationDataset(
        img_dir=config.SEG_TRAIN_IMG_DIR, mask_dir=config.SEG_TRAIN_MASK_DIR,
        img_size=config.IMG_SIZE, enhance_method=config.ENHANCE_METHOD)
    val_ds = SegmentationDataset(
        img_dir=config.SEG_VAL_IMG_DIR, mask_dir=config.SEG_VAL_MASK_DIR,
        img_size=config.IMG_SIZE, enhance_method=config.ENHANCE_METHOD)
    logger.info(f"训练集: {len(train_ds)}  验证集: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=0, pin_memory=use_pin)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0, pin_memory=use_pin)

    # 构建模型
    model = UNet(n_channels=1, n_classes=1, bilinear=True,
                 use_attention=args.attention).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"参数量: {total_params:,}   Attention: {args.attention}")

    # 损失、优化器、调度器
    criterion = BCEDiceLoss(dice_weight=config.SEG_DICE_WEIGHT)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=config.SEG_WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_dice = 0.0
    logger.info("=" * 60)
    logger.info("开始训练 (无早停，训练完整 epochs)")
    logger.info("=" * 60)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_dice = train_one_epoch(model, train_loader, optimizer, criterion, device)
        va_loss, va_dice, va_iou = validate(model, val_loader, criterion, device)
        scheduler.step()

        logger.info(
            f"Epoch [{epoch:03d}/{args.epochs}] "
            f"lr={optimizer.param_groups[0]['lr']:.2e} | "
            f"train loss={tr_loss:.4f} dice={tr_dice:.4f} | "
            f"val   loss={va_loss:.4f} dice={va_dice:.4f} iou={va_iou:.4f} | "
            f"time={time.time()-t0:.1f}s"
        )

        if va_dice > best_dice:
            best_dice = va_dice
            torch.save({
                "epoch": epoch, "state_dict": model.state_dict(),
                "dice": best_dice, "iou": va_iou,
                "args": vars(args),
            }, config.SEG_MODEL_PATH)
            logger.info(f"  ✓ 保存最优模型  Dice={best_dice:.4f}")

    logger.info("=" * 60)
    logger.info(f"训练完成  最优 Dice = {best_dice:.4f}")
    logger.close()


def parse_args():
    p = argparse.ArgumentParser(description="甲状腺结节分割训练")
    p.add_argument("--epochs",     type=int,   default=config.SEG_EPOCHS)
    p.add_argument("--batch_size", type=int,   default=config.SEG_BATCH_SIZE)
    p.add_argument("--lr",         type=float, default=config.SEG_LR)
    p.add_argument("--attention",  action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())