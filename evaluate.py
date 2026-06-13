import os
import argparse
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

import config
from models import UNet, ThyroidClassifier
from utils  import (SegmentationDataset, ClassificationDataset,
                    dice_coefficient, iou_score, pixel_accuracy)
from utils.metrics  import sensitivity, specificity, classification_metrics, MetricAccumulator
from utils.image_enhance import enhance_image

# 工具：从 checkpoint 自动还原模型配置

def load_seg_model(ckpt_path: str, device):
    """自动读取 checkpoint 中的 attention 参数，重建 UNet。"""
    ckpt     = torch.load(ckpt_path, map_location=device)
    saved_args = ckpt.get("args", {})
    use_att  = saved_args.get("attention", False)

    model = UNet(n_channels=1, n_classes=1, bilinear=True,
                 use_attention=use_att)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    print(f"  骨干: UNet  attention={use_att}")
    print(f"  Epoch={ckpt.get('epoch','?')}  "
          f"Dice={ckpt.get('dice','?'):.4f}  "
          f"IoU={ckpt.get('iou','?'):.4f}")
    return model, ckpt


def load_cls_model(ckpt_path: str, device):
    """自动读取 checkpoint 中的骨干类型，重建分类器。"""
    ckpt     = torch.load(ckpt_path, map_location=device)
    backbone = ckpt.get("backbone", "resnet34")

    model = ThyroidClassifier(n_classes=2, pretrained=False,
                              backbone=backbone)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    print(f"  骨干: {backbone}")
    print(f"  Epoch={ckpt.get('epoch','?')}  "
          f"Acc={ckpt.get('accuracy','?'):.4f}")
    return model, ckpt


# 分割评估

@torch.no_grad()
def evaluate_segmentation(args):
    """分割模型评估。"""
    device = torch.device(config.DEVICE)
    print("\n" + "=" * 60)
    print("分割模型评估")
    print("=" * 60)

    if not os.path.exists(config.SEG_MODEL_PATH):
        print(f"[ERROR] 未找到模型: {config.SEG_MODEL_PATH}")
        return

    model, _ = load_seg_model(config.SEG_MODEL_PATH, device)

    val_ds = SegmentationDataset(
        img_dir=config.SEG_VAL_IMG_DIR, mask_dir=config.SEG_VAL_MASK_DIR,
        img_size=config.IMG_SIZE, enhance_method=config.ENHANCE_METHOD)
    loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    dice_list, iou_list, acc_list, sens_list, spec_list = [], [], [], [], []

    vis_dir = "eval_results/segmentation"
    if args.visualize:
        os.makedirs(vis_dir, exist_ok=True)

    for i, (imgs, masks) in enumerate(loader):
        imgs, masks = imgs.to(device), masks.to(device)
        preds = model(imgs)

        dice_list.append(dice_coefficient(preds, masks, config.SEG_THRESHOLD))
        iou_list .append(iou_score       (preds, masks, config.SEG_THRESHOLD))
        acc_list .append(pixel_accuracy  (preds, masks, config.SEG_THRESHOLD))
        sens_list.append(sensitivity     (preds, masks, config.SEG_THRESHOLD))
        spec_list.append(specificity     (preds, masks, config.SEG_THRESHOLD))

        if args.visualize and i < 50:
            _save_seg_grid(imgs, masks, preds, i, vis_dir, config.SEG_THRESHOLD)

    def _stat(lst):
        a = np.array(lst)
        return a.mean(), a.std()

    print(f"\n验证集共 {len(val_ds)} 张")
    for name, lst in [("Dice      ", dice_list), ("IoU       ", iou_list),
                       ("PixelAcc  ", acc_list),  ("Sensitivity", sens_list),
                       ("Specificity", spec_list)]:
        m, s = _stat(lst)
        print(f"  {name}: {m:.4f} ± {s:.4f}")

    if args.visualize:
        print(f"\n可视化图 (前 50 张) → {vis_dir}")


# 分类评估

@torch.no_grad()
def evaluate_classification(args):
    """分类模型评估。"""
    device = torch.device(config.DEVICE)
    print("\n" + "=" * 60)
    print("分类模型评估")
    print("=" * 60)

    if not os.path.exists(config.CLS_MODEL_PATH):
        print(f"[ERROR] 未找到模型: {config.CLS_MODEL_PATH}")
        return

    model, _ = load_cls_model(config.CLS_MODEL_PATH, device)

    full_ds = ClassificationDataset(
        root_dir=args.cls_data, img_size=config.CLS_IMG_SIZE,
        enhance_method=config.ENHANCE_METHOD)
    loader = DataLoader(full_ds, batch_size=8, shuffle=False, num_workers=0)

    all_pred, all_label, all_prob = [], [], []

    for imgs, labels in loader:
        imgs = imgs.to(device)
        probs = torch.softmax(model(imgs), dim=1)
        preds = probs.argmax(dim=1)
        all_pred .extend(preds.cpu().numpy().tolist())
        all_label.extend(labels.numpy().tolist())
        all_prob .extend(probs[:, 1].cpu().numpy().tolist())

    m = classification_metrics(np.array(all_pred), np.array(all_label))
    print(f"\n数据集共 {len(full_ds)} 张")
    print(f"  Accuracy    : {m['accuracy']:.4f}")
    print(f"  Sensitivity : {m['sensitivity']:.4f}  (良性召回率)")
    print(f"  Specificity : {m['specificity']:.4f}  (恶性识别率)")
    print(f"  Precision   : {m['precision']:.4f}")
    print(f"  F1 Score    : {m['f1']:.4f}")
    print(f"\n混淆矩阵 (0=恶性, 1=良性):")
    print(f"  TP={m['tp']}  FP={m['fp']}")
    print(f"  FN={m['fn']}  TN={m['tn']}")

    if args.visualize:
        _save_roc_curve(all_label, all_prob, "eval_results/roc_curve.png")
        _save_confusion_matrix(m, "eval_results/confusion_matrix.png")


# 可视化工具

def _save_seg_grid(imgs, masks, preds, idx, out_dir, threshold):
    """保存四格对比图（原图、GT、预测掩码、预测概率）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img  = imgs[0, 0].cpu().numpy()
    gt   = masks[0, 0].cpu().numpy()
    pred = (torch.sigmoid(preds[0, 0]) > threshold).float().cpu().numpy()
    prob = torch.sigmoid(preds[0, 0]).cpu().numpy()

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(img,  cmap="gray"); axes[0].set_title("Input Image")
    axes[1].imshow(gt,   cmap="gray"); axes[1].set_title("Ground Truth")
    axes[2].imshow(pred, cmap="gray"); axes[2].set_title("Predicted Mask")
    axes[3].imshow(prob, cmap="hot");  axes[3].set_title("Predicted Probability")
    for ax in axes:
        ax.axis("off")
    plt.suptitle(f"Sample {idx:04d}", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"sample_{idx:04d}.png"), dpi=100)
    plt.close()


def _save_roc_curve(labels, probs, out_path):
    """保存 ROC 曲线图。"""
    try:
        from sklearn.metrics import roc_curve, auc
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fpr, tpr, _ = roc_curve(labels, probs)
        roc_auc     = auc(fpr, tpr)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"AUC = {roc_auc:.4f}")
        plt.plot([0, 1], [0, 1], "k--", lw=1)
        plt.xlabel("False Positive Rate (1-Specificity)")
        plt.ylabel("True Positive Rate (Sensitivity)")
        plt.title("ROC Curve - Thyroid Nodule Classification")
        plt.legend(); plt.tight_layout()
        plt.savefig(out_path, dpi=120); plt.close()
        print(f"\nROC 曲线 → {out_path}  (AUC={roc_auc:.4f})")
    except ImportError:
        print("scikit-learn 未安装，跳过 ROC 曲线")


def _save_confusion_matrix(m, out_path):
    """保存混淆矩阵图。"""
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cm = np.array([[m['tn'], m['fp']], [m['fn'], m['tp']]])
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Predicted Malignant", "Predicted Benign"])
        ax.set_yticklabels(["True Malignant", "True Benign"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]),
                        ha="center", va="center", fontsize=14,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        plt.colorbar(im, ax=ax)
        ax.set_title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(out_path, dpi=120); plt.close()
        print(f"混淆矩阵 → {out_path}")
    except Exception as e:
        print(f"混淆矩阵保存失败: {e}")


# 入口

def parse_args():
    p = argparse.ArgumentParser(description="模型评估脚本")
    p.add_argument("--task",      choices=["seg", "cls", "both"], default="both")
    p.add_argument("--cls_data",  type=str, default=config.CLS_DATA_ROOT)
    p.add_argument("--visualize", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.task in ("seg",  "both"): evaluate_segmentation(args)
    if args.task in ("cls",  "both"): evaluate_classification(args)