import os
import torch

# 数据集路径（请按实际情况修改）
DATA_ROOT          = "./data"

SEG_TRAIN_IMG_DIR  = os.path.join(DATA_ROOT, "Train_Folder", "img")
SEG_TRAIN_MASK_DIR = os.path.join(DATA_ROOT, "Train_Folder", "labelcol")
SEG_VAL_IMG_DIR    = os.path.join(DATA_ROOT, "Val_Folder",   "img")
SEG_VAL_MASK_DIR   = os.path.join(DATA_ROOT, "Val_Folder",   "labelcol")

CLS_DATA_ROOT      = os.path.join(DATA_ROOT, "classification")

# 模型保存路径
MODEL_SAVE_DIR     = "checkpoints"
SEG_MODEL_PATH     = os.path.join(MODEL_SAVE_DIR, "best_unet.pth")
CLS_MODEL_PATH     = os.path.join(MODEL_SAVE_DIR, "best_classifier.pth")
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# 图像尺寸
IMG_SIZE     = 256
CLS_IMG_SIZE = 224

# 图像增强参数
ENHANCE_METHOD      = "combined"
CLAHE_CLIP_LIMIT    = 2.0
CLAHE_TILE_GRID     = (8, 8)
GAMMA_VALUE         = 1.25
UNSHARP_SIGMA       = 1.0
UNSHARP_STRENGTH    = 1.5

# 分割训练超参数
SEG_BATCH_SIZE     = 4
SEG_EPOCHS         = 60
SEG_LR             = 1e-4
SEG_WEIGHT_DECAY   = 1e-5
SEG_DICE_WEIGHT    = 0.6
SEG_THRESHOLD      = 0.5

# 分类训练超参数
CLS_BACKBONE       = "resnet34"    # 可选: resnet18 / resnet34 / resnet50
CLS_BATCH_SIZE     = 8
CLS_EPOCHS         = 80            # 总轮数（阶段1=20，阶段2=60）
CLS_LR             = 1e-4
CLS_WEIGHT_DECAY   = 1e-4
CLS_PRETRAINED     = True
CLS_LABEL_SMOOTH   = 0.1           # 标签平滑系数

# 设备
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 类别标签
CLASS_NAMES = {0: "恶性 (Malignant)", 1: "良性 (Benign)"}