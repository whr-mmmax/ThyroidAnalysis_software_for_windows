import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from utils.image_enhance import enhance_image

# 跨平台安全读图函数

def _safe_imread(path: str, flags: int = cv2.IMREAD_GRAYSCALE) -> np.ndarray:
    """
    Windows 兼容的图像读取，解决 cv2.imread() 在某些特殊 PNG 格式下返回 None 的问题。
    优先使用 np.fromfile + cv2.imdecode，失败时用 PIL 降级读取。
    """
    # 字节级读取 + imdecode
    try:
        buf = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(buf, flags)
        if img is not None:
            return img
    except Exception:
        pass

    # PIL 降级
    try:
        pil_img = Image.open(path)
        if flags == cv2.IMREAD_GRAYSCALE:
            pil_img = pil_img.convert("L")
        elif flags == cv2.IMREAD_COLOR:
            pil_img = pil_img.convert("RGB")
        return np.array(pil_img, dtype=np.uint8)
    except Exception:
        pass

    return None

# 分割数据集

class SegmentationDataset(Dataset):
    """
    分割数据集，目录结构：
        root/
          img/       原图
          labelcol/  掩码（非零为结节区域）
    返回 (image, mask) 均为 (1, H, W) 的 FloatTensor，image 归一化至 [0,1]，mask 二值 {0,1}
    """

    IMG_EXTENSIONS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]

    def __init__(self,
                 img_dir:        str,
                 mask_dir:       str,
                 img_size:       int = 256,
                 enhance_method: str = "combined",
                 mask_threshold: int = 10):
        self.img_dir        = img_dir
        self.mask_dir       = mask_dir
        self.img_size       = img_size
        self.enhance_method = enhance_method
        self.mask_threshold = mask_threshold

        self.img_files = sorted([
            f for f in os.listdir(img_dir)
            if os.path.splitext(f)[1].lower() in self.IMG_EXTENSIONS
        ])

        if len(self.img_files) == 0:
            raise RuntimeError(f"在 {img_dir} 中未找到图像文件")

        # 构建图像到掩码的映射
        self.mask_files = []
        mask_dir_files  = os.listdir(mask_dir)

        for imgf in self.img_files:
            stem  = os.path.splitext(imgf)[0]
            found = None

            # 优先同名
            for ext in self.IMG_EXTENSIONS:
                candidate = os.path.join(mask_dir, stem + ext)
                if os.path.exists(candidate):
                    found = stem + ext
                    break

            # 大小写不敏感兜底
            if found is None:
                for f in mask_dir_files:
                    if os.path.splitext(f)[0].lower() == stem.lower():
                        found = f
                        break

            if found is None:
                raise FileNotFoundError(f"找不到图像 {imgf} 对应的掩码文件 (在 {mask_dir})")

            self.mask_files.append(found)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path  = os.path.join(self.img_dir,  self.img_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        # 读取图像和掩码
        img  = _safe_imread(img_path,  cv2.IMREAD_GRAYSCALE)
        mask = _safe_imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise IOError(f"无法读取图像: {img_path}")
        if mask is None:
            raise IOError(f"无法读取掩码: {mask_path}")

        # 图像增强
        img = enhance_image(img, method=self.enhance_method)

        # 缩放到统一尺寸
        img  = cv2.resize(img,  (self.img_size, self.img_size),
                          interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.img_size, self.img_size),
                          interpolation=cv2.INTER_NEAREST)

        # 掩码二值化
        binary_mask = (mask > self.mask_threshold).astype(np.float32)

        # 转为 Tensor
        img_t  = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)
        mask_t = torch.from_numpy(binary_mask).unsqueeze(0)

        return img_t, mask_t

    @property
    def name_list(self):
        return self.img_files

# 分类数据集

class ClassificationDataset(Dataset):
    """
    分类数据集，目录结构：
        root/
          0/   恶性
          1/   良性
    返回 (image, label)，image 为 (3, H, W) 经 ImageNet 归一化，label 为 0/1 长整型
    """

    IMG_EXTENSIONS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]

    def __init__(self,
                 root_dir:       str,
                 img_size:       int  = 224,
                 enhance_method: str  = "combined",
                 class_dirs:     list = None):
        self.root_dir       = root_dir
        self.img_size       = img_size
        self.enhance_method = enhance_method
        class_dirs          = class_dirs or ["0", "1"]

        self.samples = []
        for label, cls_name in enumerate(class_dirs):
            cls_path = os.path.join(root_dir, cls_name)
            if not os.path.isdir(cls_path):
                raise FileNotFoundError(f"类别文件夹不存在: {cls_path}")
            for fname in os.listdir(cls_path):
                if os.path.splitext(fname)[1].lower() in self.IMG_EXTENSIONS:
                    self.samples.append((os.path.join(cls_path, fname), label))

        if len(self.samples) == 0:
            raise RuntimeError(f"在 {root_dir} 中未找到任何图像")

        self.normalize = transforms.Normalize(mean=self.MEAN, std=self.STD)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        img = _safe_imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise IOError(f"无法读取图像: {img_path}")

        img = enhance_image(img, method=self.enhance_method)
        img = cv2.resize(img, (self.img_size, self.img_size),
                         interpolation=cv2.INTER_LINEAR)

        # 单通道复制为三通道
        img_rgb = np.stack([img, img, img], axis=2)
        img_t   = torch.from_numpy(img_rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
        img_t   = self.normalize(img_t)

        return img_t, torch.tensor(label, dtype=torch.long)

    def class_weights(self) -> torch.Tensor:
        """计算类别平衡权重。"""
        counts = [0, 0]
        for _, lbl in self.samples:
            counts[lbl] += 1
        total = sum(counts)
        weights = [total / (2 * c) for c in counts]
        return torch.tensor(weights, dtype=torch.float32)