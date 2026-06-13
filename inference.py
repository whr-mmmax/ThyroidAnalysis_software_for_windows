import os
import argparse
import cv2
import numpy as np
import torch
import torchvision.transforms as T

import config
from models import UNet, ThyroidClassifier
from utils.image_enhance import enhance_image


class ThyroidAnalyzer:
    """封装完整推理流水线。"""

    _MEAN = [0.485, 0.456, 0.406]
    _STD  = [0.229, 0.224, 0.225]

    _TTE_METHODS = ["combined", "clahe", "gamma"]   # 测试时增强的方法

    def __init__(self, seg_model_path=None, cls_model_path=None,
                 device=None, enable_tte=True):
        self.device     = torch.device(device or config.DEVICE)
        self.seg_model  = None
        self.cls_model  = None
        self.enable_tte = enable_tte

        self._load_seg_model(seg_model_path or config.SEG_MODEL_PATH)
        self._load_cls_model(cls_model_path or config.CLS_MODEL_PATH)
        self._normalize = T.Normalize(mean=self._MEAN, std=self._STD)

    # 模型加载（自动感知 attention / backbone）
    def _load_seg_model(self, path):
        if not os.path.exists(path):
            print(f"[Warning] 分割模型未找到: {path}")
            return
        ckpt    = torch.load(path, map_location=self.device)
        use_att = ckpt.get("args", {}).get("attention", False)
        model   = UNet(n_channels=1, n_classes=1, bilinear=True,
                       use_attention=use_att)
        model.load_state_dict(ckpt["state_dict"])
        model.to(self.device).eval()
        self.seg_model = model
        print(f"[OK] 分割模型  attention={use_att}  "
              f"Dice={ckpt.get('dice','?'):.4f}")

    def _load_cls_model(self, path):
        if not os.path.exists(path):
            print(f"[Warning] 分类模型未找到: {path}")
            return
        ckpt     = torch.load(path, map_location=self.device)
        backbone = ckpt.get("backbone", "resnet34")
        model    = ThyroidClassifier(n_classes=2, pretrained=False,
                                     backbone=backbone)
        model.load_state_dict(ckpt["state_dict"])
        model.to(self.device).eval()
        self.cls_model = model
        print(f"[OK] 分类模型  backbone={backbone}  "
              f"Acc={ckpt.get('accuracy','?'):.4f}")

    @property
    def seg_ready(self): return self.seg_model is not None
    @property
    def cls_ready(self): return self.cls_model is not None

    # 主推理接口
    @torch.no_grad()
    def analyze(self, image_input, threshold=None):
        """输入图像路径或 numpy 数组，返回分析结果字典。"""
        threshold = threshold or config.SEG_THRESHOLD
        raw_gray  = self._load_gray(image_input)
        h0, w0    = raw_gray.shape[:2]

        # 图像增强
        enhanced = enhance_image(raw_gray, method=config.ENHANCE_METHOD,
                                 clahe_clip=config.CLAHE_CLIP_LIMIT,
                                 clahe_tile=config.CLAHE_TILE_GRID,
                                 gamma=config.GAMMA_VALUE,
                                 unsharp_sigma=config.UNSHARP_SIGMA,
                                 unsharp_str=config.UNSHARP_STRENGTH)

        # 分割
        mask, has_nodule = None, False
        if self.seg_ready:
            mask, has_nodule = self._segment(enhanced, threshold)
            mask = cv2.resize(mask, (w0, h0), interpolation=cv2.INTER_NEAREST)

        # 叠加可视化
        overlay = self._make_overlay(enhanced, mask)

        # 分类（可选 TTE）
        class_id, class_name, probs = -1, "未加载分类模型", None
        if self.cls_ready:
            if self.enable_tte:
                class_id, class_name, probs = self._classify_tte(raw_gray)
            else:
                class_id, class_name, probs = self._classify_single(enhanced)

        return dict(enhanced=enhanced, mask=mask, overlay=overlay,
                    class_id=class_id, class_name=class_name,
                    probs=probs, has_nodule=has_nodule)

    # 分割
    def _segment(self, gray, threshold):
        """预测分割掩码并返回二值掩码及是否有结节的标志。"""
        resized = cv2.resize(gray, (config.IMG_SIZE, config.IMG_SIZE),
                             interpolation=cv2.INTER_LINEAR)
        t = torch.from_numpy(resized.astype(np.float32) / 255.0
                             ).unsqueeze(0).unsqueeze(0).to(self.device)
        prob   = torch.sigmoid(self.seg_model(t))[0, 0].cpu().numpy()
        binary = (prob > threshold).astype(np.uint8) * 255
        has    = binary.max() > 0
        if has:
            binary = self._morpho_clean(binary)
            has    = binary.max() > 0
        return binary, has

    @staticmethod
    def _morpho_clean(mask):
        """形态学后处理，去除小面积噪点。"""
        min_area = mask.shape[0] * mask.shape[1] * 0.005
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        clean = np.zeros_like(mask)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                clean[labels == i] = 255
        return clean

    # 分类：单次推理
    def _classify_single(self, gray):
        """单次分类推理。"""
        t      = self._gray_to_tensor(gray, config.CLS_IMG_SIZE)
        logits = self.cls_model(t)
        probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()
        cid    = int(probs.argmax())
        return cid, config.CLASS_NAMES[cid], probs

    # 分类：TTE（测试时增强集成）
    def _classify_tte(self, raw_gray):
        """使用多种增强方式推理，概率平均。"""
        prob_sum = None
        for method in self._TTE_METHODS:
            enh  = enhance_image(raw_gray, method=method)
            t    = self._gray_to_tensor(enh, config.CLS_IMG_SIZE)
            p    = torch.softmax(self.cls_model(t), dim=1)[0].cpu().numpy()
            prob_sum = p if prob_sum is None else prob_sum + p

        probs = prob_sum / len(self._TTE_METHODS)
        cid   = int(probs.argmax())
        return cid, config.CLASS_NAMES[cid], probs

    # 工具函数
    def _gray_to_tensor(self, gray, size):
        """灰度图转为归一化的 RGB Tensor。"""
        resized = cv2.resize(gray, (size, size), interpolation=cv2.INTER_LINEAR)
        rgb     = np.stack([resized, resized, resized], axis=2)
        t       = torch.from_numpy(rgb.astype(np.float32) / 255.0
                                   ).permute(2, 0, 1)
        return self._normalize(t).unsqueeze(0).to(self.device)

    @staticmethod
    def _load_gray(image_input):
        """加载图像为灰度图，支持路径或 numpy 数组。"""
        if isinstance(image_input, str):
            from utils.dataset import _safe_imread
            img = _safe_imread(image_input, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise IOError(f"无法读取: {image_input}")
            return img
        elif isinstance(image_input, np.ndarray):
            if len(image_input.shape) == 3:
                return cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY)
            return image_input
        raise TypeError(f"不支持的输入类型: {type(image_input)}")

    @staticmethod
    def _make_overlay(gray, mask=None):
        """生成分割结果叠加图（绿色轮廓 + 半透明红色区域）。"""
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        if mask is None or mask.max() == 0:
            return bgr
        m = cv2.resize(mask, (bgr.shape[1], bgr.shape[0]),
                       interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(bgr, contours, -1, (0, 255, 0), 2)
        oc = bgr.copy()
        oc[m > 0] = (0, 0, 180)
        return cv2.addWeighted(bgr, 0.7, oc, 0.3, 0)


# 命令行运行

def main():
    p = argparse.ArgumentParser(description="甲状腺超声图像分析")
    p.add_argument("--image",     required=True)
    p.add_argument("--output",    default=None)
    p.add_argument("--seg_model", default=None)
    p.add_argument("--cls_model", default=None)
    p.add_argument("--tte",       action="store_true", help="启用测试时增强集成")
    args = p.parse_args()

    analyzer = ThyroidAnalyzer(args.seg_model, args.cls_model,
                               enable_tte=args.tte)
    result   = analyzer.analyze(args.image)

    print("\n" + "=" * 50)
    print("分析结果")
    print("=" * 50)
    print(f"结节检测: {'检测到' if result['has_nodule'] else '未检测到'}")
    if result["probs"] is not None:
        mode = "(TTE集成)" if args.tte else ""
        print(f"分类结果 {mode}: {result['class_name']}")
        print(f"  P(恶性) = {result['probs'][0]*100:.2f}%")
        print(f"  P(良性) = {result['probs'][1]*100:.2f}%")

    if args.output:
        cv2.imwrite(args.output, result["overlay"])
        print(f"\n叠加图 → {args.output}")
    else:
        cv2.imshow("Thyroid Analysis", result["overlay"])
        cv2.waitKey(0); cv2.destroyAllWindows()


if __name__ == "__main__":
    main()