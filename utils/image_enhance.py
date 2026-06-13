import cv2
import numpy as np

# 单项增强函数

def apply_clahe(image: np.ndarray,
                clip_limit:    float = 2.0,
                tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """自适应直方图均衡化，提升局部对比度，抑制过度增强。"""
    image = _ensure_uint8_gray(image)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)


def apply_histogram_eq(image: np.ndarray) -> np.ndarray:
    """全局直方图均衡化。"""
    image = _ensure_uint8_gray(image)
    return cv2.equalizeHist(image)


def apply_gamma(image: np.ndarray, gamma: float = 1.25) -> np.ndarray:
    """Gamma 校正，调整整体亮度。"""
    image = _ensure_uint8_gray(image)
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(256)], dtype=np.uint8)
    return cv2.LUT(image, table)


def apply_unsharp(image: np.ndarray,
                  sigma:    float = 1.0,
                  strength: float = 1.5) -> np.ndarray:
    """非锐化掩模，增强边界高频细节。"""
    image  = _ensure_uint8_gray(image)
    ksize  = _sigma_to_ksize(sigma)
    blurred = cv2.GaussianBlur(image.astype(np.float32), ksize, sigma)
    sharp  = image.astype(np.float32) + strength * (image.astype(np.float32) - blurred)
    return np.clip(sharp, 0, 255).astype(np.uint8)


def apply_bilateral(image: np.ndarray,
                    d: int = 9,
                    sigma_color: float = 75.0,
                    sigma_space: float = 75.0) -> np.ndarray:
    """双边滤波，保边去噪。"""
    image = _ensure_uint8_gray(image)
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


# 组合增强流水线

def enhance_image(image: np.ndarray,
                  method: str = "combined",
                  clahe_clip:    float = 2.0,
                  clahe_tile:    tuple = (8, 8),
                  gamma:         float = 1.25,
                  unsharp_sigma: float = 1.0,
                  unsharp_str:   float = 1.5) -> np.ndarray:
    """
    主增强入口。
    method: 'clahe', 'gamma', 'unsharp', 'bilateral', 'hist_eq', 'combined'(推荐)
    """
    image = _ensure_uint8_gray(image)

    if method == "clahe":
        return apply_clahe(image, clahe_clip, clahe_tile)
    elif method == "gamma":
        return apply_gamma(image, gamma)
    elif method == "unsharp":
        return apply_unsharp(image, unsharp_sigma, unsharp_str)
    elif method == "bilateral":
        return apply_bilateral(image)
    elif method == "hist_eq":
        return apply_histogram_eq(image)
    elif method == "combined":
        out = apply_clahe(image, clahe_clip, clahe_tile)
        out = apply_gamma(out, gamma)
        out = apply_unsharp(out, unsharp_sigma, unsharp_str)
        return out
    else:
        raise ValueError(f"未知增强方法: {method}")


def enhance_for_display(image: np.ndarray) -> np.ndarray:
    """用于 GUI 可视化的快速增强，返回 RGB 图像。"""
    gray = _ensure_uint8_gray(image)
    enhanced = enhance_image(gray, method="combined")
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)


# 内部工具

def _ensure_uint8_gray(image: np.ndarray) -> np.ndarray:
    """确保输入为 uint8 灰度图。"""
    if image is None:
        raise ValueError("输入图像为 None")
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.dtype != np.uint8:
        image = ((image - image.min()) /
                 (image.max() - image.min() + 1e-8) * 255).astype(np.uint8)
    return image


def _sigma_to_ksize(sigma: float) -> tuple:
    """由 sigma 推导高斯核尺寸（奇数）。"""
    k = int(6 * sigma + 1)
    return (k | 1, k | 1)