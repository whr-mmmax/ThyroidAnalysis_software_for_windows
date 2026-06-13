import sys
import os
import cv2
import numpy as np
import torch

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QProgressBar, QTextEdit,
    QGroupBox, QSplitter, QStatusBar, QFrame, QSizePolicy,
    QMessageBox, QScrollArea, QToolBar, QAction, QSpacerItem,
)
from PyQt5.QtCore  import Qt, QThread, pyqtSignal, QSize, QTimer
from PyQt5.QtGui   import (QImage, QPixmap, QFont, QColor, QPalette,
                            QIcon, QPainter)

import config

# Unicode 路径安全的图像读写（修复 Windows 中文路径问题）

def _safe_imread(path: str, flags: int = cv2.IMREAD_GRAYSCALE) -> np.ndarray:
    """读取任意 Unicode 路径的图像，优先 np.fromfile + cv2.imdecode，降级 PIL。"""
    try:
        buf = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(buf, flags)
        if img is not None:
            return img
    except Exception:
        pass
    try:
        from PIL import Image
        pil = Image.open(path)
        pil = pil.convert("L" if flags == cv2.IMREAD_GRAYSCALE else "RGB")
        return np.array(pil, dtype=np.uint8)
    except Exception:
        pass
    return None


def _safe_imwrite(path: str, img: np.ndarray) -> bool:
    """保存图像到任意 Unicode 路径，优先 cv2.imencode + np.tofile，降级 PIL。"""
    try:
        ext = os.path.splitext(path)[1].lower()
        if ext not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
            ext = ".png"
        ok, buf = cv2.imencode(ext, img)
        if ok:
            buf.tofile(path)
            return True
    except Exception:
        pass
    try:
        from PIL import Image
        if len(img.shape) == 3:
            pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            pil = Image.fromarray(img)
        pil.save(path)
        return True
    except Exception:
        pass
    return False


# 推理线程（防止 GUI 卡顿）

class AnalysisWorker(QThread):
    """后台推理线程。"""
    finished  = pyqtSignal(dict)
    error     = pyqtSignal(str)
    progress  = pyqtSignal(int, str)

    def __init__(self, image_path: str, analyzer):
        super().__init__()
        self.image_path = image_path
        self.analyzer   = analyzer

    def run(self):
        try:
            self.progress.emit(20, "正在进行图像增强...")
            self.progress.emit(40, "运行分割网络 (U-Net)...")
            result = self.analyzer.analyze(self.image_path)
            self.progress.emit(80, "运行分类网络 (ResNet-18)...")
            self.progress.emit(100, "分析完成")
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


# 图像显示标签

class ImageLabel(QLabel):
    """自适应缩放图像标签。"""
    def __init__(self, title: str = "", parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(220, 220)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("""
            QLabel {
                background-color: #1a1a2e;
                border: 2px solid #16213e;
                border-radius: 8px;
                color: #8892b0;
                font-size: 13px;
            }
        """)
        self._title = title
        self._pixmap_orig = None
        self.setText(f"[ {title} ]")

    def set_cv_image(self, img_bgr: np.ndarray):
        """从 BGR numpy 数组更新图像。"""
        if img_bgr is None:
            self.setText(f"[ {self._title} ]")
            self._pixmap_orig = None
            return
        if len(img_bgr.shape) == 2:
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg  = QImage(rgb.tobytes(), w, h, ch * w, QImage.Format_RGB888)
        self._pixmap_orig = QPixmap.fromImage(qimg)
        self._refresh_pixmap()

    def _refresh_pixmap(self):
        if self._pixmap_orig is None:
            return
        scaled = self._pixmap_orig.scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(scaled)

    def resizeEvent(self, event):
        self._refresh_pixmap()
        super().resizeEvent(event)


# 主窗口

class MainWindow(QMainWindow):

    APP_TITLE   = "甲状腺超声图像智能分析系统"
    WIN_SIZE    = (1280, 800)

    def __init__(self):
        super().__init__()
        self.analyzer    = None
        self.current_img = None
        self.worker      = None
        self._init_analyzer()
        self._build_ui()
        self.setWindowTitle(self.APP_TITLE)
        self.resize(*self.WIN_SIZE)

    # 初始化分析器
    def _init_analyzer(self):
        try:
            from inference import ThyroidAnalyzer
            self.analyzer = ThyroidAnalyzer()
        except Exception as e:
            self.analyzer = None
            QTimer.singleShot(500, lambda: QMessageBox.warning(
                self, "模型加载",
                f"模型加载失败或模型文件不存在:\n{e}\n\n"
                "请先运行训练脚本生成模型:\n"
                "  python train_segmentation.py\n"
                "  python train_classification.py"))

    # UI 构建
    def _build_ui(self):
        self.setStyleSheet(STYLE_SHEET)
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(10)
        root.setContentsMargins(12, 10, 12, 10)

        root.addWidget(self._build_header())
        root.addWidget(self._build_toolbar())

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        self.progress_bar.setFixedHeight(6)
        root.addWidget(self.progress_bar)

        body = QHBoxLayout()
        body.addWidget(self._build_image_panels(), 3)
        body.addWidget(self._build_result_panel(), 1)
        root.addLayout(body, 1)

        root.addWidget(self._build_log_panel())

        self.statusBar().showMessage("就绪 — 请加载超声图像")

    def _build_header(self) -> QWidget:
        w   = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lbl = QLabel(self.APP_TITLE)
        lbl.setObjectName("headerTitle")
        sub = QLabel("Thyroid Ultrasound Nodule Detection & Classification")
        sub.setObjectName("headerSub")
        lay.addWidget(lbl)
        lay.addStretch()
        lay.addWidget(sub)
        return w

    def _build_toolbar(self) -> QWidget:
        w   = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        self.btn_open    = QPushButton("📂  打开图像")
        self.btn_analyze = QPushButton("🔬  开始分析")
        self.btn_save    = QPushButton("💾  保存结果")
        self.btn_clear   = QPushButton("🗑  清除")
        self.btn_reload  = QPushButton("🔄  重载模型")

        self.btn_analyze.setEnabled(False)
        self.btn_save   .setEnabled(False)

        self.btn_open   .clicked.connect(self.open_image)
        self.btn_analyze.clicked.connect(self.run_analysis)
        self.btn_save   .clicked.connect(self.save_result)
        self.btn_clear  .clicked.connect(self.clear_all)
        self.btn_reload .clicked.connect(self._reload_models)

        for btn in (self.btn_open, self.btn_analyze, self.btn_save,
                    self.btn_clear, self.btn_reload):
            btn.setFixedHeight(36)
            lay.addWidget(btn)

        lay.addStretch()

        dev = config.DEVICE.upper()
        lbl = QLabel(f"🖥  {dev}")
        lbl.setObjectName("deviceLabel")
        lay.addWidget(lbl)
        return w

    def _build_image_panels(self) -> QGroupBox:
        box = QGroupBox("图像分析视图")
        lay = QHBoxLayout(box)
        lay.setSpacing(8)

        self.lbl_original  = ImageLabel("原始超声图像")
        self.lbl_enhanced  = ImageLabel("增强图像")
        self.lbl_mask      = ImageLabel("结节分割掩码")
        self.lbl_overlay   = ImageLabel("叠加可视化结果")

        for lbl in (self.lbl_original, self.lbl_enhanced,
                    self.lbl_mask, self.lbl_overlay):
            lay.addWidget(lbl)

        return box

    def _build_result_panel(self) -> QGroupBox:
        box = QGroupBox("诊断结果")
        lay = QVBoxLayout(box)
        lay.setSpacing(12)

        self.lbl_nodule_title = QLabel("结节检测")
        self.lbl_nodule_title.setObjectName("resultTitle")
        self.lbl_nodule_result = QLabel("—")
        self.lbl_nodule_result.setObjectName("resultValue")
        self.lbl_nodule_result.setAlignment(Qt.AlignCenter)

        self.lbl_class_title = QLabel("良恶性判断")
        self.lbl_class_title.setObjectName("resultTitle")
        self.lbl_class_result = QLabel("—")
        self.lbl_class_result.setObjectName("resultMain")
        self.lbl_class_result.setAlignment(Qt.AlignCenter)

        self.lbl_conf_mal = QLabel("恶性 (Malignant):")
        self.bar_mal = QProgressBar(); self.bar_mal.setRange(0, 100)
        self.bar_mal.setTextVisible(True); self.bar_mal.setValue(0)
        self.bar_mal.setObjectName("barMal")

        self.lbl_conf_ben = QLabel("良性 (Benign):")
        self.bar_ben = QProgressBar(); self.bar_ben.setRange(0, 100)
        self.bar_ben.setTextVisible(True); self.bar_ben.setValue(0)
        self.bar_ben.setObjectName("barBen")

        sep = QFrame(); sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color:#334;")

        lay.addWidget(self.lbl_nodule_title)
        lay.addWidget(self.lbl_nodule_result)
        lay.addWidget(sep)
        lay.addWidget(self.lbl_class_title)
        lay.addWidget(self.lbl_class_result)
        lay.addSpacing(8)
        lay.addWidget(self.lbl_conf_mal)
        lay.addWidget(self.bar_mal)
        lay.addSpacing(4)
        lay.addWidget(self.lbl_conf_ben)
        lay.addWidget(self.bar_ben)
        lay.addStretch()

        return box

    def _build_log_panel(self) -> QGroupBox:
        box = QGroupBox("分析日志")
        lay = QVBoxLayout(box)
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumHeight(120)
        self.log_box.setObjectName("logBox")
        lay.addWidget(self.log_box)
        return box

    # 操作槽
    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择超声图像", "",
            "图像文件 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)")
        if not path:
            return
        self.current_img = path
        raw = _safe_imread(path, cv2.IMREAD_GRAYSCALE)
        if raw is None:
            QMessageBox.critical(
                self, "图像读取失败",
                f"无法读取图像文件:\n{path}\n\n"
                "可能原因:\n"
                "  • 文件已损坏或格式不受支持\n"
                "  • 文件被其他程序占用\n"
                "  • 权限不足")
            self._log(f"❌ 无法读取: {os.path.basename(path)}")
            self.current_img = None
            return
        self.lbl_original.set_cv_image(raw)
        self.lbl_enhanced.set_cv_image(None)
        self.lbl_mask    .set_cv_image(None)
        self.lbl_overlay .set_cv_image(None)
        self._reset_result_panel()
        self.btn_analyze.setEnabled(True)
        self.btn_save   .setEnabled(False)
        self._log(f"✅ 已加载图像: {os.path.basename(path)}")
        self.statusBar().showMessage(f"已加载: {path}")

    def run_analysis(self):
        if not self.current_img:
            return
        if self.analyzer is None:
            QMessageBox.warning(self, "模型未就绪",
                                "请先训练并保存模型文件,\n"
                                "然后点击「重载模型」。")
            return

        self._log("🔬 开始分析...")
        self.btn_analyze.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        self.worker = AnalysisWorker(self.current_img, self.analyzer)
        self.worker.finished.connect(self._on_analysis_done)
        self.worker.error   .connect(self._on_analysis_error)
        self.worker.progress.connect(self._on_progress)
        self.worker.start()

    def _on_progress(self, val: int, msg: str):
        self.progress_bar.setValue(val)
        self.statusBar().showMessage(msg)

    def _on_analysis_done(self, result: dict):
        self.progress_bar.setValue(100)
        QTimer.singleShot(800, lambda: self.progress_bar.setVisible(False))
        self.btn_analyze.setEnabled(True)
        self.btn_save   .setEnabled(True)
        self._result = result

        self.lbl_enhanced.set_cv_image(result["enhanced"])
        if result["mask"] is not None:
            self.lbl_mask.set_cv_image(result["mask"])
        self.lbl_overlay.set_cv_image(result["overlay"])

        nod_txt = "✅ 检测到结节" if result["has_nodule"] else "⭕ 未检测到结节"
        self.lbl_nodule_result.setText(nod_txt)
        self.lbl_nodule_result.setStyleSheet(
            "color: #e76f51;" if result["has_nodule"] else "color: #2a9d8f;")

        if result["probs"] is not None:
            cid   = result["class_id"]
            probs = result["probs"]
            self.lbl_class_result.setText(result["class_name"])
            if cid == 0:
                self.lbl_class_result.setStyleSheet(
                    "color:#e63946; font-size:20px; font-weight:bold;")
            else:
                self.lbl_class_result.setStyleSheet(
                    "color:#2a9d8f; font-size:20px; font-weight:bold;")
            self.bar_mal.setValue(int(probs[0] * 100))
            self.bar_ben.setValue(int(probs[1] * 100))
            self._log(f"📊 分类结果: {result['class_name']}  "
                      f"P(恶)={probs[0]*100:.1f}%  P(良)={probs[1]*100:.1f}%")
        else:
            self.lbl_class_result.setText("分类模型未加载")
            self._log("⚠️ 分类模型未加载")

        self.statusBar().showMessage("分析完成")
        self._log("✅ 分析完成")

    def _on_analysis_error(self, msg: str):
        self.progress_bar.setVisible(False)
        self.btn_analyze.setEnabled(True)
        self._log(f"❌ 分析出错: {msg}")
        QMessageBox.critical(self, "分析错误", msg)

    def save_result(self):
        if not hasattr(self, "_result"):
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "保存叠加结果", "thyroid_result.png",
            "PNG 图像 (*.png);;JPEG 图像 (*.jpg)")
        if not path:
            return
        ok = _safe_imwrite(path, self._result["overlay"])
        if ok:
            self._log(f"💾 结果已保存: {path}")
            self.statusBar().showMessage(f"已保存: {path}")
        else:
            QMessageBox.critical(
                self, "保存失败",
                f"无法保存到:\n{path}\n\n"
                "可能原因:\n"
                "  • 目标目录不存在或权限不足\n"
                "  • 磁盘空间不足")
            self._log(f"❌ 保存失败: {path}")

    def clear_all(self):
        for lbl in (self.lbl_original, self.lbl_enhanced,
                    self.lbl_mask, self.lbl_overlay):
            lbl.set_cv_image(None)
        self._reset_result_panel()
        self.log_box.clear()
        self.current_img = None
        self.btn_analyze.setEnabled(False)
        self.btn_save   .setEnabled(False)
        self.statusBar().showMessage("就绪")

    def _reload_models(self):
        self._log("🔄 重载模型中...")
        self._init_analyzer()
        if self.analyzer:
            self._log("✅ 模型重载成功")
        else:
            self._log("❌ 模型重载失败，请检查 checkpoints/ 目录")

    def _reset_result_panel(self):
        self.lbl_nodule_result.setText("—")
        self.lbl_nodule_result.setStyleSheet("color: #8892b0;")
        self.lbl_class_result .setText("—")
        self.lbl_class_result .setStyleSheet("color: #8892b0; font-size:18px;")
        self.bar_mal.setValue(0)
        self.bar_ben.setValue(0)

    def _log(self, msg: str):
        import time
        ts = time.strftime("%H:%M:%S")
        self.log_box.append(f"[{ts}] {msg}")
        self.log_box.verticalScrollBar().setValue(
            self.log_box.verticalScrollBar().maximum())


# 样式表

STYLE_SHEET = """
QMainWindow, QWidget {
    background-color: #0d1117;
    color: #c9d1d9;
    font-family: "Microsoft YaHei", "Segoe UI", sans-serif;
    font-size: 13px;
}
QGroupBox {
    border: 1px solid #21262d;
    border-radius: 8px;
    margin-top: 10px;
    padding-top: 12px;
    background-color: #161b22;
    color: #8b949e;
    font-weight: bold;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    color: #58a6ff;
}
QPushButton {
    background-color: #21262d;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 4px 14px;
    color: #c9d1d9;
    min-width: 90px;
}
QPushButton:hover  { background-color: #30363d; border-color:#58a6ff; }
QPushButton:pressed{ background-color: #0d1117; }
QPushButton:disabled{ color:#484f58; }
QProgressBar {
    border: none;
    background-color: #21262d;
    border-radius: 3px;
}
QProgressBar::chunk {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                  stop:0 #1f6feb, stop:1 #388bfd);
    border-radius: 3px;
}
QProgressBar#barMal::chunk { background-color: #cf222e; }
QProgressBar#barBen::chunk { background-color: #1a7f37; }
QTextEdit#logBox {
    background-color: #0d1117;
    border: 1px solid #21262d;
    border-radius: 6px;
    color: #7ee787;
    font-family: Consolas, monospace;
    font-size: 12px;
}
QLabel#headerTitle {
    font-size: 18px;
    font-weight: bold;
    color: #58a6ff;
}
QLabel#headerSub {
    font-size: 12px;
    color: #484f58;
}
QLabel#resultTitle {
    color: #8b949e;
    font-size: 12px;
}
QLabel#resultValue {
    color: #c9d1d9;
    font-size: 14px;
    font-weight: bold;
}
QLabel#resultMain {
    color: #58a6ff;
    font-size: 18px;
    font-weight: bold;
}
QLabel#deviceLabel {
    background-color: #1f6feb22;
    border: 1px solid #1f6feb44;
    border-radius: 4px;
    padding: 2px 8px;
    color: #388bfd;
    font-size: 12px;
}
QScrollBar:vertical {
    background:#0d1117; width:8px; border-radius:4px;
}
QScrollBar::handle:vertical {
    background:#30363d; border-radius:4px; min-height:20px;
}
QStatusBar { color: #484f58; font-size: 12px; }
"""


# 入口

def main():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps,    True)

    app = QApplication(sys.argv)
    app.setApplicationName("甲状腺超声分析系统")

    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()