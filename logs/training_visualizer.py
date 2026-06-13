import re
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator

# 字体设置（优先中文，回退英文）
matplotlib.rcParams['axes.unicode_minus'] = False
for _font in ('Microsoft YaHei', 'SimHei', 'WenQuanYi Micro Hei', 'Arial Unicode MS'):
    try:
        matplotlib.font_manager.findfont(_font, fallback_to_default=False)
        matplotlib.rcParams['font.family'] = _font
        break
    except Exception:
        pass

# 全局样式
plt.rcParams.update({
    'axes.grid':         True,
    'grid.alpha':        0.3,
    'grid.linestyle':    '--',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'figure.dpi':        150,
    'savefig.dpi':       150,
    'figure.facecolor':  'white',
    'axes.facecolor':    '#F8FAFC',
    'xtick.labelsize':   8,
    'ytick.labelsize':   8,
})

# 颜色方案
C = dict(
    train  = '#2563EB',   # 训练
    val    = '#DC2626',   # 验证
    sens   = '#16A34A',   # 敏感度
    spec   = '#7C3AED',   # 特异度
    f1     = '#EA580C',   # F1
    iou    = '#0891B2',   # IoU
    lr     = '#64748B',   # 学习率
    best   = '#F59E0B',   # 最优点标记
    ph1_bg = '#DBEAFE',   # Phase1 背景
    ph2_bg = '#FEF3C7',   # Phase2 背景
)


def parse_cls_log(path: str) -> dict:
    """解析分类日志（两阶段：冻结骨干20ep + 全网络微调60ep）。"""
    EP_RE = re.compile(
        r'\[阶段(\d)\]\s+Epoch\s+(\d+)/\d+\s+'
        r'lr=([0-9.e+\-]+)\s*\|\s*'
        r'train\s+loss=([0-9.]+)\s+acc=([0-9.]+)\s*\|\s*'
        r'val\s+loss=([0-9.]+)\s+acc=([0-9.]+)\s+'
        r'sens=([0-9.]+)\s+spec=([0-9.]+)\s+f1=([0-9.]+)'
    )
    CKPT_RE = re.compile(r'保存最优模型')

    rows, best_eps, last_gep = [], [], None

    with open(path, encoding='utf-8') as f:
        for line in f:
            m = EP_RE.search(line)
            if m:
                phase, ep = int(m.group(1)), int(m.group(2))
                # 阶段1: 1-20, 阶段2: 21-80
                gep = ep if phase == 1 else ep + 20
                last_gep = gep
                rows.append({
                    'phase':   phase,   'epoch': ep,       'gep':     gep,
                    'lr':      float(m.group(3)),
                    'tr_loss': float(m.group(4)),  'tr_acc': float(m.group(5)),
                    'vl_loss': float(m.group(6)),  'vl_acc': float(m.group(7)),
                    'vl_sens': float(m.group(8)),  'vl_spec':float(m.group(9)),
                    'vl_f1':   float(m.group(10)),
                })
            elif CKPT_RE.search(line) and last_gep is not None:
                best_eps.append(last_gep)

    if not rows:
        raise ValueError(f'未解析到任何 epoch 记录，请检查日志路径：{path}')

    d = {k: [r[k] for r in rows] for k in rows[0]}
    d['best_eps'] = best_eps
    return d


def parse_seg_log(path: str) -> dict:
    """解析分割日志（单阶段，60ep）。"""
    EP_RE = re.compile(
        r'Epoch\s+\[(\d+)/\d+\]\s+'
        r'lr=([0-9.e+\-]+)\s*\|\s*'
        r'train\s+loss=([0-9.]+)\s+dice=([0-9.]+)\s*\|\s*'
        r'val\s+loss=([0-9.]+)\s+dice=([0-9.]+)\s+iou=([0-9.]+)'
    )
    CKPT_RE = re.compile(r'保存最优模型')

    rows, best_eps, last_ep = [], [], None

    with open(path, encoding='utf-8') as f:
        for line in f:
            m = EP_RE.search(line)
            if m:
                ep = int(m.group(1))
                last_ep = ep
                rows.append({
                    'epoch':   ep,
                    'lr':      float(m.group(2)),
                    'tr_loss': float(m.group(3)),  'tr_dice': float(m.group(4)),
                    'vl_loss': float(m.group(5)),  'vl_dice': float(m.group(6)),
                    'vl_iou':  float(m.group(7)),
                })
            elif CKPT_RE.search(line) and last_ep is not None:
                best_eps.append(last_ep)

    if not rows:
        raise ValueError(f'未解析到任何 epoch 记录，请检查日志路径：{path}')

    d = {k: [r[k] for r in rows] for k in rows[0]}
    d['best_eps'] = best_eps
    return d


def _phase_bg(ax, p1_end: int, total: int):
    """绘制两阶段背景色块及分界虚线。"""
    ax.axvspan(0.5,           p1_end + 0.5, alpha=0.18, color=C['ph1_bg'], zorder=0)
    ax.axvspan(p1_end + 0.5, total  + 0.5, alpha=0.18, color=C['ph2_bg'], zorder=0)
    ax.axvline(p1_end + 0.5, color='#94A3B8', ls='--', lw=1.0, zorder=1)


def _star_ckpts(ax, ep_list: list, y_list: list, best_eps: list):
    """在保存checkpoint的epoch上标注星号。"""
    added = False
    for be in best_eps:
        try:
            i = ep_list.index(be)
        except ValueError:
            continue
        ax.scatter(be, y_list[i], s=110, color=C['best'], marker='*',
                   zorder=8, label=('Checkpoint' if not added else ''),
                   edgecolors='white', linewidths=0.5)
        added = True


def _annotate_max(ax, ep_list: list, y_list: list, prefix: str = 'Best') -> tuple:
    """标注最大值，返回（最大值, 对应epoch）。"""
    bv  = max(y_list)
    bep = ep_list[y_list.index(bv)]
    ax.axhline(bv, color=C['best'], ls=':', lw=1.0, alpha=0.7, zorder=5)

    x_span = max(ep_list) - min(ep_list) + 1
    y_span = max(y_list)  - min(y_list) + 1e-6

    # 文字偏移方向自动调整
    x_off = -x_span * 0.18 if bep > (min(ep_list) + x_span * 0.6) else x_span * 0.04
    y_off = -y_span * 0.08 if bv  > (min(y_list)  + y_span * 0.7) else y_span * 0.06

    ax.annotate(
        f'{prefix}={bv:.4f}  @ep{bep}',
        xy=(bep, bv), xytext=(bep + x_off, bv + y_off),
        fontsize=7.5, color=C['best'], fontweight='bold',
        arrowprops=dict(arrowstyle='->', color=C['best'], lw=0.8),
    )
    return bv, bep


def _fmt_ax(ax, ep_list: list, ylabel: str = '', ylim: tuple = None,
            legend_kw: dict = None):
    """统一坐标轴格式。"""
    ax.set_xlim(min(ep_list) - 0.5, max(ep_list) + 0.5)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_xlabel('Epoch', fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
    kw = dict(fontsize=8, loc='best')
    if legend_kw:
        kw.update(legend_kw)
    ax.legend(**kw)


def plot_cls(d: dict, out: str = 'cls_training_curves.png'):
    """分类任务2x3宫格可视化。"""
    ep  = d['gep']
    P1  = 20          # Phase1 最后一个epoch
    END = max(ep)     # 80

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        'Classification Training Curves  ·  ResNet-34\n'
        'Phase 1 – Frozen Backbone (ep 1–20)        '
        'Phase 2 – Full Fine-tune (ep 21–80)\n'
        f'Train 1436 / Val 359          '
        f'Best Val Acc = {max(d["vl_acc"]):.4f}    '
        f'Best Val F1 = {max(d["vl_f1"]):.4f}',
        fontsize=11, fontweight='bold', y=1.03,
    )

    # 子图 (0,0)：训练/验证损失
    ax = axes[0, 0]
    _phase_bg(ax, P1, END)
    ax.plot(ep, d['tr_loss'], c=C['train'], lw=1.8, label='Train Loss')
    ax.plot(ep, d['vl_loss'], c=C['val'],   lw=1.8, label='Val Loss')
    ax.fill_between(ep, d['tr_loss'], d['vl_loss'],
                    alpha=0.07, color='#94A3B8', label='Gen. Gap')
    ax.set_title('Loss', fontweight='bold')
    _fmt_ax(ax, ep, ylabel='Cross-Entropy Loss')

    # 子图 (0,1)：准确率 + checkpoint星标
    ax = axes[0, 1]
    _phase_bg(ax, P1, END)
    ax.plot(ep, d['tr_acc'], c=C['train'], lw=1.8, label='Train Acc')
    ax.plot(ep, d['vl_acc'], c=C['val'],   lw=1.8, label='Val Acc')
    _star_ckpts(ax, ep, d['vl_acc'], d['best_eps'])
    bv, be = _annotate_max(ax, ep, d['vl_acc'], prefix='Best Acc')
    ax.set_title(f'Accuracy  (Best Val = {bv:.4f}  @ep {be})', fontweight='bold')
    _fmt_ax(ax, ep, ylabel='Accuracy', ylim=(0.45, 1.05))

    # 子图 (0,2)：学习率（对数坐标）
    ax = axes[0, 2]
    _phase_bg(ax, P1, END)
    ax.semilogy(ep, d['lr'], c=C['lr'], lw=1.8, label='LR')
    ph2_start_lr = d['lr'][P1]
    ax.annotate(
        f'LR restart\n= {ph2_start_lr:.1e}',
        xy=(P1 + 1, ph2_start_lr),
        xytext=(P1 + 6, ph2_start_lr * 3),
        fontsize=7.5, color='#475569',
        arrowprops=dict(arrowstyle='->', lw=0.8, color='#475569'),
    )
    ax.set_title('Learning Rate Schedule  (log)', fontweight='bold')
    _fmt_ax(ax, ep, ylabel='Learning Rate')

    # 子图 (1,0)：验证敏感度与特异度
    ax = axes[1, 0]
    _phase_bg(ax, P1, END)
    ax.plot(ep, d['vl_sens'], c=C['sens'], lw=1.8, label='Sensitivity (Recall+)')
    ax.plot(ep, d['vl_spec'], c=C['spec'], lw=1.8, label='Specificity (Recall−)')
    ax.fill_between(ep, d['vl_sens'], d['vl_spec'], alpha=0.07, color='gray')
    ax.set_title('Val Sensitivity & Specificity', fontweight='bold')
    _fmt_ax(ax, ep, ylabel='Score', ylim=(0.30, 1.05))

    # 子图 (1,1)：验证F1
    ax = axes[1, 1]
    _phase_bg(ax, P1, END)
    ax.plot(ep, d['vl_f1'], c=C['f1'], lw=1.8, label='Val F1')
    ax.fill_between(ep, 0, d['vl_f1'], alpha=0.10, color=C['f1'])
    _annotate_max(ax, ep, d['vl_f1'], prefix='Best F1')
    ax.set_title('Val F1 Score', fontweight='bold')
    _fmt_ax(ax, ep, ylabel='F1', ylim=(0.30, 1.05))

    # 子图 (1,2)：所有验证指标总览
    ax = axes[1, 2]
    _phase_bg(ax, P1, END)
    ax.plot(ep, d['vl_acc'],  c=C['val'],  lw=1.8,             label='Val Acc')
    ax.plot(ep, d['vl_f1'],   c=C['f1'],   lw=1.4, ls='-.',    label='Val F1')
    ax.plot(ep, d['vl_sens'], c=C['sens'], lw=1.2, ls='--',    label='Sensitivity')
    ax.plot(ep, d['vl_spec'], c=C['spec'], lw=1.2, ls=':',     label='Specificity')
    ax.set_title('All Val Metrics Overview', fontweight='bold')
    _fmt_ax(ax, ep, ylabel='Score', ylim=(0.30, 1.05),
            legend_kw=dict(fontsize=7.5, ncol=2, loc='lower right'))

    # 底部阶段图例
    ph1 = mpatches.Patch(fc=C['ph1_bg'], ec='none', label='Phase 1: Frozen backbone (ep 1–20)')
    ph2 = mpatches.Patch(fc=C['ph2_bg'], ec='none', label='Phase 2: Full fine-tune (ep 21–80)')
    fig.legend(handles=[ph1, ph2], loc='lower center', ncol=2,
               fontsize=9, bbox_to_anchor=(0.5, -0.01), framealpha=0.9)

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(out, bbox_inches='tight')
    print(f'  ✓  {out}')
    plt.close()


def plot_seg(d: dict, out: str = 'seg_training_curves.png'):
    """分割任务2x2宫格可视化。"""
    ep = d['epoch']

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        'Segmentation Training Curves  ·  UNet + Attention  (17.6 M params)\n'
        f'Train 2401 / Val 600      60 Epochs      '
        f'Best Val Dice = {max(d["vl_dice"]):.4f}      '
        f'Best IoU = {max(d["vl_iou"]):.4f}',
        fontsize=11, fontweight='bold', y=1.02,
    )

    # 子图 (0,0)：训练/验证损失
    ax = axes[0, 0]
    ax.plot(ep, d['tr_loss'], c=C['train'], lw=1.8, label='Train Loss')
    ax.plot(ep, d['vl_loss'], c=C['val'],   lw=1.8, label='Val Loss')
    ax.fill_between(ep, d['tr_loss'], d['vl_loss'],
                    alpha=0.10, color='#94A3B8', label='Gap')
    ax.set_title('Loss  (Train vs Val)', fontweight='bold')
    _fmt_ax(ax, ep, ylabel='Loss')

    # 子图 (0,1)：训练/验证Dice + checkpoint星标
    ax = axes[0, 1]
    ax.plot(ep, d['tr_dice'], c=C['train'], lw=1.8, label='Train Dice')
    ax.plot(ep, d['vl_dice'], c=C['val'],   lw=1.8, label='Val Dice')
    _star_ckpts(ax, ep, d['vl_dice'], d['best_eps'])
    bv, be = _annotate_max(ax, ep, d['vl_dice'], prefix='Best')
    ax.set_title(f'Dice Score  (Best Val = {bv:.4f}  @ep {be})', fontweight='bold')
    _fmt_ax(ax, ep, ylabel='Dice', ylim=(0.40, 1.00))

    # 子图 (1,0)：验证IoU
    ax = axes[1, 0]
    ax.plot(ep, d['vl_iou'], c=C['iou'], lw=1.8, label='Val IoU')
    ax.fill_between(ep, 0, d['vl_iou'], alpha=0.12, color=C['iou'])
    _annotate_max(ax, ep, d['vl_iou'], prefix='Best IoU')
    ax.set_title('Val IoU', fontweight='bold')
    _fmt_ax(ax, ep, ylabel='IoU (Jaccard)', ylim=(0.25, 0.92))

    # 子图 (1,1)：学习率（对数坐标）
    ax = axes[1, 1]
    ax.semilogy(ep, d['lr'], c=C['lr'], lw=1.8, label='Learning Rate')
    final_lr = d['lr'][-1]
    ax.annotate(
        f'Final LR = {final_lr:.2e}',
        xy=(ep[-1], final_lr),
        xytext=(ep[-1] - len(ep) * 0.3, final_lr * 8),
        fontsize=7.5, color='#475569',
        arrowprops=dict(arrowstyle='->', lw=0.8, color='#475569'),
    )
    ax.set_title('Learning Rate Schedule  (log)', fontweight='bold')
    _fmt_ax(ax, ep, ylabel='LR (log scale)')

    plt.tight_layout()
    plt.savefig(out, bbox_inches='tight')
    print(f'  ✓  {out}')
    plt.close()


def plot_summary(cd: dict, sd: dict, out: str = 'summary.png'):
    """生成双任务性能对比摘要图（三栏布局）。"""
    fig = plt.figure(figsize=(15, 5.5))
    fig.suptitle('Training Summary  ·  Classification  &  Segmentation',
                 fontsize=13, fontweight='bold')

    # 左栏：分类最优epoch的各指标柱状图
    ax1 = fig.add_subplot(1, 3, 1)
    bi = cd['vl_acc'].index(max(cd['vl_acc']))
    names_c = ['Acc', 'Sens', 'Spec', 'F1']
    vals_c  = [cd['vl_acc'][bi], cd['vl_sens'][bi],
               cd['vl_spec'][bi], cd['vl_f1'][bi]]
    clrs_c  = [C['val'], C['sens'], C['spec'], C['f1']]
    bars = ax1.bar(names_c, vals_c, color=clrs_c, alpha=0.85,
                   width=0.55, edgecolor='white')
    for b, v in zip(bars, vals_c):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.012,
                 f'{v:.3f}', ha='center', va='bottom',
                 fontsize=9.5, fontweight='bold')
    ax1.set_ylim(0, 1.15)
    ax1.set_title(
        f'Classification\n(Best-Acc Epoch = ep {cd["gep"][bi]})',
        fontweight='bold')
    ax1.set_ylabel('Score')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # 中栏：分割最优epoch的各指标柱状图
    ax2 = fig.add_subplot(1, 3, 2)
    si = sd['vl_dice'].index(max(sd['vl_dice']))
    names_s = ['Dice', 'IoU']
    vals_s  = [sd['vl_dice'][si], sd['vl_iou'][si]]
    clrs_s  = [C['val'], C['iou']]
    bars = ax2.bar(names_s, vals_s, color=clrs_s, alpha=0.85,
                   width=0.4, edgecolor='white')
    for b, v in zip(bars, vals_s):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.012,
                 f'{v:.4f}', ha='center', va='bottom',
                 fontsize=9.5, fontweight='bold')
    ax2.set_ylim(0, 1.10)
    ax2.set_title(
        f'Segmentation\n(Best-Dice Epoch = ep {sd["epoch"][si]})',
        fontweight='bold')
    ax2.set_ylabel('Score')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # 右栏：文字摘要卡片
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.axis('off')
    ax3.set_title('Experiment Summary Card', fontweight='bold')
    card = (
        "┌──── Classification ─────────────────┐\n"
        f"│  Backbone  :  ResNet-34             │\n"
        f"│  Dataset   :  1436 train / 359 val  │\n"
        f"│  Schedule  :  Phase1 frozen  20 ep  │\n"
        f"│              Phase2 fine-tune 60 ep  │\n"
        f"│  Best Acc  :  {max(cd['vl_acc']):.4f}               │\n"
        f"│  Best F1   :  {max(cd['vl_f1']):.4f}               │\n"
        "├──── Segmentation ───────────────────┤\n"
        f"│  Arch      :  UNet + Attention      │\n"
        f"│  Params    :  17.6 M                │\n"
        f"│  Dataset   :  2401 train / 600 val  │\n"
        f"│  Epochs    :  60                    │\n"
        f"│  Best Dice :  {max(sd['vl_dice']):.4f}               │\n"
        f"│  Best IoU  :  {max(sd['vl_iou']):.4f}               │\n"
        "└─────────────────────────────────────┘"
    )
    ax3.text(0.03, 0.95, card, transform=ax3.transAxes,
             fontsize=8.5, va='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.55', fc='#F1F5F9',
                       ec='#CBD5E1', lw=0.9, alpha=0.95))

    plt.tight_layout()
    plt.savefig(out, bbox_inches='tight')
    print(f'  ✓  {out}')
    plt.close()


def main():
    ap = argparse.ArgumentParser(
        description='从分类/分割日志生成训练曲线图',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument('--cls', default='cls_train.log', help='分类日志路径')
    ap.add_argument('--seg', default='seg_train.log', help='分割日志路径')
    ap.add_argument('--out', default='.', help='输出目录')
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print('-' * 56)
    print('  训练日志可视化工具')
    print('-' * 56)

    # 解析日志
    print('\n[1/3] 正在解析日志...')
    try:
        cls_d = parse_cls_log(args.cls)
    except FileNotFoundError:
        print(f'  ✗  文件不存在：{args.cls}')
        return
    except ValueError as e:
        print(f'  ✗  {e}')
        return

    try:
        seg_d = parse_seg_log(args.seg)
    except FileNotFoundError:
        print(f'  ✗  文件不存在：{args.seg}')
        return
    except ValueError as e:
        print(f'  ✗  {e}')
        return

    print(f'  分类 → {len(cls_d["gep"])} 轮次，最佳准确率 = {max(cls_d["vl_acc"]):.4f}，最佳F1 = {max(cls_d["vl_f1"]):.4f}')
    print(f'  分割 → {len(seg_d["epoch"])} 轮次，最佳Dice = {max(seg_d["vl_dice"]):.4f}，最佳IoU = {max(seg_d["vl_iou"]):.4f}')

    # 生成图像
    print('\n[2/3] 正在生成图表...')
    plot_cls(cls_d, str(out_dir / 'cls_training_curves.png'))
    plot_seg(seg_d, str(out_dir / 'seg_training_curves.png'))

    print('\n[3/3] 正在生成摘要...')
    plot_summary(cls_d, seg_d, str(out_dir / 'summary.png'))

    print('\n' + '-' * 56)
    print('  完成！生成的文件：')
    for name in ('cls_training_curves.png', 'seg_training_curves.png', 'summary.png'):
        print(f'    {out_dir / name}')
    print('-' * 56)


if __name__ == '__main__':
    main()